"""Inference and LAS I/O for the ratio-grouped XY Point2MAE training notebook.

Only model loading/inference requires PyTorch/CUDA. Temporary hooks expose the
existing masked reconstruction without editing the upstream training model.
"""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

CENTROID_ONLY = 'subtract cloud XYZ centroid only; no rescaling'
XY_RADIUS = 'subtract cloud XYZ mean, divide by max XY radius'


def _xyz_array(xyz, minimum=1):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < minimum:
        raise ValueError(f'Expected at least {minimum} points with shape [N, 3].')
    if not np.isfinite(xyz).all():
        raise ValueError('Point coordinates must all be finite.')
    return xyz


def prepare_cloud(xyz, normalization):
    """Replay saved training preprocessing; return float32 points, centroid, scale."""
    xyz = _xyz_array(xyz, minimum=8)
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    if normalization == CENTROID_ONLY:
        scale = 1.0
    elif normalization == XY_RADIUS:
        scale = float(np.linalg.norm(centered[:, :2], axis=1).max())
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError('Radius-scaled checkpoint requires nonzero horizontal extent.')
    else:
        raise ValueError(f'Unknown checkpoint normalization: {normalization!r}')
    points = np.ascontiguousarray(centered / scale, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError('Preprocessed points cannot be represented as finite float32.')
    return points, centroid, scale


def restore_coordinates(points, centroid, scale):
    """Undo cloud preprocessing after patch-centre offsets have been restored."""
    points = _xyz_array(points)
    centroid = np.asarray(centroid, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid centroid or scale.')
    return _xyz_array(points * scale + centroid)


def load_checkpoint_model(path, model_root, cuda_device=0, *, mask_ratio=None):
    """Strictly load a best.pt/last.pt produced by Point2MAE_training.ipynb.

    Uses embedded architecture and preprocessing; fixed-count or non-XY
    checkpoints are rejected rather than silently changing their behavior.
    Returns (model, run_config, checkpoint_info).
    mask_ratio overrides inference masking only; None uses the saved ratio.
    """
    if mask_ratio is not None and (type(mask_ratio) not in (int, float)
                                  or not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1):
        raise ValueError('mask_ratio must be a finite number strictly between 0 and 1, or None.')
    import torch
    from torch import nn
    from point2mae_grouping import hierarchy_for_cloud, configure_group_dividers

    path = Path(path).expanduser().resolve()
    model_root = Path(model_root).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    if not (model_root / 'models/Point_M2AE.py').is_file():
        raise FileNotFoundError(f'Local Point-M2AE source not found in {model_root}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'config', 'model_state_dict'} <= checkpoint.keys():
        raise ValueError('Expected a training-notebook checkpoint containing config and model_state_dict.')
    run_config = deepcopy(checkpoint['config'])
    if not isinstance(run_config, dict) or not isinstance(run_config.get('model'), dict):
        raise ValueError('Checkpoint is missing its model configuration.')
    if run_config.get('positional_coordinates') != ['x', 'y']:
        raise ValueError('This notebook supports XY positional checkpoints from Point2MAE_training.ipynb.')
    if run_config.get('normalization') not in (CENTROID_ONLY, XY_RADIUS):
        raise ValueError('Checkpoint normalization is missing or unsupported; do not guess preprocessing.')
    config = SimpleNamespace(**run_config['model'])
    if not hasattr(config, 'group_divisors'):
        raise ValueError('This checkpoint uses fixed counts; a ratio-grouped training checkpoint is required.')
    initial_groups, _ = hierarchy_for_cloud(8, config.group_divisors, config.group_sizes, config.mask_ratio)
    checkpoint_mask_ratio = config.mask_ratio
    if mask_ratio is not None:
        config.mask_ratio = float(mask_ratio)
        run_config['model']['mask_ratio'] = config.mask_ratio
    # Initial counts are placeholders from the saved configuration. Validate
    # the override against each actual cloud in forward, not an 8-point dummy.
    if len(config.encoder_dims) != 3 or len(config.decoder_dims) != 2:
        raise ValueError('Reconstruction requires the three-encoder/two-decoder-scale architecture.')
    if not torch.cuda.is_available():
        raise RuntimeError('Point-M2AE reconstruction requires the configured CUDA GPU environment.')
    if type(cuda_device) is not int or not 0 <= cuda_device < torch.cuda.device_count():
        raise ValueError('CUDA device index is not available.')
    torch.cuda.set_device(cuda_device)
    device = torch.device('cuda', cuda_device)
    sys.path.insert(0, str(model_root))
    import models.Point_M2AE as local_model_module
    if Path(local_model_module.__file__).resolve() != model_root / 'models/Point_M2AE.py':
        raise RuntimeError('Another models package is cached; restart the kernel.')

    class XYPositionEmbedding(nn.Sequential):
        def __init__(self, width):
            super().__init__(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))

        def forward(self, centers):
            return super().forward(centers[..., :2])

    class ReconstructionModel(local_model_module.Point_M2AE):
        def __init__(self):
            super().__init__(SimpleNamespace(**{**vars(config), 'num_groups': initial_groups}))
            # Same module names and shapes as the training notebook; all weights
            # and BatchNorm running statistics are supplied by strict loading.
            self.h_encoder.encoder_pos_embeds = nn.ModuleList([
                XYPositionEmbedding(width) for width in config.encoder_dims])
            self.decoder_pos_embeds = nn.ModuleList([
                XYPositionEmbedding(width) for width in config.decoder_dims])

        def forward(self, pts, eval=False, **kwargs):
            if pts.ndim != 3 or pts.shape[0] != 1 or pts.shape[-1] != 3:
                raise ValueError('Expected one full cloud with shape [1, N, 3].')
            configure_group_dividers(self.group_dividers, pts.shape[1], config.group_divisors,
                                     config.group_sizes, config.mask_ratio)
            return super().forward(pts, eval=eval, **kwargs)

    model = ReconstructionModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    info = {'path': str(path), 'epoch': checkpoint.get('epoch'),
            'checkpoint_mask_ratio': checkpoint_mask_ratio,
            'inference_mask_ratio': config.mask_ratio,
            'best_validation_chamfer': checkpoint.get('best_validation_chamfer')}
    return model, run_config, info


def predict_masked_patches(model, points):
    """Capture one ordinary reconstruction forward, returning cloud-relative patches.

    points: [1, N, 3] on the model's device. Caller sets eval/no-grad mode.
    Patch IDs refer to the middle scale, where the supplied decoder predicts.
    """
    import torch

    captured = {}

    def capture_centers(module, inputs, output):
        captured['centers'] = output[1][0]

    def capture_mask(module, inputs, output):
        # Store the tensor now: upstream reverses the mask list after this hook.
        captured['mask'] = output[2][1][0]

    def capture_predictions(module, inputs, output):
        captured['relative'] = output.reshape(output.shape[0], -1, 3)

    handles = []
    try:
        handles.append(model.group_dividers[1].register_forward_hook(capture_centers))
        handles.append(model.h_encoder.register_forward_hook(capture_mask))
        handles.append(model.rec_head.register_forward_hook(capture_predictions))
        loss = model(points, eval=False)
        centers, mask, relative = captured['centers'], captured['mask'], captured['relative']
        patch_ids = torch.nonzero(mask, as_tuple=False).flatten()
        if len(patch_ids) == 0 or relative.shape[0] != len(patch_ids):
            raise RuntimeError('No masked reconstruction targets or incompatible decoder output.')
        patches = relative + centers[mask].unsqueeze(1)
        if not torch.isfinite(patches).all().item() or not torch.isfinite(loss).item():
            raise FloatingPointError('Checkpoint produced nonfinite predictions or loss.')
        return {'patches': patches, 'patch_ids': patch_ids, 'centers': centers, 'loss': float(loss.item())}
    finally:
        for handle in handles:
            handle.remove()


def reconstruct_cloud(model, points, passes=8, seed=42):
    """Predict masked patches repeatedly, keeping the first prediction per patch.

    Returns an XYZ array in preprocessed cloud coordinates plus coverage/loss
    metadata. No original points are appended. Coverage need not reach 100%:
    upstream mask propagation always exposes middle-level centre zero.
    """
    import torch

    if type(passes) is not int or passes < 1:
        raise ValueError('passes must be a positive integer.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    points = _xyz_array(points, minimum=8)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('The supplied model must be on a CUDA device.')
    tensor = torch.as_tensor(np.ascontiguousarray(points, dtype=np.float32), device=device).unsqueeze(0)
    prior_mode = model.training
    numpy_state = np.random.get_state()
    model.eval()
    all_patches, covered, losses = [], set(), []
    reference_centers = None
    try:
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            for _ in range(passes):
                result = predict_masked_patches(model, tensor)
                centers = result['centers'].cpu().numpy()
                if reference_centers is None:
                    reference_centers = centers.copy()
                elif not np.array_equal(reference_centers, centers):
                    raise RuntimeError('Grouping centres changed between passes; patch IDs cannot be merged safely.')
                ids = result['patch_ids'].cpu().numpy()
                new = np.array([int(index) not in covered for index in ids])
                if new.any():
                    all_patches.append(result['patches'].cpu().numpy()[new].reshape(-1, 3))
                    covered.update(map(int, ids[new]))
                losses.append(result['loss'])
    finally:
        np.random.set_state(numpy_state)
        model.train(prior_mode)
    if not all_patches:
        raise RuntimeError('No patches were reconstructed.')
    reconstruction = np.concatenate(all_patches)
    report = {'passes': passes, 'seed': seed, 'original_points': len(points),
              'predicted_points': len(reconstruction), 'covered_middle_patches': len(covered),
              'total_middle_patches': len(reference_centers),
              'patch_coverage_fraction': len(covered) / len(reference_centers),
              'covered_patch_ids': sorted(covered),
              'mean_masked_chamfer_model_units': float(np.mean(losses)),
              'loss_per_pass': losses,
              'output_kind': 'predicted masked patches only; first prediction per middle patch'}
    return reconstruction, report


def save_reconstruction_las(path, xyz, source_header, source_path):
    """Write new XYZ points with source scales/offsets and CRS records, exclusively.

    Other point attributes use zero defaults: predicted points have no measured
    intensity, classification, return number, RGB or timestamp correspondence.
    """
    import laspy
    from laspy.vlrs.vlrlist import VLRList

    path = Path(path).expanduser().resolve()
    if path == Path(source_path).expanduser().resolve():
        raise ValueError('Output must differ from the source LAS path.')
    if path.suffix.lower() != '.las':
        raise ValueError('Choose a .las output filename.')
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    xyz = _xyz_array(xyz)
    scales, offsets = np.asarray(source_header.scales), np.asarray(source_header.offsets)
    if not np.isfinite(scales).all() or (scales <= 0).any() or not np.isfinite(offsets).all():
        raise ValueError('Source LAS has invalid coordinate scales or offsets.')
    integers = np.rint((xyz - offsets) / scales)
    bounds = np.iinfo(np.int32)
    if (integers < bounds.min).any() or (integers > bounds.max).any():
        raise ValueError('Predicted coordinates exceed LAS int32 range at the source scales/offsets.')
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.scales = scales.copy()
    header.offsets = offsets.copy()
    header.vlrs = VLRList([deepcopy(v) for v in source_header.vlrs if v.user_id == 'LASF_Projection'])
    header.evlrs = VLRList([deepcopy(v) for v in (source_header.evlrs or []) if v.user_id == 'LASF_Projection'])
    header.global_encoding.wkt = source_header.global_encoding.wkt
    header.generating_software = 'Point-M2AE reconstruction'
    output = laspy.LasData(header)
    output.points = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
    output.x, output.y, output.z = xyz.T
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a notebook rerun from overwriting a prior result.
    with path.open('xb') as stream:
        output.write(stream, do_compress=False)
    return path


def plot_comparison(original, reconstructed, max_points=100000, seed=42):
    """Side-by-side 3D scatter with shared limits and elevation colours."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    original, reconstructed = _xyz_array(original), _xyz_array(reconstructed)
    if type(max_points) is not int or max_points < 1:
        raise ValueError('max_points must be a positive integer.')
    origin = original.mean(axis=0)
    lower = np.minimum(original.min(axis=0), reconstructed.min(axis=0)) - origin
    upper = np.maximum(original.max(axis=0), reconstructed.max(axis=0)) - origin
    spans = upper - lower
    padding = np.maximum(spans * .03, max(float(spans.max()), 1.) * .001)
    vmin = min(original[:, 2].min(), reconstructed[:, 2].min())
    vmax = max(original[:, 2].max(), reconstructed[:, 2].max())
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.)
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(15, 6), layout='constrained')
    axes = [fig.add_subplot(1, 2, i + 1, projection='3d') for i in range(2)]
    for ax, xyz, title in zip(axes, (original, reconstructed), ('Original LAS', 'Reconstructed patches')):
        sample = xyz if len(xyz) <= max_points else xyz[rng.choice(len(xyz), max_points, replace=False)]
        local = sample - origin
        artist = ax.scatter(*local.T, c=sample[:, 2], cmap='viridis', norm=norm, s=1, alpha=.8)
        ax.set(xlim=(lower[0]-padding[0], upper[0]+padding[0]),
               ylim=(lower[1]-padding[1], upper[1]+padding[1]),
               zlim=(lower[2]-padding[2], upper[2]+padding[2]),
               xlabel='X offset', ylabel='Y offset', zlabel='Z offset',
               title=f'{title}: {len(xyz):,} points ({len(sample):,} shown)')
        ax.set_box_aspect(spans + 2 * padding)
        ax.view_init(elev=25, azim=-60)
    fig.colorbar(artist, ax=axes, shrink=.65, label='Elevation Z (source coordinate units)')
    fig.suptitle('Common axes relative to the input centroid; LAS export uses original coordinates')
    return fig
