"""Export generated slope-unit points with world coordinates and provenance."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import laspy
import numpy as np


def _provenance_array(values, count: int, name: str, dtype, *, minimum: int = 0) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,):
        raise ValueError(f"{name} must contain one value per XYZ point")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} values must be integers")
    maximum = np.iinfo(dtype).max
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{name} values must be between {minimum} and {maximum}")
    return array.astype(dtype, copy=False)


def write_point_clouds(
    xyz,
    source_header: laspy.LasHeader,
    las_path: str | Path,
    ply_path: str | Path,
    su_id: int,
    patch_ids=None,
    pass_ids=None,
    iteration_ids=None,
    voxel_counts=None,
    support_counts=None,
) -> dict[str, str]:
    """Write fresh LAS 1.4 and ASCII PLY files; refuse existing artifacts.

    XYZ is expressed in the source's global coordinate system. LAS retains
    the source coordinate scales, offsets, and compound CRS. Generated points
    carry per-point provenance for raw predictions, voxel counts, or matched
    patch support counts. Input labels are never copied. PLY stores double XYZ
    with enough significant digits for float64 round trips.
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("xyz must be a nonempty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("xyz must contain only finite coordinates")
    if isinstance(su_id, bool) or not isinstance(su_id, Integral) or not 0 <= su_id <= np.iinfo(np.uint32).max:
        raise ValueError("SU_id must be an integer between 0 and 4294967295")
    voxel_representative = voxel_counts is not None
    matched_representative = support_counts is not None
    if voxel_representative and matched_representative:
        raise ValueError('voxel_counts and support_counts cannot be combined')
    if voxel_representative:
        if patch_ids is not None or pass_ids is not None or iteration_ids is not None:
            raise ValueError('voxel_counts cannot be combined with patch, pass or iteration IDs')
        count_values = _provenance_array(voxel_counts, len(points), 'voxel_count', np.uint32, minimum=1)
    elif matched_representative:
        if patch_ids is None or pass_ids is not None or iteration_ids is not None:
            raise ValueError('support_counts require patch_ids and exclude pass and iteration IDs')
        patch_values = _provenance_array(patch_ids, len(points), 'patch_id', np.uint32)
        support_values = _provenance_array(support_counts, len(points), 'support_count', np.uint32, minimum=1)
    else:
        if patch_ids is None or pass_ids is None:
            raise ValueError('Raw predictions require patch_ids and pass_ids')
        patch_values = _provenance_array(patch_ids, len(points), 'patch_id', np.uint32)
        pass_values = _provenance_array(
            pass_ids, len(points), 'reconstruction_pass', np.uint16,
            minimum=1 if iteration_ids is not None else 0,
        )
        if iteration_ids is not None:
            iteration_values = _provenance_array(iteration_ids, len(points), 'iteration_id', np.uint32, minimum=1)
    las_path, ply_path = Path(las_path), Path(ply_path)
    if las_path.resolve() == ply_path.resolve():
        raise ValueError("LAS and PLY require distinct output paths")
    for path in (las_path, ply_path):
        if path.exists():
            raise FileExistsError(f"Output artifact already exists: {path}")

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray(source_header.scales, dtype=np.float64).copy()
    header.offsets = np.asarray(source_header.offsets, dtype=np.float64).copy()
    source_crs = source_header.parse_crs()
    if source_crs is not None:
        header.add_crs(source_crs)
    extra_dims = [laspy.ExtraBytesParams(name='SU_id', type=np.uint32)]
    if voxel_representative:
        extra_dims.append(laspy.ExtraBytesParams(name='voxel_count', type=np.uint32))
    elif matched_representative:
        extra_dims.extend([
            laspy.ExtraBytesParams(name='patch_id', type=np.uint32),
            laspy.ExtraBytesParams(name='support_count', type=np.uint32),
        ])
    else:
        extra_dims.extend([
            laspy.ExtraBytesParams(name='patch_id', type=np.uint32),
            laspy.ExtraBytesParams(name='reconstruction_pass', type=np.uint16),
        ])
        if iteration_ids is not None:
            extra_dims.append(laspy.ExtraBytesParams(name='iteration_id', type=np.uint32))
    header.add_extra_dims(extra_dims)
    cloud = laspy.LasData(header)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
    # Assign before creating files so LAS integer-coordinate overflow leaves
    # no apparently valid artifact on disk.
    cloud.x, cloud.y, cloud.z = points.T
    cloud["SU_id"] = np.full(len(points), su_id, dtype=np.uint32)
    if voxel_representative:
        cloud['voxel_count'] = count_values
    elif matched_representative:
        cloud['patch_id'] = patch_values
        cloud['support_count'] = support_values
    else:
        cloud['patch_id'] = patch_values
        cloud['reconstruction_pass'] = pass_values
        if iteration_ids is not None:
            cloud['iteration_id'] = iteration_values

    las_path.parent.mkdir(parents=True, exist_ok=True)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Exclusive creation also protects against a file appearing after
        # the initial checks.
        with las_path.open("xb") as stream:
            created.append(las_path)
            cloud.write(stream, do_compress=False)
        with ply_path.open("x", encoding="ascii", newline="\n") as stream:
            created.append(ply_path)
            stream.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(points)}\n"
                "property double x\nproperty double y\nproperty double z\nend_header\n"
            )
            np.savetxt(stream, points, fmt="%.17g %.17g %.17g")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {"las": str(las_path), "ply": str(ply_path)}
