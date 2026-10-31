"""CPU surface-point extraction for LAS/LAZ clouds; no learned model required."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import tempfile
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceConfig:
    voxel_size: float = 0.5
    xy_grid_size: float = 0.5
    xy_minimum_points: int = 1
    percentile: float = 10.0
    mls_neighbors: int = 32
    mls_radius: float | None = 2.0
    mls_degree: int = 2
    mls_bandwidth: float = 1.0
    mls_minimum_neighbors: int = 6
    local_neighbors: int = 32
    local_radius: float | None = 2.0
    local_bandwidth: float = 1.0
    local_minimum_neighbors: int = 6

    def validate(self):
        for name in ('voxel_size','xy_grid_size','mls_bandwidth','local_bandwidth'):
            _positive(getattr(self,name),name)
        if isinstance(self.percentile,bool) or not np.isfinite(self.percentile) or not 0 <= self.percentile <= 100:
            raise ValueError('percentile must be finite and in [0, 100]')
        _integer(self.xy_minimum_points,'xy_minimum_points',1)
        _fit_parameters(self.mls_neighbors,self.mls_radius,self.mls_bandwidth,
                        self.mls_minimum_neighbors,self.mls_degree)
        _fit_parameters(self.local_neighbors,self.local_radius,self.local_bandwidth,
                        self.local_minimum_neighbors)
        return self


def _positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _integer(value,name,minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _xyz(points):
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected a nonempty finite N x 3 XYZ array')
    return points


def _groups(coordinates,size):
    _positive(size,'grid size')
    scaled=(coordinates-coordinates.min(axis=0))/size
    if not np.isfinite(scaled).all() or np.any(scaled>=2.**63):
        raise ValueError('Grid size is too small for the coordinate range')
    keys=np.floor(scaled).astype(np.int64)
    _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    return inverse,counts


def _group_means(points,inverse,counts):
    origin=points.min(axis=0)
    return np.column_stack([origin[i]+np.bincount(inverse,weights=points[:,i]-origin[i])/counts
                            for i in range(points.shape[1])])


def voxel_downsample(points,voxel_size):
    """One XYZ centroid per occupied 3D voxel, anchored at the cloud minimum."""
    points=_xyz(points)
    inverse,counts=_groups(points,voxel_size)
    return _group_means(points,inverse,counts),counts


def xy_surface(points,grid_size,statistic,*,percentile=10.,minimum_points=1):
    """One mean-XY/statistic-Z point per sufficiently populated XY grid cell."""
    points=_xyz(points)
    _integer(minimum_points,'minimum_points',1)
    if statistic not in ('mean','median','minimum','maximum','percentile'):
        raise ValueError('Unsupported XY elevation statistic')
    if not np.isfinite(percentile) or not 0<=percentile<=100:
        raise ValueError('percentile must be in [0, 100]')
    inverse,counts=_groups(points[:,:2],grid_size)
    result=_group_means(points,inverse,counts)
    if statistic!='mean':
        order=np.argsort(inverse,kind='stable')
        start=np.r_[0,np.cumsum(counts)[:-1]]
        z=points[order,2]
        if statistic=='minimum': result[:,2]=np.minimum.reduceat(z,start)
        elif statistic=='maximum': result[:,2]=np.maximum.reduceat(z,start)
        else:
            quantile=50. if statistic=='median' else percentile
            result[:,2]=[np.percentile(z[a:a+n],quantile,method='linear') for a,n in zip(start,counts)]
    keep=counts>=minimum_points
    return result[keep],counts[keep]


def _fit_parameters(neighbors,radius,bandwidth,minimum_neighbors,degree=None):
    required=3
    if degree is not None:
        if type(degree) is not int or degree not in (1,2):
            raise ValueError('MLS degree must be 1 (plane) or 2 (quadratic)')
        required=3 if degree==1 else 6
    _integer(minimum_neighbors,'minimum_neighbors',required)
    _integer(neighbors,'neighbors',minimum_neighbors)
    _positive(bandwidth,'bandwidth')
    if radius is not None: _positive(radius,'radius')


def _neighborhoods(coordinates,neighbors,radius):
    tree=cKDTree(coordinates)
    k=min(neighbors,len(coordinates))
    bound=np.inf if radius is None else radius
    for start in range(0,len(coordinates),4096):
        distances,indices=tree.query(coordinates[start:start+4096],k=k,distance_upper_bound=bound)
        if k==1: distances,indices=distances[:,None],indices[:,None]
        for offset,(d,ids) in enumerate(zip(distances,indices)):
            valid=np.isfinite(d)&(ids<len(coordinates))
            yield start+offset,d[valid],ids[valid]


def mls_surface(points,*,neighbors=32,radius=2.,degree=2,bandwidth=1.,minimum_neighbors=6):
    """Moving weighted least-squares z=f(x,y), evaluated at each input XY.

    Uses XY neighbors and Gaussian weights exp(-d_xy**2/(2*bandwidth**2)).
    Centered/scaled polynomial coordinates protect large geospatial coordinates.
    This is a 2.5D terrain smoother, not an overhang/multilayer surface model.
    """
    points=_xyz(points)
    _fit_parameters(neighbors,radius,bandwidth,minimum_neighbors,degree)
    output=points.copy(); fallback=0; supports=np.zeros(len(points),dtype=np.int64)
    for i,d,ids in _neighborhoods(points[:,:2],neighbors,radius):
        supports[i]=len(ids)
        if len(ids)<minimum_neighbors:
            fallback+=1; continue
        local=points[ids]-points[i]
        scale=max(float(d.max()),1e-12)
        x,y=(local[:,:2]/scale).T
        columns=[np.ones(len(ids)),x,y]
        if degree==2: columns.extend([x*x,x*y,y*y])
        design=np.column_stack(columns)
        sqrt_weight=np.exp(-.25*(d/bandwidth)**2)
        coefficients,_,rank,_=np.linalg.lstsq(design*sqrt_weight[:,None],local[:,2]*sqrt_weight,rcond=1e-10)
        if rank!=design.shape[1] or not np.isfinite(coefficients).all():
            fallback+=1; continue
        output[i,2]+=coefficients[0]
    return output,{'fallback_points':fallback,'neighbor_counts':supports}


def project_local_planes(points,*,neighbors=32,radius=2.,bandwidth=1.,minimum_neighbors=6):
    """Fit weighted local PCA planes in XYZ and project each query orthogonally."""
    points=_xyz(points)
    _fit_parameters(neighbors,radius,bandwidth,minimum_neighbors)
    output=points.copy(); fallback=0; supports=np.zeros(len(points),dtype=np.int64)
    for i,d,ids in _neighborhoods(points,neighbors,radius):
        supports[i]=len(ids)
        if len(ids)<minimum_neighbors:
            fallback+=1; continue
        local=points[ids]-points[i]
        weight=np.exp(-.5*(d/bandwidth)**2); weight/=weight.sum()
        center=np.sum(local*weight[:,None],axis=0)
        centered=local-center
        covariance=(centered*weight[:,None]).T@centered
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        if eigenvalues[1]<=max(eigenvalues[2]*1e-10,1e-14):
            fallback+=1; continue
        normal=eigenvectors[:,0]
        output[i]+=np.dot(center,normal)*normal
    return output,{'fallback_points':fallback,'neighbor_counts':supports}


SURFACE_METHODS = ('voxel', 'xy_mean', 'xy_median', 'xy_minimum', 'xy_maximum',
                   'xy_percentile', 'mls', 'local_plane')


def validate_methods(methods):
    """Return a nonempty, unique ordered selection of supported methods."""
    if (not isinstance(methods, (list, tuple)) or not methods
            or any(not isinstance(name, str) or name not in SURFACE_METHODS for name in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError(f'methods must be a nonempty list of unique names from {SURFACE_METHODS}')
    return tuple(methods)


def extract_surfaces(points,config,*,methods=None):
    """Apply selected methods independently; None preserves the all-methods default."""
    selected = SURFACE_METHODS if methods is None else validate_methods(methods)
    config.validate(); points=_xyz(points)
    methods={
        'voxel':lambda:voxel_downsample(points,config.voxel_size),
        **{f'xy_{name}':(lambda name=name:xy_surface(points,config.xy_grid_size,name,
              percentile=config.percentile,minimum_points=config.xy_minimum_points))
           for name in ('mean','median','minimum','maximum','percentile')},
        'mls':lambda:mls_surface(points,neighbors=config.mls_neighbors,radius=config.mls_radius,
                    degree=config.mls_degree,bandwidth=config.mls_bandwidth,minimum_neighbors=config.mls_minimum_neighbors),
        'local_plane':lambda:project_local_planes(points,neighbors=config.local_neighbors,radius=config.local_radius,
                    bandwidth=config.local_bandwidth,minimum_neighbors=config.local_minimum_neighbors),
    }
    for name in selected:
        method = methods[name]
        start=time.perf_counter(); xyz,details=method()
        if not len(xyz):
            raise ValueError(f'{name}: no cells meet the minimum population; reduce xy_minimum_points')
        _xyz(xyz)
        details=details if isinstance(details,dict) else {'support_counts':details,'fallback_points':0}
        yield name,xyz,details,time.perf_counter()-start


REFERENCE_FIELDS=('ref_inside_fraction','ref_dz_mean','ref_dz_mae','ref_dz_rmse','ref_dz_p95_abs',
                  'ref_coverage_mean','ref_coverage_p95')


def reference_scorer(reference):
    """Return score(xyz) comparing a surface with measured reference points (e.g. source ground LiDAR).

    dz is surface Z minus a linear TIN of the reference at each surface XY; points outside
    the reference convex hull are excluded and reported through ref_inside_fraction.
    Coverage is the 3D distance from each reference point to its nearest surface point.
    """
    from scipy.interpolate import LinearNDInterpolator
    reference=_xyz(reference); origin=reference.mean(axis=0)
    tin=LinearNDInterpolator(reference[:,:2]-origin[:2],reference[:,2]-origin[2])
    def score(xyz):
        xyz=_xyz(xyz)
        dz=(xyz[:,2]-origin[2])-tin(xyz[:,:2]-origin[:2]); dz=dz[np.isfinite(dz)]
        coverage=cKDTree(xyz).query(reference,k=1)[0]
        result={'ref_inside_fraction':len(dz)/len(xyz),'ref_coverage_mean':float(coverage.mean()),
                'ref_coverage_p95':float(np.percentile(coverage,95))}
        if not len(dz): return {**dict.fromkeys(REFERENCE_FIELDS),**result}
        return {**result,'ref_dz_mean':float(dz.mean()),'ref_dz_mae':float(np.abs(dz).mean()),
                'ref_dz_rmse':float(np.sqrt(np.mean(dz**2))),'ref_dz_p95_abs':float(np.percentile(np.abs(dz),95))}
    return score


def _reference_file(reference_root,relative,suffix):
    stem=relative.stem[:-len(suffix)] if suffix and relative.stem.endswith(suffix) else relative.stem
    for extension in ('.las','.laz'):
        candidate=reference_root/relative.parent/(stem+extension)
        if candidate.is_file(): return candidate
    raise FileNotFoundError(f'No reference cloud for {relative} under {reference_root}')


def _save_las(path,xyz,source_header):
    scales=np.asarray(source_header.scales); offsets=np.asarray(source_header.offsets)
    if not np.isfinite(scales).all() or (scales<=0).any() or not np.isfinite(offsets).all():
        raise ValueError('Invalid source LAS scales/offsets')
    integers=np.rint((xyz-offsets)/scales)
    if np.any(integers<np.iinfo(np.int32).min) or np.any(integers>np.iinfo(np.int32).max):
        raise OverflowError('Surface coordinates exceed source LAS integer range')
    header=laspy.LasHeader(point_format=6,version='1.4')
    header.scales=scales.copy();header.offsets=offsets.copy()
    header.vlrs=laspy.vlrs.vlrlist.VLRList([deepcopy(v) for v in source_header.vlrs if v.user_id=='LASF_Projection'])
    header.evlrs=laspy.vlrs.vlrlist.VLRList([deepcopy(v) for v in (source_header.evlrs or []) if v.user_id=='LASF_Projection'])
    header.global_encoding.wkt=source_header.global_encoding.wkt
    header.generating_software='Surface extraction'
    cloud=laspy.LasData(header); cloud.x,cloud.y,cloud.z=xyz.T
    with path.open('xb') as stream: cloud.write(stream,do_compress=False)


def comparison_figure(original,surfaces,*,max_points=20000):
    """Nine common-axis 3D panels; display subsampling never changes exports."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    _integer(max_points,'max_points',1)
    panels=[('Input reconstructed cloud',original),*surfaces.items()]
    lower=np.min([a.min(axis=0) for _,a in panels],axis=0)
    upper=np.max([a.max(axis=0) for _,a in panels],axis=0)
    origin=original.mean(axis=0);span=upper-lower
    padding=np.maximum(span*.03,max(float(span.max()),1.)*.001)
    norm=Normalize(lower[2],upper[2] if upper[2]>lower[2] else lower[2]+1)
    fig=plt.figure(figsize=(17,14),layout='constrained'); axes=[]
    rng=np.random.default_rng(42)
    for index,(name,points) in enumerate(panels,1):
        axis=fig.add_subplot(3,3,index,projection='3d'); axes.append(axis)
        shown=points if len(points)<=max_points else points[rng.choice(len(points),max_points,replace=False)]
        local=shown-origin
        artist=axis.scatter(*local.T,c=shown[:,2],cmap='terrain',norm=norm,s=2,alpha=.8,rasterized=True)
        axis.set(xlim=(lower[0]-origin[0]-padding[0],upper[0]-origin[0]+padding[0]),
                 ylim=(lower[1]-origin[1]-padding[1],upper[1]-origin[1]+padding[1]),
                 zlim=(lower[2]-origin[2]-padding[2],upper[2]-origin[2]+padding[2]),
                 xlabel='X offset',ylabel='Y offset',zlabel='Z offset',title=f'{name}: {len(points):,} points')
        axis.set_box_aspect(span+2*padding);axis.view_init(elev=25,azim=-60)
    fig.colorbar(artist,ax=axes,shrink=.5,label='Elevation Z (source units)')
    fig.suptitle('Independent methods on the same input • common axes and elevation scale')
    return fig


def _json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run_surface_pipeline(input_path,output_dir,config=None,*,source_glob='*_first_predictions.las',
                         limit=None,plot_units=3,plot_max_points=20000,
                         reference_root=None,reference_suffix='_first_predictions',methods=None,
                         mirror_source=False):
    """Snapshot selected files and export selected surfaces (all eight by default).

    With reference_root, each input <dir>/<stem><reference_suffix>.las is also scored
    against reference_root/<dir>/<stem>.las (see reference_scorer).
    mirror_source requires one method and writes <dir>/<stem>_surface.las/.npz.
    """
    config=(config or SurfaceConfig()).validate()
    methods = SURFACE_METHODS if methods is None else validate_methods(methods)
    if mirror_source and len(methods) != 1:
        raise ValueError('mirror_source requires exactly one surface method')
    source=Path(input_path).expanduser().resolve(); output=Path(output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(f'Choose a new output directory: {output}')
    if not source.exists(): raise FileNotFoundError(source)
    if reference_root is not None:
        reference_root=Path(reference_root).expanduser().resolve()
        if not reference_root.is_dir(): raise FileNotFoundError(reference_root)
    if limit is not None: _integer(limit,'limit',1)
    _integer(plot_units,'plot_units',0);_integer(plot_max_points,'plot_max_points',1)
    if source.is_file():
        if source.suffix.lower() not in ('.las','.laz'): raise ValueError('Input must be LAS/LAZ')
        files=[source];root=source.parent
    else:
        root=source
        files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.las','.laz')
                     and p.relative_to(root).match(source_glob))
    if limit is not None: files=files[:limit]
    if not files: raise ValueError('No matching LAS/LAZ input files')
    if mirror_source:
        targets = [p.relative_to(root).with_suffix('') for p in files]
        if len(set(targets)) != len(targets):
            raise ValueError('Input files with the same stem would overwrite the same surface output')
    snapshot=[{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files]
    manifest={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'input_path':str(source),
              'input_root':str(root),'output_dir':str(output),'parameters':asdict(config),'source_glob':source_glob,
              'methods':list(methods),
              'mirror_source':mirror_source,
              'limit':limit,'files':snapshot,'counts':{'selected':len(files),'processed':0,'failed':0},
              'units':[],'failures':[],
              'reference_root':None if reference_root is None else str(reference_root),
              'reference_suffix':reference_suffix,
              'metric_note':'Distances to input measure geometric change, not surface accuracy. '
                            'ref_* columns compare with the reference cloud: dz = surface Z minus reference TIN Z.',
              'coordinate_units':'Source CRS units; no reprojection. LAS coordinates quantized to source precision.'}
    output.mkdir(parents=True,exist_ok=False);_json(output/'manifest.json',manifest)
    fields=['source_file','method','input_points','output_points','retained_fraction','fallback_points',
            'z_min','z_max','nearest_input_mean','nearest_input_p95','paired_displacement_rmse','seconds','las_file','npz_file',
            *REFERENCE_FIELDS]
    started=time.perf_counter()
    try:
        with (output/'summary.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for position,(path,snapshot_entry) in enumerate(zip(files,snapshot),1):
                relative=path.relative_to(root); published=None
                try:
                    stat=path.stat()
                    if stat.st_size!=snapshot_entry['size'] or stat.st_mtime_ns!=snapshot_entry['mtime_ns']:
                        raise RuntimeError('Input changed after snapshot; rerun when reconstruction completes')
                    cloud=laspy.read(path);points=_xyz(cloud.xyz);tree=cKDTree(points)
                    reference_path=scorer=input_reference=None
                    if reference_root is not None:
                        reference_path=_reference_file(reference_root,relative,reference_suffix)
                        scorer=reference_scorer(laspy.read(reference_path).xyz);input_reference=scorer(points)
                    unit_relative=relative.parent if mirror_source else Path('surfaces')/relative.parent/relative.stem
                    records=[];surfaces={}
                    with tempfile.TemporaryDirectory(prefix='.pending_',dir=output) as temporary:
                        staged=Path(temporary)/'unit';staged.mkdir()
                        for name,xyz,details,seconds in extract_surfaces(points,config,methods=methods):
                            basename=f'{relative.stem}_surface' if mirror_source else name
                            _save_las(staged/f'{basename}.las',xyz,cloud.header)
                            np.savez_compressed(staged/f'{basename}.npz',xyz=xyz,**details)
                            distance=tree.query(xyz,k=1)[0]
                            record={'source_file':relative.as_posix(),'method':name,'input_points':len(points),
                                    'output_points':len(xyz),'retained_fraction':len(xyz)/len(points),
                                    'fallback_points':details['fallback_points'],'z_min':float(xyz[:,2].min()),
                                    'z_max':float(xyz[:,2].max()),'nearest_input_mean':float(distance.mean()),
                                    'nearest_input_p95':float(np.percentile(distance,95)),
                                    'paired_displacement_rmse':float(np.sqrt(np.mean(np.sum((xyz-points)**2,axis=1))))
                                        if name in ('mls','local_plane') else None,
                                    'seconds':round(seconds,4),'las_file':(unit_relative/f'{basename}.las').as_posix(),
                                    'npz_file':(unit_relative/f'{basename}.npz').as_posix(),
                                    **(scorer(xyz) if scorer else dict.fromkeys(REFERENCE_FIELDS))}
                            records.append(record);surfaces[name]=xyz
                        comparison_file=None
                        if manifest['counts']['processed']<plot_units:
                            import matplotlib.pyplot as plt
                            fig=comparison_figure(points,surfaces,max_points=plot_max_points)
                            figure_name=f'{relative.stem}_surface_comparison.png' if mirror_source else 'comparison.png'
                            try: fig.savefig(staged/figure_name,dpi=130)
                            finally: plt.close(fig)
                            comparison_file=(unit_relative/figure_name).as_posix()
                        destination=output/unit_relative;destination.parent.mkdir(parents=True,exist_ok=True)
                        if mirror_source:
                            destination.mkdir(parents=True,exist_ok=True)
                            created=[]
                            try:
                                for artifact in staged.iterdir():
                                    target=destination/artifact.name
                                    # Same filesystem; publish without overwriting existing files.
                                    target.hardlink_to(artifact)
                                    created.append(target)
                            except BaseException:
                                for target in created: target.unlink()
                                raise
                            published=destination
                        else:
                            if destination.exists(): raise FileExistsError(destination)
                            staged.rename(destination);published=destination
                    unit={'source_file':relative.as_posix(),'methods':records,'comparison_file':comparison_file,
                          'reference_file':None if reference_path is None else str(reference_path),
                          'input_reference':input_reference}
                    manifest['units'].append(unit);manifest['counts']['processed']+=1
                    writer.writerows(records);stream.flush()
                except Exception as exc:
                    if published is not None: raise
                    failure={'source_file':relative.as_posix(),'reason':f'{type(exc).__name__}: {exc}'}
                    manifest['failures'].append(failure);manifest['counts']['failed']+=1
                    print('FAILED',relative,failure['reason'],flush=True)
                manifest['elapsed_seconds']=round(time.perf_counter()-started,2)
                _json(output/'manifest.json',manifest)
                if position==1 or position%max(10,len(files)//20)==0 or position==len(files):
                    print(f'{position}/{len(files)}: {manifest["counts"]}; {manifest["elapsed_seconds"]:.1f}s',flush=True)
        manifest['status']='complete' if not manifest['counts']['failed'] else 'completed_with_errors'
    except BaseException as exc:
        manifest['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        manifest['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        manifest['elapsed_seconds']=round(time.perf_counter()-started,2)
        manifest['finished_at']=datetime.now(timezone.utc).isoformat()
        _json(output/'manifest.json',manifest)
        _json(output/'failures.json',manifest['failures'])
    return manifest
