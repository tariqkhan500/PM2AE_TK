"""Measure only the saved training split; never alter input clouds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    manifest = args.run / 'train.csv'
    rows = list(csv.DictReader(manifest.open()))
    output = args.run / 'cloud_statistics'
    output.mkdir(exist_ok=True)
    measured = []
    crs_counts = {}
    rng = np.random.default_rng(42)
    for number, row in enumerate(rows, 1):
        cloud = laspy.read(row['path'])
        xyz = np.asarray(cloud.xyz, dtype=np.float64)
        n = len(xyz)
        assert n == int(row['point_count']), row['path']
        assert np.isfinite(xyz).all(), row['path']
        crs = cloud.header.parse_crs()
        if crs is None:
            raise ValueError(f'Missing CRS: {row["path"]}')
        horizontal = crs.sub_crs_list[0] if crs.is_compound else crs
        assert horizontal.is_projected, row['path']
        assert all(abs(a.unit_conversion_factor - 1) < 1e-12 for a in horizontal.axis_info[:2]), row['path']
        crs_name = crs.name
        crs_counts[crs_name] = crs_counts.get(crs_name, 0) + 1
        xyz -= xyz.mean(axis=0)
        xy = xyz[:, :2]
        extent = np.ptp(xyz, axis=0)
        bbox_area = extent[0] * extent[1]
        try:
            hull_area = ConvexHull(xy).volume  # In 2D, volume is area.
        except QhullError:
            hull_area = float('nan')
        sample = rng.choice(n, min(64, n), replace=False)
        nn = cKDTree(xy).query(xy[sample], k=2)[0][:, 1]
        ks = [k for k in (8, 16, 32) if k <= n]
        distances = cKDTree(xyz).query(xyz[sample], k=ks)[0]
        result = dict(unit_key=row['unit_key'], district=row['district'], points=n,
                      xy_bbox_area_m2=float(bbox_area), xy_hull_area_m2=float(hull_area),
                      bbox_density_pts_m2=float(n / bbox_area) if bbox_area > 0 else float('nan'),
                      hull_density_pts_m2=float(n / hull_area) if hull_area > 0 else float('nan'),
                      width_x_m=float(extent[0]), width_y_m=float(extent[1]),
                      height_range_m=float(extent[2]),
                      normalization_xy_radius_m=float(np.linalg.norm(xy, axis=1).max()),
                      sampled_median_xy_nn_m=float(np.median(nn)))
        for k in (8, 16, 32):
            result[f'sampled_median_xyz_patch{k}_radius_m'] = (
                float(np.median(distances[:, ks.index(k)])) if k in ks else float('nan'))
        measured.append(result)
        if number % 2000 == 0:
            print(f'Measured {number}/{len(rows)} training clouds', flush=True)
    with (output / 'per_cloud.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)

    def stats(values):
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a)]
        if not len(a):
            return {'valid_clouds': 0}
        names = ['min', 'p1', 'p5', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max']
        result = dict(zip(names, map(float, np.percentile(a, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]))))
        result.update(mean=float(a.mean()), std_population=float(a.std()), valid_clouds=len(a))
        return result

    metrics = {key: stats([r[key] for r in measured]) for key in measured[0] if key not in ('unit_key', 'district')}
    total_points = sum(r['points'] for r in measured)
    valid_hull = [r for r in measured if np.isfinite(r['xy_hull_area_m2']) and r['xy_hull_area_m2'] > 0]
    summary = dict(manifest=str(manifest.resolve()), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   clouds=len(measured), total_points=total_points, crs_counts=crs_counts,
                   metrics=metrics,
                   pooled_hull_density_pts_m2=sum(r['points'] for r in valid_hull) / sum(r['xy_hull_area_m2'] for r in valid_hull),
                   pooled_bbox_density_pts_m2=total_points / sum(r['xy_bbox_area_m2'] for r in measured),
                   counts_below={str(k):sum(r['points'] < k for r in measured) for k in (32,64,128,256,512,1024,2048,4096,8192,16384,32768)},
                   district={d:dict(clouds=sum(r['district']==d for r in measured),
                       points=stats([r['points'] for r in measured if r['district']==d]),
                       hull_density=stats([r['hull_density_pts_m2'] for r in measured if r['district']==d]))
                       for d in sorted(set(r['district'] for r in measured))},
                   method='Exact counts and XY convex-hull/bounding-box areas for every training cloud. '
                          'Neighbour statistics use up to 64 randomly sampled points per cloud, seed 42. '
                          'XYZ patch k includes the query point, matching model KNN grouping. '
                          'Convex hull fills concavities/holes; these are footprint density estimates, not true polygon or terrain-surface densities.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    report = ['# Training-cloud statistics', '',
              f'Manifest: `{manifest.resolve()}`', '',
              f'{len(measured):,} training clouds; {total_points:,} total points.', '',
              'All training files were checked for projected horizontal coordinates in metres.', '',
              '| Metric | Mean | Min | P5 | P25 | Median | P75 | P95 | Max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels = {'points': 'Points/cloud', 'xy_hull_area_m2': 'XY convex-hull area (m²)',
              'hull_density_pts_m2': 'Hull density (points/m²)',
              'bbox_density_pts_m2': 'Bounding-box density (points/m²)',
              'height_range_m': 'Z range (m)',
              'normalization_xy_radius_m': 'Normalization XY radius (m)',
              'sampled_median_xy_nn_m': 'Per-cloud median XY nearest-neighbour distance (m)',
              'sampled_median_xyz_patch8_radius_m': 'Per-cloud median 8-point XYZ radius (m)',
              'sampled_median_xyz_patch16_radius_m': 'Per-cloud median 16-point XYZ radius (m)',
              'sampled_median_xyz_patch32_radius_m': 'Per-cloud median 32-point XYZ radius (m)'}
    for key, label in labels.items():
        report.append('| ' + label + ' | ' + ' | '.join(
            f'{metrics[key][stat]:,.3f}' for stat in ('mean','min','p5','p25','median','p75','p95','max')) + ' |')
    report += ['', f'Pooled hull density: {summary["pooled_hull_density_pts_m2"]:.3f} points/m² '
               '(sum of points / sum of valid hull areas; not a geographic union).', '',
               '| District | Clouds | Mean points | Median points | Median hull density |',
               '|---|---:|---:|---:|---:|']
    for district, values in summary['district'].items():
        report.append(f'| {district} | {values["clouds"]:,} | {values["points"]["mean"]:,.1f} | '
                      f'{values["points"]["median"]:,.1f} | {values["hull_density"]["median"]:.3f} |')
    report += ['', '## Interpretation', '', summary['method'], '',
               'These are horizontal footprint densities, not density per sloping terrain surface area. '
               'The mean density weights clouds equally; pooled density weights them by footprint area. '
               'Neighbour radii summarize sampled original points, not the actual FPS-selected centers. '
               'Second- and third-scale model groups operate on centers, not raw points.', '',
               'The notebook divides XYZ by each cloud’s maximum XY radius about its mean. '
               'Therefore `local_radius` is dimensionless; multiply by that cloud’s normalization radius '
               'to obtain the corresponding physical XYZ distance in metres.', '',
               'See `per_cloud.csv`, `summary.json`, and `distributions.png` for full results.', '',
               'Method references: [SciPy ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html), '
               '[SciPy cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html).']
    (output / 'report.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes,
            ['points', 'hull_density_pts_m2', 'sampled_median_xyz_patch8_radius_m'],
            ['Points per cloud', 'Points / XY convex-hull m²', '8-point XYZ patch radius (m)']):
        a = np.array([r[key] for r in measured])
        a = a[np.isfinite(a) & (a > 0)]
        ax.hist(a, bins=np.geomspace(a.min(), a.max(), 50), color='#287a9f')
        ax.set_xscale('log')
        ax.axvline(np.median(a), color='#c14d27', label=f'Median: {np.median(a):,.2f}')
        ax.set(xlabel=label, ylabel='Training clouds')
        ax.legend()
    fig.suptitle(f'Training split only — {len(measured):,} clouds')
    fig.tight_layout()
    fig.savefig(output / 'distributions.png', dpi=180)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metrics','district')}, indent=2))
    print('POINTS', json.dumps(metrics['points']))
    print('DENSITY', json.dumps(metrics['hull_density_pts_m2']))
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
