"""Point-count-based grouping for the three-scale Point-M2AE notebook.

Counts use successive ceiling divisions, with small-cloud decoder minima.
The policy needs only the standard library. Configuring native dividers imports
knn_cuda lazily when a neighbourhood size changes. Input points and learned
parameters are never modified by this module.
"""
import math


def _validate_triple(values, name, minimum):
    if (not isinstance(values, (list, tuple)) or len(values) != 3
            or any(type(value) is not int or value < minimum for value in values)):
        raise ValueError(f'{name} must contain three integers, each at least {minimum}.')


def hierarchy_for_cloud(num_points, group_divisors, group_sizes, mask_ratio):
    """Return ``(group_counts, neighbour_counts)`` for one complete cloud.

    With divisors [8, 2, 4], a 2048-point cloud has [256, 128, 32]
    centres. Each count is ceil(previous_count / divisor), raised to the
    decoder's minimum [8, 4, 3] and bounded by the available input count.
    There is no fixed upper cap. Group sizes specify KNN neighbours, not
    divisors; neighbourhoods may overlap and do not partition all input points.

    Final-scale KNN size is reduced if necessary to reserve a masked
    middle-scale reconstruction target under upstream mask back-propagation.
    Impossible configurations raise ValueError before running CUDA kernels.
    """
    if type(num_points) is not int or num_points < 8:
        raise ValueError('At least 8 original points are required (integer point count).')
    _validate_triple(group_divisors, 'group_divisors', 1)
    _validate_triple(group_sizes, 'group_sizes', 2)
    if (type(mask_ratio) not in (float, int) or not math.isfinite(mask_ratio)
            or not 0 < mask_ratio < 1):
        raise ValueError('mask_ratio must be a finite number strictly between 0 and 1.')

    groups = []
    available = num_points
    for divisor, minimum in zip(group_divisors, (8, 4, 3)):
        count = min(available, max(minimum, (available + divisor - 1) // divisor))
        groups.append(count)
        available = count

    masked = int(mask_ratio * groups[-1])
    if not 1 <= masked < groups[-1]:
        raise ValueError('Mask ratio must leave masked and visible top-scale patches '
                         'for masked reconstruction.')
    visible = groups[-1] - masked
    # A visible top patch reveals <= K middle centres. The supplied encoder
    # additionally scatters masked entries to index zero; reserve that slot too.
    top_neighbours = min(group_sizes[-1], (groups[-2] - 2) // visible)
    if top_neighbours < 2:
        raise ValueError('Cannot retain masked reconstruction targets with two neighbours; '
                         'increase mask_ratio or the final group divisor.')
    sizes = [min(group_sizes[0], num_points), min(group_sizes[1], groups[0]), top_neighbours]
    return groups, sizes


def configure_group_dividers(dividers, num_points, group_divisors, group_sizes, mask_ratio):
    """Apply the policy to an upstream model's three FPS/KNN Group modules.

    Pass the original configured group_sizes every time, so a small cloud's
    adapted KNN sizes are restored when the next cloud is larger. No model
    weights or reconstruction-head dimensions change. Returns the active counts.
    """
    if len(dividers) != 3:
        raise ValueError('Point-M2AE requires exactly three group dividers.')
    groups, sizes = hierarchy_for_cloud(num_points, group_divisors, group_sizes, mask_ratio)
    replacements = {}
    for index, (divider, size) in enumerate(zip(dividers, sizes)):
        if divider.group_size != size:
            from knn_cuda import KNN
            replacements[index] = KNN(k=size, transpose_mode=True)
    for index, (divider, count, size) in enumerate(zip(dividers, groups, sizes)):
        divider.num_group = count
        divider.group_size = size
        if index in replacements:
            divider.knn = replacements[index]
    return groups, sizes
