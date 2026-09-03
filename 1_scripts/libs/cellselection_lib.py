# -*- coding: utf-8 -*-
"""
Data-selection constraint for analyses involving unlabeled V1/PM
populations (V1unl / PMunl), used as putative "cells projecting to other
areas" contrasted with the labeled (V1lab / PMlab) projection population.

Two constraints are enforced, and ONLY for cells that are both (a) in V1
or PM, and (b) unlabeled (celldata['redcell'] == 0):
    1. Proximity: must be within `radius` microns (3D) of at least one
       labeled cell in the SAME area, in the SAME session.
    2. Layer: must be superficial / layer 2/3 (depth < depth_thr).

Every other cell -- labeled cells (any area), and unlabeled cells outside
V1/PM -- is left completely untouched and always passes.

Self-contained: computes distances directly from celldata['xloc'],
['yloc'], ['depth'] rather than reusing/relying on ses.distmat_xyz
(avoids relying on a cached distance matrix that may be stale or built
under a different diagonal/NaN-masking convention).

Matthijs Oude Lohuis, Champalimaud Research
"""

import numpy as np

CONSTRAINED_AREAS = ('V1', 'PM')


def get_constrained_mask(celldata):
    """
    Boolean mask of cells the constraint actually applies to: unlabeled
    cells in V1 or PM. Everything else is untouched by design.
    """
    roi_name = celldata['roi_name'].to_numpy()
    is_labeled = celldata['redcell'].to_numpy().astype(bool)
    is_target_area = np.isin(roi_name, CONSTRAINED_AREAS)
    return is_target_area & ~is_labeled


def filter_nearlabeled_layer23(ses, radius=50, depth_thr=300, lateral_only=True):
    """
    Boolean selection mask over ses.celldata (same length/order).

    Parameters
    ----------
    ses : Session
        Must have ses.celldata with 'roi_name', 'redcell', 'depth',
        'xloc', 'yloc' columns.
    radius : float
        Proximity threshold in microns. Default 50.
    depth_thr : float
        Layer 2/3 depth cutoff in microns. Default 300.
    lateral_only : bool
        If True, proximity is computed in 2D (xloc/yloc only), ignoring
        depth differences (e.g. if all cells of interest are already
        known to be in the same imaging plane). Default False: full 3D
        distance (xloc, yloc, depth).

    Returns
    -------
    idx_pass : boolean numpy array, len(ses.celldata)
        True for every cell that's allowed into the analysis: labeled
        cells and non-V1/PM cells are always True; unlabeled V1/PM cells
        are True only if they satisfy BOTH the proximity and layer
        constraints.
    """
    celldata = ses.celldata
    N = len(celldata)

    roi_name   = celldata['roi_name'].to_numpy()
    is_labeled = celldata['redcell'].to_numpy().astype(bool)
    depth      = celldata['depth'].to_numpy().astype(float)
    x          = celldata['xloc'].to_numpy().astype(float)
    y          = celldata['yloc'].to_numpy().astype(float)

    idx_constrained = get_constrained_mask(celldata)

    # start everyone as passing; we will only ever turn OFF cells inside
    # idx_constrained below, so labeled cells / non-V1/PM cells (for which
    # idx_constrained is False) are guaranteed to stay True.
    idx_pass = np.ones(N, dtype=bool)

    if not np.any(idx_constrained):
        return idx_pass

    for area in CONSTRAINED_AREAS:
        idx_area_constrained = idx_constrained & (roi_name == area)
        if not np.any(idx_area_constrained):
            continue

        idx_area_labeled = is_labeled & (roi_name == area)
        if not np.any(idx_area_labeled):
            # no labeled cells in this area this session -> no unlabeled
            # cell here can satisfy the proximity constraint
            idx_pass[idx_area_constrained] = False
            continue

        xu, yu, zu = x[idx_area_constrained], y[idx_area_constrained], depth[idx_area_constrained]
        xl, yl, zl = x[idx_area_labeled], y[idx_area_labeled], depth[idx_area_labeled]

        if lateral_only:
            d = np.sqrt((xu[:, None] - xl[None, :]) ** 2
                        + (yu[:, None] - yl[None, :]) ** 2)
        else:
            d = np.sqrt((xu[:, None] - xl[None, :]) ** 2
                        + (yu[:, None] - yl[None, :]) ** 2
                        + (zu[:, None] - zl[None, :]) ** 2)

        near_ok = np.any(d <= radius, axis=1)
        idx_pass[idx_area_constrained] = near_ok

    # layer 2/3 constraint -- ONLY for the constrained population; combine
    # with (not overwrite) the proximity result via logical AND
    idx_pass[idx_constrained] &= (depth[idx_constrained] < depth_thr)

    return idx_pass


def filter_target_groups(area, label, target_areas=None, target_labels=None):
    """
    Boolean mask selecting only neurons belonging to the requested
    (area, label) combinations -- the "which groups do I want" argument
    shared across every downstream analysis script (information-theoretic,
    dimensionality reduction, etc.) so the same selector works everywhere.

    Parameters
    ----------
    area, label : array-like, same length
        Typically the output of infotheory.celldata_utils.get_area_label.
    target_areas : list of str or None
        e.g. ['V1', 'PM'] to restrict to those areas, or None for "all
        areas, no restriction".
    target_labels : list of str or None
        e.g. ['lab'] to restrict to labeled cells only, or None for "all
        labels, no restriction".

    Returns
    -------
    idx_target : boolean numpy array, len(area)
        True for neurons matching the requested area(s) AND label(s).
        Combine with the anatomical/QC filters via logical AND, e.g.:
            idx_valid = idx_anat & idx_qc & filter_target_groups(area, label, target_areas, target_labels)

    Examples
    --------
    filter_target_groups(area, label)                                   # everything
    filter_target_groups(area, label, target_labels=['lab'])            # labeled cells only, any area
    filter_target_groups(area, label, target_areas=['PM'])              # PM only, any label
    filter_target_groups(area, label, target_areas=['PM'], target_labels=['lab'])  # PM labeled only
    """
    area = np.asarray(area)
    label = np.asarray(label)
    idx = np.ones(len(area), dtype=bool)
    if target_areas is not None:
        idx &= np.isin(area, target_areas)
    if target_labels is not None:
        idx &= np.isin(label, target_labels)
    return idx
