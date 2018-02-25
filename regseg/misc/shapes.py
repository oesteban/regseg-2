# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generating phantom shapes
~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import sys
import numpy as np
from scipy import ndimage


def ball(volsize, radius, dims=3):
    """
    Generate a ball of radius
    """

    if not isinstance(volsize, (list, tuple)):
        volsize = [volsize] * dims

    assert np.all(np.array(volsize) > 0)
    assert radius < min(volsize) * 0.5

    result = np.zeros(shape=tuple(volsize), dtype=int)
    center = (np.array(volsize) - 1) * 0.5

    for x in range(0, int(volsize[0])):
        for y in range(0, int(volsize[1])):
            for z in range(0, int(volsize[2])):
                if np.linalg.norm(center - [x, y, z]) < radius:
                    result[x, y, z] = 1
    return result


def gen_ball(datashape=(51, 51, 51), radius=20, cortex=True):
    wm = ball(datashape, radius)

    if cortex:
        ball2 = ball(14, 4.4)
        gm = ndimage.binary_dilation(wm, structure=ball2).astype(np.uint8) - wm
        bg = np.ones_like(wm) - (gm + wm)
        return [bg, wm, gm]

    bg = np.ones_like(wm) - wm
    return [bg, wm]


def gen_gyrus(datashape=(61, 61, 61), radius=20, cortex=True):
    modelbase = ball(datashape, radius)
    center_pix = ((np.array(datashape) - 1) * 0.5).astype(np.uint8)
    modelbase[center_pix[0], center_pix[1]:, :] = 0
    # displ_pix = ((np.array(datashape) - 1) * 0.20).astype(np.uint8)
    # modelbase[:displ_pix[0], center_pix[1], center_pix[2]] = 0
    # modelbase[displ_pix[0]:, center_pix[1], :] = 0
    ball1 = ball(9, 4.5)
    wm = ndimage.binary_opening(ndimage.binary_erosion(
        modelbase, structure=ball1).astype(np.uint8),
        structure=ball1).astype(np.uint8)

    if cortex:
        ball2 = ball(9, 3.5)
        gm = ndimage.binary_dilation(wm, structure=ball2).astype(np.uint8) - wm
        bg = np.ones_like(modelbase) - (gm + wm)
        return [bg, wm, gm]
    bg = np.ones_like(modelbase) - wm
    return [bg, wm]


def gen_box(datashape=(51, 51, 51), coverage=0.4, cortex=True):
    modelbase = np.zeros(shape=datashape)
    extent = np.around(coverage * np.array(datashape))
    padding = np.around(0.5 * (np.array(datashape) - extent))
    end = np.array(datashape) - padding
    modelbase[padding[0]:end[0], padding[1]:end[1], padding[2]:end[2]] = 1

    ball1 = ball(11, 4.5)
    wm = ndimage.binary_opening(ndimage.binary_erosion(
        modelbase, structure=ball1).astype(np.uint8),
        structure=ball1).astype(np.uint8)

    if cortex:
        ball2 = ball(11, 4.4)
        gm = ndimage.binary_dilation(wm, structure=ball2).astype(np.uint8) - wm
        bg = np.ones_like(modelbase) - (gm + wm)
        return [bg, wm, gm]
    bg = np.ones_like(modelbase) - wm
    return [bg, wm]


def gen_l(datashape=(51, 51, 51), cortex=True):
    modelbase = np.zeros(shape=datashape)
    center = np.around(0.5 * np.array(datashape))
    extent = np.around(0.4 * np.array(datashape))
    padding = np.around(0.5 * (np.array(datashape) - extent))
    end = np.array(datashape) - padding
    modelbase[padding[0]:end[0], padding[1]:end[1], padding[2]:end[2]] = 1
    modelbase[center[0]:end[0], center[1]:end[1], center[2]:end[2]] = 0

    ball1 = ball(11, 4.5)
    wm = ndimage.binary_opening(ndimage.binary_erosion(
        modelbase, structure=ball1).astype(np.uint8),
        structure=ball1).astype(np.uint8)

    if cortex:
        ball2 = ball(11, 4.4)
        gm = ndimage.binary_dilation(wm, structure=ball2).astype(np.uint8) - wm
        bg = np.ones_like(modelbase) - (gm + wm)
        return [bg, wm, gm]
    bg = np.ones_like(modelbase) - wm
    return [bg, wm]


def generate_shape(name, datashape=(51, 51, 51), cortex=True):
    if name not in ['box', 'L', 'ball', 'gyrus']:
        raise RuntimeError('Model "%s" does not exist' % name)

    return getattr(sys.modules[__name__], 'gen_%s' % name.lower())(
        datashape=datashape, cortex=cortex)
