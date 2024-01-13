#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:16:27 2024

@author: azelcer
"""
import numpy as np
import numba


_PARALELLIZE = False


@numba.njit(cache=False)
def inner(idx, a):
    for k in [0]:
        idx = k
    x = a[idx]
    return x


@numba.njit(parallel=_PARALELLIZE, cache=False)
def outer(a):
    rv = 0.
    total = 100
    for i in numba.prange(total):
        x = inner(i, a)
        rv += x
    return rv


@numba.njit(parallel=True, cache=False)
def simple(a):
    rv = 0.
    total = 100
    for i in numba.prange(total):
        for k in [0]:
            idx = k
        x = a[idx]
        rv += x
    return rv


@numba.njit(parallel=True, cache=False)
def paralelo(a):
    rv = 0.
    total = 100
    for i in numba.prange(total):
        rv += a[0]
    return rv


if __name__ == '__main__':
    a = np.ones((100,))
    v0 = simple(a)
    vp = paralelo(a)
    v4 = outer(a)
    print(v0, vp, v4)
