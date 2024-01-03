#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:16:27 2024

@author: azelcer
"""
import numpy as np
import numba


@numba.njit
def product_like(*args):
    # chequear que args sean todos arrays del mismo tipo
    l = len(args)
    if l == 0:
        return np.zeros((0,), dtype=np.float64)
    type_ = args[0].dtype
    for arg in args:
        if arg.dtype == type_:
            ...  # != is not implemented
        else:
            raise TypeError("Mixed ndarray types")
    lenghts = np.zeros((l,), dtype=np.int64)
    for i in range(l):
        lenghts[i] = len(args[i])
        if lenghts[i] == 0:
            return np.zeros((0,), dtype=type_)
    # lenghts = [len(args[i]) for i in range(l)]
    # cur_pos = [0 for _ in args]
    cur_pos = np.zeros((l,), dtype=np.int64)

    while True:
        # rv = np.array(cur_pos, dtype=type_)
        rv = np.zeros((l,), dtype=type_)
        for idx in range(l):
            rv[idx] = args[idx][cur_pos[idx]]
        # rv = np.array([args[idx][cur_pos[idx]] for idx in range(l)])
        # rv = (args[idx][cur_pos[idx]] for idx in range(l))
        yield rv  # pasar al loop interno
        idx_to_move = l-1
        while idx_to_move >= 0:
            if cur_pos[idx_to_move] >= lenghts[idx_to_move]-1:
                cur_pos[idx_to_move] = 0
                idx_to_move -= 1
            else:
                cur_pos[idx_to_move] += 1
                break
        if idx_to_move < 0:
            break


@numba.njit
def tontosum(a, b):
    rv = 0
    for x in product_like(a, b):
        rv += x[0] * x[1]
    return rv


if __name__ == '__main__':
    from itertools import product
    a = np.arange(1., 234, 1)
    b = np.arange(6, 80, 1.34)
    rv1 = [np.array(p) for p in product(a, b)]
    rv2 = list(product_like(a, b))
    print(all((np.all(r1 == r2) for r1, r2 in zip(rv1, rv2))))
    ...
