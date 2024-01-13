#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:16:27 2024

@author: azelcer
"""
import numpy as np
import numba
import time
import itertools

# import tracemalloc
# spec = [
#     ('_lenghts', numba.int64[:]),
#     ('_pos', numba.int64[:]),
#     ('_finished', numba.bool_)
# ]

# @numba.experimental.jitclass(spec)
# class Bag(object):
#     def __init__(self, *args):
#         self._lenghts = np.empty(len(args), dtype=np.int64)
#         self._pos = np.zeros(len(args), dtype=np.int64)
#         self._finished = False
#         for i in range(len(args)):
#             self._lenghts[i] = len(args[i])

#     def prepare_output(self, *args) -> list:
#         return [x[0] for x in args]

#     def next(self, out: list, *args) -> bool:
#         if self._finished:
#             return False
#         l = len(self._lenghts)
#         for idx in range(l):
#             out[idx] = self._data[idx][self._pos[idx]]
#         # rv = [args[idx][self._pos[idx]] for idx in range(l)]
#         idx_to_move = l-1
#         while idx_to_move >= 0:
#             if self._pos[idx_to_move] >= self._lenghts[idx_to_move] - 1:
#                 self._pos[idx_to_move] = 0
#                 idx_to_move -= 1
#             else:
#                 self._pos[idx_to_move] += 1
#                 break
#         if idx_to_move < 0:
#             self._finished = True
#         return True


@numba.extending.overload(itertools.product)
def p_like(*args):
    l = len(args)
    if l == 0:
        raise numba.core.errors.TypingError("At last one argument needed")
    dtype = None
    for i, arg in enumerate(args):
        if not isinstance(arg, numba.types.Array):
            raise numba.core.errors.TypingError(f"Parameter #{i} is not an array")
        if dtype is None:
            dtype = arg.dtype
        if dtype != arg.dtype:
            raise numba.core.errors.TypingError("Mixed types in parameters")

    def _prod(*args):
        lenghts = np.empty((l,), dtype=np.int64)
        for i in range(l):
            lenghts[i] = len(args[i])
            if lenghts[i] == 0:
                return np.empty((0,), dtype=dtype)

        cur_pos = [0 for _ in args]
        # cur_pos = np.zeros((l,), dtype=np.int64)

        while True:
            rv = np.empty((l,), dtype=dtype)
            for idx in range(l):
                rv[idx] = args[idx][cur_pos[idx]]
            # rv = [args[idx][cur_pos[idx]] for idx in range(l)]
            yield rv

            idx_to_move = l-1
            while idx_to_move >= 0:
                if cur_pos[idx_to_move] >= lenghts[idx_to_move] - 1:
                    cur_pos[idx_to_move] = 0
                    idx_to_move -= 1
                else:
                    cur_pos[idx_to_move] += 1
                    break
            if idx_to_move < 0:
                break
    return _prod


@numba.njit
def product_like(*args):
    # itertools.products behaves strangely with an empty argument list
    l = len(args)
    if l == 0:
        return np.empty((0,), dtype=np.float64)
    # chequear que args sean todos arrays del mismo tipo
    type_ = args[0].dtype
    for arg in args:
        if arg.dtype == type_:
            ...  # != for types is not implemented
        else:
            raise TypeError("Mixed ndarray types")
    lenghts = np.empty((l,), dtype=np.int64)
    for i in range(l):
        lenghts[i] = len(args[i])
        if lenghts[i] == 0:
            return np.empty((0,), dtype=type_)

    # cur_pos = [0 for _ in args]
    cur_pos = np.zeros((l,), dtype=np.int64)

    while True:
        rv = np.empty((l,), dtype=type_)
        for idx in range(l):
            rv[idx] = args[idx][cur_pos[idx]]
        # rv = [args[idx][cur_pos[idx]] for idx in range(l)]
        # rv = [arg[pos] for arg, pos in zip(args, cur_pos)]
        yield rv

        # loop interno, no sirve de mucho
        # for i in range(lenghts[l-1]):
        #     cur_pos[-1] = i
        #     rv = [args[idx][cur_pos[idx]] for idx in range(l)]
        #     yield rv

        idx_to_move = l-1  # Era -2 para loop interno
        while idx_to_move >= 0:
            if cur_pos[idx_to_move] >= lenghts[idx_to_move] - 1:
                cur_pos[idx_to_move] = 0
                idx_to_move -= 1
            else:
                cur_pos[idx_to_move] += 1
                break
        if idx_to_move < 0:
            break


@numba.njit
def plike2(*args):
    # itertools.products behaves strangely with an empty argument list
    l = len(args)
    if l == 0:
        return np.empty((0,), dtype=np.float64)
    # chequear que args sean todos arrays del mismo tipo
    type_ = args[0].dtype
    for arg in args:
        if arg.dtype == type_:
            ...  # != for types is not implemented
        else:
            raise TypeError("Mixed ndarray types")
    lenghts = [len(arg) for arg in args]
    lv = 1
    lenghts.reverse()
    for i in range(l):
        lv *= lenghts[i]
        lenghts[i] = lv
    lenghts.reverse()

    for idx in range(lenghts[0]):
        rv = np.empty((l,), dtype=type_)
        for i in range(l-1):
            div = idx // lenghts[i+1]
            rem = idx % lenghts[i+1]
            rv[i] = args[i][div]
            idx = rem
        rv[i+1] = args[i+1][idx]
        yield rv


@numba.njit
def prepplike3(*args):
    l = len(args)
    lenghts = [len(arg) for arg in args]
    lv = 1
    lenghts.reverse()
    for i in range(l):
        lv *= lenghts[i]
        lenghts[i] = lv
    lenghts.reverse()
    return np.array(lenghts, dtype=np.int64)


@numba.njit
def plike3(idx, lenghts, *args):
    l = len(lenghts)
    rv = np.empty((l,), dtype=args[0].dtype)
    print(idx)
    idx = int(idx)
    print(idx)
    for i in range(l-1):
        div = int(idx // lenghts[i+1])
        rem = int(idx % lenghts[i+1])
        # rv[i] = args[i][div]
        tmp = args[i]
        rv[i] = tmp[div]
        idx = int(rem)
    rv[int(i+1)] = args[int(i+1)][int(idx)]
    return rv


@numba.njit
def tontotal(*args):
    # itertools.products behaves strangely with an empty argument list
    l = len(args)
    type_ = args[0].dtype
    lenghts = np.empty((l,), dtype=np.int64)
    for i in range(l):
        lenghts[i] = len(args[i])
        if lenghts[i] == 0:
            return 0.

    # cur_pos = [0 for _ in args]
    cur_pos = np.zeros((l,), dtype=np.int64)
    valor = 0.0
    rv = np.empty((l,), dtype=type_)
    while True:
        xtra = 1.
        for idx in range(l):
            rv[idx] = args[idx][cur_pos[idx]]
            xtra *= rv[idx]
        valor += xtra

        idx_to_move = l-1  # Era -2 para loop interno
        while idx_to_move >= 0:
            if cur_pos[idx_to_move] >= lenghts[idx_to_move] - 1:
                cur_pos[idx_to_move] = 0
                idx_to_move -= 1
            else:
                cur_pos[idx_to_move] += 1
                break
        if idx_to_move < 0:
            break
    return valor


@numba.njit
def tontosum0(*args):
    rv = 0.
    for x in product_like(*args):
        rv += x[0] * x[1] * x[2]
    return rv


@numba.njit
def tontosum1(*args):
    rv = 0.
    for x in itertools.product(*args):
        rv += x[0] * x[1] * x[2]
    return rv


def tontosum2(*args):
    rv = 0.
    for x in itertools.product(*args):
        rv += x[0] * x[1] * x[2]
    return rv


@numba.njit
def tontosum3(*args):
    rv = 0.
    for x in plike2(*args):
        rv += x[0] * x[1] * x[2]
    return rv


@numba.njit#(parallel=True)
def tontosum4(*args):
    prep = prepplike3(*args)
    rv = 0.
    total = 1
    for a in args:
        total *= len(a)
    for i in numba.prange(total):
        x = plike3(int(i), prep, *args)
        rv += x[0] * x[1] * x[2]
    return rv


if __name__ == '__main__':
    a = np.arange(1., 200, 1)
    b = np.arange(6., 180, 1)
    c = np.arange(6., 79, .75)
    # rv1 = [list(p) for p in itertools.product(a, b, c)]
    # rv2 = [p for p in product_like(a, b, c)]
    # print(all((np.all(r1 == r2) for r1, r2 in zip(rv1, rv2))))

    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()
    v0 = tontosum0(a, b, c)
    t0 = time.time()
    v0 = tontosum0(a, b, c)
    tf = time.time()
    print(tf-t0)
    v1 = tontosum1(a, b, c)
    t0 = time.time()
    v1 = tontosum1(a, b, c)
    tf = time.time()
    print(tf-t0)
    t0 = time.time()
    v2 = tontosum2(a, b, c)
    tf = time.time()
    print(tf-t0)
    v3 = tontosum3(a, b, c)
    t0 = time.time()
    v3 = tontosum3(a, b, c)
    tf = time.time()
    print(tf-t0)
    v4 = tontosum4(a, b, c)
    t0 = time.time()
    v4 = tontosum4(a, b, c)
    tf = time.time()
    print(tf-t0)
    v9 = tontotal(a, b, c)
    t0 = time.time()
    v9 = tontotal(a, b, c)
    tf = time.time()
    print(tf-t0)
    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # print("[ Top 10 differences ]")
    # for stat in top_stats[:10]:
    #     print(stat)


    # a = np.arange(3, 50)
    # b = np.arange(6, 300)
    # c = np.arange(0, 2)

    # print(len(a), len(b), len(c))
    # d1 = list(plike2(a, b, c))
    # d2 = [np.array(x) for x in itertools.product(a, b, c)]
    # d3 = list(product_like(a, b, c))
    # print(all((np.all(r1 == r2) for r1, r2 in zip(d1, d2))))
    # print(all((np.all(r1 == r2) for r1, r2 in zip(d1, d3))))
