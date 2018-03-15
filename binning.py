#!/usr/bin/env python3
# coding: utf-8

from __future__ import division, print_function

import numpy


def sse(nums):
    arr = numpy.array(nums) - numpy.mean(nums)
    return numpy.sum(arr ** 2)


def v_opt_dp(x, num_bins):
    if not (0 < num_bins <= len(x)):
        raise ValueError('invalid input')

    shape = num_bins, len(x)
    matrix = numpy.ones(shape) * (-1)
    track_matrix = numpy.zeros(shape, dtype='int')

    # only 1 bin
    for i in range(len(x)):
        # (num_bins - 1) bins, first i numbers
        # bins > numbers, NOT allowed
        if num_bins - 1 > i:
            continue
        # 1 bin, last len(x) - i numbers
        # bins > numbers, NOT allowed
        # if 1 > len(x) - i:
        #     continue
        matrix[0, i] = sse(x[i:])
        track_matrix[0] = len(x)

    # k + 1 bins
    for k in range(1, num_bins):
        for i in range(len(x)):
            # (num_bins - k - 1) bins, first i numbers
            # bins > numbers, NOT allowed
            if num_bins - k - 1 > i:
                continue
            # k bin, last len(x) - i numbers
            # bins > numbers, NOT allowed
            if k > len(x) - i:
                continue

            min_cost = -1
            # TODO: marginal
            for j in range(i + 1, len(x)):
                cost = matrix[k - 1, j] + sse(x[i:j])
                if min_cost < 0 or min_cost > cost:
                    min_cost = cost
                    track_matrix[k, i] = j
            matrix[k, i] = min_cost

    i = 0
    k = num_bins - 1
    intervals = []
    while k >= 0:
        i_next = track_matrix[k, i]
        intervals.append([i, i_next])
        i = i_next
        k -= 1

    bins = []
    for a, b in intervals:
        bins.append(x[a:b])
        # print(a, b)

    # print(matrix)
    # print('track_matrix\n', track_matrix)
    return matrix, bins


def test_v_opt_dp():
    x = [3, 1, 18, 11, 13, 17]
    num_bins = 4
    matrix, bins = v_opt_dp(x, num_bins)
    print(bins)
    for row in matrix:
        print(row)

if __name__ == '__main__':
    test_v_opt_dp()
