#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

def bh(pv, weight=None,size=None):
	"""Convert p-values to q-values using Benjamini–Hochberg procedure
	pv:		numpy.array of p-values
	weight:	numpy.array of optional weights. Same shape with pv. Uniform weights if not specified.
	size:	Total number of tests. If specified, pv indicates the smallest p-values. If unspecified, size is the length of pv.
	Return:	numpy.array of q-values
	Ref:	Controlling the false discovery rate: a practical and powerful approach to multiple testing, 1995
	"""
	import numpy as np
	import logging
	assert len(pv.shape) == 1 and pv.size > 0
	assert np.isfinite(pv).all() and pv.min() >= 0 and pv.max() <= 1
	dtype = 'f8'
	n0 = pv.size
	if weight is None:
		weight = np.ones(n0)
	else:
		assert weight.shape == pv.shape
		assert np.isfinite(weight).all() and weight.min() >= 0 and weight.max() > 0

	# Shrink data
	pv2, ids = np.unique(pv, return_inverse=True)
	n = pv2.size
	if n == 1:
		logging.warning('Identical p-value in all entries.')
	w = np.zeros(n, dtype=dtype)
	for xi in range(n0):
		w[ids[xi]] += weight[xi]

	# BH method
	w = np.cumsum(w)
	if size is None:
		size = w[-1]
	else:
		assert size>=w[-1]
	w /= size
	pv2 /= w
	pv2[~np.isfinite(pv2)] = 1
	pv2 = np.clip(pv2,0,1)
	for xi in range(n - 2, -1, -1):
		pv2[xi] = min(pv2[xi], pv2[xi + 1])

	# Recover index
	ans = pv2[ids].astype(dtype, copy=False)
	assert ans.shape == pv.shape
	return ans
