#!/usr/bin/python3
# Copyright 2020-2022, 2025, Lingfei Wang
#
# This file is part of airqtl.

def groupby(array):
	"""Groups 1-D array to dict(value=np.array(index))
	"""
	from collections import defaultdict
	import numpy as np
	d=defaultdict(list)
	for xi in range(len(array)):
		d[array[xi]].append(xi)
	d=dict(d)
	d={x:np.array(y) for x,y in d.items()}
	return d

def smallest_dtype_int(arr):
	"""Convert array to smallest dtype that can hold all values.
	arr:	Array to convert
	Return:	Converted array"""
	import numpy as np
	if not np.issubdtype(arr.dtype,np.integer):
		raise ValueError('Input array must be integer type.')
	if np.issubdtype(arr.dtype,np.signedinteger):
		#Test the extreme value
		m1=arr.min()
		m2=arr.max()
		if m1>=np.iinfo(np.int8).min and m2<=np.iinfo(np.int8).max:
			arr=arr.astype(np.int8)
		elif m1>=np.iinfo(np.int16).min and m2<=np.iinfo(np.int16).max:
			arr=arr.astype(np.int16)
		elif m1>=np.iinfo(np.int32).min and m2<=np.iinfo(np.int32).max:
			arr=arr.astype(np.int32)
		elif m1>=np.iinfo(np.int64).min and m2<=np.iinfo(np.int64).max:
			arr=arr.astype(np.int64)
		else:
			raise ValueError('Input array is too large.')
	elif np.issubdtype(arr.dtype,np.unsignedinteger):
		#Test the largest value
		m=arr.max()
		if m<=np.iinfo(np.uint8).max:
			arr=arr.astype(np.uint8)
		elif m<=np.iinfo(np.uint16).max:
			arr=arr.astype(np.uint16)
		elif m<=np.iinfo(np.uint32).max:
			arr=arr.astype(np.uint32)
		elif m<=np.iinfo(np.uint64).max:
			arr=arr.astype(np.uint64)
		else:
			raise ValueError('Input array is too large.')
	else:
		raise ValueError('Input array must be integer type.')
	return arr
