#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

"""
Pretending to import unimportable modules when not needed
"""

class dummy_class:
	"""
	Dummy class for unimportable modules
	"""
	def __getattr__(self,key):
		if key=='__mro_entries__':
			return lambda *a,**ka:tuple()
		return self
	def __call__(self,*a,**ka):
		return self


try:
	import torch
except ModuleNotFoundError:
	torch=dummy_class()

assert __name__ != "__main__"
