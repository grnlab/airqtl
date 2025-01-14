#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

"""
IO
"""

import abc
import logging
from collections import OrderedDict
from functools import partial
from os import linesep
from typing import Callable, Optional, Tuple, Union


class base_writer(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def __setitem__(self,k,v):
		pass

class null_writer(base_writer):
	"""
	Do nothing. Useful for benchmarking time complexity.
	"""
	def __setitem__(self,k,v):
		pass

class filtered_sparse_writer(base_writer):
	"""
	Dummy writer that simply reports to master. See filtered_sparse_master_writer.
	"""
	def __init__(self,id,p):
		"""
		id:	Variable index of writer
		p:	Parent/master
		"""
		self.id=id
		self.p=p
	def __setitem__(self,k,v):
		self.p.setitem(self.id,k,v)

class filtered_sparse_master_writer(metaclass=abc.ABCMeta):
	open=partial(open,mode='w+')
	def __init__(self,paths:list[Union[str,list[list[str],str]]],shapes:list[Tuple[int,int,...]],filt:Callable,check_update:bool=False,sep='_'):
		"""
		paths:			List of paths to output files for each variable. For variables higher than two dimensions, the corresponding path should be a list of list of str, where each list of str corresponds to names of a dimension, plus two strs as universal file name prefix and suffix.
		shapes:			List of shapes of variables
		filt:			Filtering function to determine which entries to output. 
						filt(indices:list[NDArray],value1:NDArray,value2:NDArray,...)->NDArray[bool]
						Accepts variables as indices and numpy.ndarray in the same order as in paths and returns a boolean numpy.ndarray of the same size.
		check_update:	Whether to check for value updates. Raises NotImplementedError if True and updates are detected. Saves time and memory if False.
		sep:			Separator for file names. Only used if paths contains list.
		"""
		import itertools
		self.n=len(paths)
		if len(shapes)!=self.n:
			raise TypeError('Number of paths and shapes must match.')
		self.filt=filt
		assert all(len(x)>=2 for x in shapes)
		assert all(all(y>0 for y in x) for x in shapes)
		assert all(x[-2]==shapes[0][-2] and x[-1]==shapes[0][-1] for x in shapes[1:])
		self.shapes=shapes
		self.check_update=check_update
		self.writers=[]
		self.cache={}
		self.f={}
		for xi in range(self.n):
			assert isinstance(paths[xi],str) ^ len(shapes[xi])>2
			self.writers.append(filtered_sparse_writer(xi,self))
			if isinstance(paths[xi],str):
				self.cache[xi]={}
				self.f[xi]=None
			elif isinstance(paths[xi],list):
				assert len(paths[xi])==len(shapes[xi])
				assert all(len(x)==shapes[xi][i] for i,x in enumerate(paths[xi][:-2]))
				for xj in itertools.product(*[range(x) for x in shapes[xi][:-2]]):
					pid=tuple([xi]+list(xj))
					self.cache[pid]={}
					self.f[pid]=None
			else:
				raise TypeError('Incorrect type for paths.')
		self.past=[]
		for xi in self.f:
			if isinstance(xi,int):
				p=paths[xi]
			else:
				p=paths[xi[0]][-2]+sep.join([paths[xi[0]][i][x] for i,x in enumerate(xi[1:])])+paths[xi[0]][-1]
			logging.info(f'Writing file {p}')
			self.f[xi]=self.open(p)
	def __del__(self):
		self.close()
	def close_file(self,x):
		self.f[x].close()
		self.f[x]=None
	def close(self):
		if not hasattr(self,'f'):
			return
		if any(len(x)>0 for x in self.cache.values()):
			logging.warning('Cache not empty. Results could be lost.')
		#Finalize headers
		for xi in self.f:
			if self.f[xi] is None:
				continue
			self.close_file(xi)
	def setitem(self,id,k,v):
		import itertools

		import numpy as np
		assert id>=0 and id<self.n
		#Process keys
		if not isinstance(k,tuple):
			k=(k,)
		if any(not isinstance(x,int) for x in k[:-2]):
			raise NotImplementedError('Only slice is supported for dimensions other than the last two.')
		if any(not isinstance(x,slice) for x in k[-2:]):
			raise NotImplementedError('Only slice is supported for the last two dimensions.')
		if len(k)<len(self.shapes[id]):
			k=k+(slice(None,None,None),)*(len(self.shapes[id])-len(k))
		if len(k)!=len(self.shapes[id]):
			raise TypeError('Incorrect number of dimensions for key.')
		if not isinstance(v,np.ndarray):
			raise TypeError('Only numpy.ndarray is supported.')

		k1=k[:-2]
		k=tuple([x.indices(self.shapes[id][-2:][i]) for i,x in enumerate(k[-2:])])	
		id=tuple([id]+list(k1))
		if len(id)==1:
			id=id[0]
		ks=[np.arange(*x) for x in k]
		t1=tuple(len(x) for x in ks)
		if v.shape!=t1:
			raise TypeError('Incorrect shape for value. Expected: {}. Actual: {}.'.format(t1,v.shape))
			
		self.cache[id][k]=v
		if not all(k in x for x in self.cache.values()):
			# Wait for all variable caches to be filled
			return

		# Check overlap with past entries
		if self.check_update:
			ksset=[set(x) for x in ks]
			if any(all(len(ksset[y]&x[y])>0 for y in range(len(ksset))) for x in self.past):
				raise NotImplementedError('Rewriting past entries is not supported.')
		# Reconstruct original matrix
		cache=[]
		for xi in range(self.n):
			if len(self.shapes[xi])==2:
				cache.append(self.cache[xi][k])
				continue
			shape=self.shapes[xi][:-2]+tuple([len(x) for x in ks])
			t1=np.zeros(shape,dtype=self.cache[list(itertools.islice(filter(lambda x:x[0]==xi,self.cache),1))[0]][k].dtype)
			mask=np.zeros(shape,dtype=bool)
			for xj in filter(lambda x:isinstance(x,tuple) and x[0]==xi,self.cache):
				t2=tuple(xj[1:])
				assert self.cache[xj][k].shape==shape[len(t2):]
				t1[t2]=self.cache[xj][k]
				mask[t2]=True
			assert mask.all()
			cache.append(t1)
		# Perform filtering
		idx0=self.filt(ks,*cache)
		if idx0.shape!=list(itertools.islice(self.cache.values(),1))[0][k].shape:
			raise TypeError('Incorrect shape for filtered output.')
		idx0=np.nonzero(idx0)
		# Write to file
		self.write(k,ks,idx0)
		# Cleanup
		if self.check_update:
			self.past.append(ksset)
		for xi in self.cache.values():
			del xi[k]
	@abc.abstractmethod
	def write(self,k,ks,idx0):
		pass

class filtered_sparse_tsv_master_writer(filtered_sparse_master_writer):
	def __init__(self,paths:list[str],shapes:list[Tuple[int,int,...]],names:list[list[str]],columns:list[str],filt:Callable,compression=None,nth=16,**ka):
		"""
		names:			Row and column names of the output matrix
		compression:	Compression method. Accepts: 'gzip', 'lz4', or None (for no compression).
		nth:			Number of threads for compression. Only used for compression='zstd'.
		"""
		if compression=='gzip':
			import gzip
			self.open=partial(gzip.open,mode='wt')
		elif compression=='zstd':
			raise NotImplementedError('zstd compression lib is bugged.')
			import zstandard as zstd
			self.open=partial(zstd.open,mode='w',cctx=zstd.ZstdCompressor(threads=nth))
		elif compression=='lz4':
			import lz4
			self.open=partial(lz4.frame.open,mode='wt')
		elif compression is None:
			self.open=partial(open,mode='w')
		else:
			raise NotImplementedError('Compression method not supported.')
		self.names=names
		assert len(names[0])==shapes[0][-2] and len(names[1])==shapes[0][-1]
		super().__init__(paths,shapes,filt,**ka)
		for xi in self.f.values():
			xi.write('\t'.join(columns)+'\tvalue'+linesep)
		self.columns=columns
	def write(self,k,ks,idx0):
		idx1=[self.names[x][ks[x][idx0[x]]] for x in range(len(idx0))]
		idx1=list(zip(*idx1))
		assert all(len(x)==len(self.columns) for x in idx1)
		idx1=['\t'.join(x)+'\t' for x in idx1]
		for xi in self.cache:
			s=self.cache[xi][k][idx0]
			s=['{:.10G}'.format(x) for x in s]
			s=''.join([x+y+linesep for x,y in zip(idx1,s)])
			self.f[xi].write(s)

class filtered_sparse_master_writer2(metaclass=abc.ABCMeta):
	open=partial(open,mode='w+')
	def __init__(self,paths:list[str],shapes:list[list[int]],filt:Callable,check_update:bool=False):
		"""
		paths:			List of paths to output files for each variable.
		shapes:			List of shapes of variables
		filt:			Filtering function to determine which entries to output. 
						filt(indices:list[NDArray],value1:NDArray,value2:NDArray,...)->list[NDArray[bool]]
						Accepts variables as indices and numpy.ndarray in the same order as in paths and returns a boolean numpy.ndarray of the same size.
		check_update:	Whether to check for value updates. Raises NotImplementedError if True and updates are detected. Saves time and memory if False.
		sep:			Separator for file names. Only used if paths contains list.
		"""
		import itertools
		self.n=len(paths)
		if len(shapes)!=self.n:
			raise TypeError('Number of paths and shapes must match.')
		self.filt=filt
		assert all(len(x)>=1 for x in shapes)
		assert all(all(y>0 for y in x) for x in shapes)
		self.shapes=shapes
		self.check_update=check_update
		self.writers=[]
		self.cache={}
		self.f={}
		for xi in range(self.n):
			self.writers.append(filtered_sparse_writer(xi,self))
			self.cache[xi]={}
			p=paths[xi]
			logging.info(f'Writing file {p}')
			self.f[xi]=self.open(p)
		self.past=[]
	def __del__(self):
		self.close()
	def close_file(self,x):
		self.f[x].close()
		self.f[x]=None
	def close(self):
		if not hasattr(self,'f'):
			return
		if any(len(x)>0 for x in self.cache.values()):
			logging.warning('Cache not empty. Results could be lost.')
		#Finalize headers
		for xi in self.f:
			if self.f[xi] is None:
				continue
			self.close_file(xi)
	def setitem(self,id,k,v):
		"""
		Writer `id` calls this this function to set its keys `k` with values `v`.
		"""
		import itertools

		import numpy as np
		assert id>=0 and id<self.n
		#Process keys
		if not isinstance(k,tuple):
			raise NotImplementedError
			k=(k,)
		# if any(not isinstance(x,int) for x in k[:-2]):
		# 	raise NotImplementedError('Only slice is supported for dimensions other than the last two.')
		# if any(not isinstance(x,slice) for x in k[-2:]):
		# 	raise NotImplementedError('Only slice is supported for the last two dimensions.')
		if len(k)<len(self.shapes[id]):
			k=k+(slice(None,None,None),)*(len(self.shapes[id])-len(k))
		if len(k)!=len(self.shapes[id]):
			raise TypeError('Incorrect number of dimensions for key.')
		if not isinstance(v,np.ndarray):
			raise TypeError('Only numpy.ndarray is supported.')

		ks=[np.arange(*x.indices(self.shapes[id][i])) if isinstance(x,slice) else x for i,x in enumerate(k)]
		t1=tuple(len(x) for x in ks)
		if v.shape!=t1:
			raise TypeError('Incorrect shape for value. Expected: {}. Actual: {}.'.format(t1,v.shape))
		
		self.cache[id][k]=v
		if not all(k in x for x in self.cache.values()):
			# Wait for all variable caches to be filled
			return

		# Check overlap with past entries
		if self.check_update:
			ksset=[set(x) for x in ks]
			if any(all(len(ksset[y]&x[y])>0 for y in range(len(ksset))) for x in self.past):
				raise NotImplementedError('Rewriting past entries is not supported.')
		# Reconstruct original matrix
		cache=[]
		for xi in range(self.n):
			if len(self.shapes[xi])==2:
				cache.append(self.cache[xi][k])
				continue
			shape=self.shapes[xi][:-2]+tuple([len(x) for x in ks])
			t1=np.zeros(shape,dtype=self.cache[list(itertools.islice(filter(lambda x:x[0]==xi,self.cache),1))[0]][k].dtype)
			mask=np.zeros(shape,dtype=bool)
			for xj in filter(lambda x:isinstance(x,tuple) and x[0]==xi,self.cache):
				t2=tuple(xj[1:])
				assert self.cache[xj][k].shape==shape[len(t2):]
				t1[t2]=self.cache[xj][k]
				mask[t2]=True
			assert mask.all()
			cache.append(t1)
		# Perform filtering
		idx0=self.filt(ks,*cache)
		if idx0.shape!=list(itertools.islice(self.cache.values(),1))[0][k].shape:
			raise TypeError('Incorrect shape for filtered output.')
		idx0=np.nonzero(idx0)
		# Write to file
		self.write(k,ks,idx0)
		# Cleanup
		if self.check_update:
			self.past.append(ksset)
		for xi in self.cache.values():
			del xi[k]
	@abc.abstractmethod
	def write(self,k,ks,idx0):
		pass

class filtered_sparse_tsv_master_writer2(filtered_sparse_master_writer2):
	def __init__(self,paths:list[str],names:list[OrderedDict[str,list[str]]],filt:Callable,compression=None,nth=16,**ka):
		"""
		names:			Indices of each dimension for each variable
		compression:	Compression method. Accepts: 'gzip', 'lz4', or None (for no compression).
		nth:			Number of threads for compression. Only used for compression='zstd'.
		"""
		if compression=='gzip':
			import gzip
			self.open=partial(gzip.open,mode='wt')
		elif compression=='zstd':
			raise NotImplementedError('zstd compression lib is bugged.')
			import zstandard as zstd
			self.open=partial(zstd.open,mode='w',cctx=zstd.ZstdCompressor(threads=nth))
		elif compression=='lz4':
			import lz4
			self.open=partial(lz4.frame.open,mode='wt')
		elif compression is None:
			self.open=partial(open,mode='w')
		else:
			raise NotImplementedError('Compression method not supported.')
		self.names=names
		assert len(names[0])==shapes[0][-2] and len(names[1])==shapes[0][-1]
		super().__init__(paths,shapes,filt,**ka)
		for xi in self.f.values():
			xi.write('\t'.join(columns)+'\tvalue'+linesep)
		self.columns=columns
	def write(self,k,ks,idx0):
		idx1=[self.names[x][ks[x][idx0[x]]] for x in range(len(idx0))]
		idx1=list(zip(*idx1))
		assert all(len(x)==len(self.columns) for x in idx1)
		idx1=['\t'.join(x)+'\t' for x in idx1]
		for xi in self.cache:
			s=self.cache[xi][k][idx0]
			s=['{:.10G}'.format(x) for x in s]
			s=''.join([x+y+linesep for x,y in zip(idx1,s)])
			self.f[xi].write(s)

class ndarray_csv_writer(base_writer):
	"""
	CSV writer for full matrix.
	"""
	def __init__(self,path:str,shape,index=None,columns=None,compression:Optional[str]=None,nth=16,**ka):
		"""
		path:			Path to output file
		shape:			Shape of matrix
		index,
		columns:		Rows/column names to be saved in output file
		compression:	Compression method. Accepts: 'gzip', 'lz4', or None (for no compression).
		nth:			Number of threads for compression. Only used for compression='zstd'.
		ka:				Keyword arguments to be passed to pandas.DataFrame.to_csv
		"""
		import pandas as pd
		if columns is not None:
			assert len(columns)==shape[1]
		if index is not None:
			assert len(index)==shape[0]
		if len(shape)!=2:
			raise NotImplementedError('Only 2D arrays are supported.')
		self.shape=shape
		self.index=index
		self.ka=ka
		self.f=None
		if compression=='gzip':
			import gzip
			f=gzip.open
		elif compression=='zstd':
			raise NotImplementedError('zstd compression lib is bugged.')
			import zstandard as zstd
			f=partial(zstd.open,cctx=zstd.ZstdCompressor(threads=nth))
		elif compression=='lz4':
			import lz4
			f=lz4.frame.open
		elif compression is None:
			f=open
		else:
			raise NotImplementedError('Compression method not supported.')
		logging.info(f'Writing file {path}')
		self.f=f(path,'wb')
		if columns is not None:
			#Write header
			pd.DataFrame([],columns=columns).to_csv(self.f,index=False,header=True,compression=None,mode='wb',**ka)
		#Index of next element to write
		self.i=(0,0)
	def __del__(self):
		self.close()
	def close(self):
		if hasattr(self,'f') and self.f is not None:
			self.f.close()
			self.f=None
	def __setitem__(self,k,v):
		import numpy as np
		import pandas as pd
		if not isinstance(k,tuple):
			k=(k,)
		if any(not isinstance(x,slice) for x in k):
			raise NotImplementedError('Only slice is supported.')
		if len(k)<len(self.shape):
			k=k+(slice(None,None,None),)*(len(self.shape)-len(k))
		if len(k)!=len(self.shape):
			raise TypeError('Incorrect number of dimensions for key.')
		k=[x.indices(self.shape[i]) for i,x in enumerate(k)]
		if any(x[2]!=1 for x in k):
			raise NotImplementedError('Only step=1 is supported.')
		if any(x[0]!=y for x,y in zip(k,self.i)):
			raise NotImplementedError('Only sequential writing is supported.')
		if k[1][0]>0 or k[1][1]<self.shape[1]:
			raise NotImplementedError('Only full row writing is supported.')
		if not isinstance(v,np.ndarray):
			raise TypeError('Only numpy.ndarray is supported.')
		if v.shape[1]!=self.shape[1]:
			raise TypeError('Incorrect number of columns for value.')
		pd.DataFrame(v,index=self.index[self.i[0]:self.i[0]+v.shape[0]]).to_csv(self.f,index=True,header=False,compression=None,mode='wb',**self.ka)
		self.i=(self.i[0]+v.shape[0],0)

def filter_cis(locs,ids,b,p,r,s,s0,dist:Union[Tuple[int,int],int]=(-1000000,1000000)):
	"""
	Filt function for filtered_sparse_writer or its derivatives to filter cis relations between SNP and gene
	locs:	Locations of SNPs and genes as [np.array(shape=(n_snp,2)) for chr and location,np.array(shape=(n_gene,3)) for start and stop]
	ids:	Indices of SNPs and genes to be filtered
	b,p,r,s,s0:	Values to be filtered
	dist:	Distance bound from SNP to TSS of gene on the gene's strand to be considered cis
	Return:	np.array of shape (n_snp,n_gene) as boolean mask of cis relations
	"""
	import numpy as np

	from .utils.eqtl import find_cis
	assert all(len(x)>0 for x in ids)
	ans=np.zeros([len(x) for x in ids],dtype=bool)
	t1=find_cis([locs[0][ids[0]],locs[1][ids[1]]],dist)
	ans[t1[0],t1[1]]=True
	return ans

assert __name__ != "__main__"
			

		
	
