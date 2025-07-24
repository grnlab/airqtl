#!/usr/bin/python3
# Copyright 2025, Lingfei Wang and Yuhe Wang
#
# This file is part of airqtl.

import abc
from collections.abc import Iterable
from typing import Callable, Optional, Tuple, Union
import functools

from .utils.importing import torch

@functools.total_ordering
class base(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def __init__(self)->None:
		pass
	@abc.abstractmethod
	def __repr__(self)->str:
		pass
	@abc.abstractmethod
	def tensor(self)->torch.Tensor:
		pass
	@abc.abstractmethod
	def __gt__(self,other)->bool:
		pass
	@abc.abstractmethod
	def __eq__(self,other)->bool:
		pass

class air(base):
	"""
	Array of Interleaved Repeats
	Using compressed form to store data with interleaved repeats
	"""
	# List of elementwise operations grouped by the number of operands
	# Each element is a set: {pytorch function}
	HANDLED_FUNCTIONS_elem=[
		{
			torch.abs,
			torch.acos,
			torch.asin,
			torch.atan,
			torch.cos,
			torch.cosh,
			torch.exp,
			torch.log,
			torch.log10,
			torch.neg,
			torch.sin,
			torch.sinh,
			torch.sqrt,
			torch.tan,
			torch.tanh,
		},
		{
			torch.add,
			torch.mul,
			torch.pow,
		},
	]
	HANDLED_FUNCTIONS_other={
	}
	def __init__(self,v:torch.Tensor,repeat:Optional[list[Optional[torch.Tensor]]])->None:
		"""
		v:		data as torch.tensor
		repeat:	Repeat count of each entry in v as [torch.tensor(shape=(v.shape[x],),dtype=int) or None (indicating normal axis) for x in range(v.ndim)].
				If each entry in repeat is None, indicates each entry inf v appears once in the corresponding dimension.
				if repeat is None, indicates all entries of repeat is None
		"""
		self.v=v
		if repeat is None:
			repeat=[None]*v.ndim
		if len(repeat)!=v.ndim:
			raise ValueError('repeat must have the same length as v.ndim.')
		if any(repeat[x] is not None and repeat[x].min()<=0 for x in range(v.ndim)):
			raise ValueError('repeat[x] must be positive.')
		if any(repeat[x] is not None and repeat[x].shape!=(v.shape[x],) for x in range(v.ndim)):
			raise ValueError('length of repeat[x] must have the same length as v.shape[x].')		
		self.r=[x if x is None else x.clone() for x in repeat]
		self._refresh()
	def _refresh(self)->None:
		if self.v.requires_grad:
			raise NotImplementedError('requires_grad not supported.')
		self.requires_grad=self.v.requires_grad
		self.device=self.v.device
		self.dtype=self.v.dtype
		self.ndim=self.v.ndim
		self.n=[torch.cat([torch.tensor([0],dtype=x.dtype,device=x.device,requires_grad=False),x.cumsum(0)]) if x is not None else None for x in self.r]
		self.shape=tuple(int(self.n[x][-1]) if self.n[x] is not None else self.v.shape[x] for x in range(self.ndim))
	def __repr__(self)->str:
		return "AIR(full_shape={}, air_shape={},axes={})".format(self.shape,self.v.shape,','.join([str(x) for x in range(len(self.r)) if self.r[x] is not None]) if self.r is not None else None)
	def reduce(self,inplace:bool=False)->Union[torch.Tensor,'air']:
		"""
		Converts to reduced form.
		"""
		if inplace:
			raise ValueError('inplace not supported.')
		if all(x is None or (x==1).all() for x in self.r):
			return self.v
		return self
	def _resolve_axis(self,axis:Union[list[int],int,None])->list[int]:
		"""
		Resolves negative and None axes to list of integers.
		"""
		if axis is None:
			axis=list(range(self.ndim))
		if isinstance(axis,int):
			axis=[axis]
		if not all(x>=-self.ndim and x<self.ndim for x in axis):
			raise ValueError('Axes must be within range.')
		axis=[x%self.ndim for x in axis]
		return axis
	def tofull(self,axis:Union[list[Optional[int]],int,None]=None)->Union[torch.Tensor,'air']:
		"""
		Converts to full form for the specified axis.
		axis:	Axis or list of axes to convert to full form. If None, converts all axes.
		Return:	Full form of self for the given axes. If all axes are full form, returns torch.tensor. Otherwise, returns the same class.
		"""
		axis=list(filter(lambda x:self.r[x] is not None,self._resolve_axis(axis)))
		if len(axis)==0:
			#No tofull needed
			return self.reduce()
		#Repeat interleaved
		v=self.v
		for xi in axis:
			v=v.repeat_interleave(self.r[xi],dim=xi)
		if any([self.r[x] is not None and x not in axis for x in range(self.ndim)]):
			#Output self class
			return self.__class__(v,[self.r[x] if x not in axis else None for x in range(self.ndim)]).reduce()
		#Output tensor
		return v
	def tensor(self)->torch.Tensor:
		ans=self.tofull()
		assert isinstance(ans,torch.Tensor)
		return ans
	def tofull_elem(self,other:'air')->Tuple[Union[torch.Tensor,'air'],Union[torch.Tensor,'air']]:
		"""
		Converts to full form for elementwise operation with another object of the same class.
		TODO:	Improve performance by doing partial tofull where applicable.
		"""
		if not isinstance(other,self.__class__):
			raise TypeError('Can only apply this operator on the same class.')
		if self.shape!=other.shape:
			raise TypeError('Shape must be the same.')
		#Determine which axes to convert
		axis=list(filter(lambda x:((self.r[x] is None) ^ (other.r[x] is None)) or (self.r[x] is not None and other.r[x] is not None and (len(self.r[x])!=len(other.r[x]) or (self.r[x]!=other.r[x]).any())),range(self.ndim)))
		ans=(self.tofull(axis=axis),other.tofull(axis=axis))
		assert not all(isinstance(x,self.__class__) for x in ans) or all(x is None or y is None or (len(x)==len(y) and (x==y).all()) for x,y in zip(ans[0].r,ans[1].r))
		return ans
	def swapaxes(self,axis0:int,axis1:int)->'air':
		axes=self._resolve_axis([axis0,axis1])
		if axes[0]==axes[1]:
			return self
		axes=[min(axes),max(axes)]
		return self.__class__(self.v.swapaxes(*axes),self.r[:axes[0]]+[self.r[axes[1]]]+self.r[axes[0]+1:axes[1]]+[self.r[axes[0]]]+self.r[axes[1]+1:])
	def to(self,*a,**ka)->'air':
		return self.__class__(self.v.to(*a,**ka),self.r)
	@property
	def T(self)->'air':
		"""
		Swaps the last two axes.
		"""
		return self.swapaxes(-1,-2)
	@property
	def mT(self)->'air':
		"""
		Swaps the last two axes.
		"""
		return self.swapaxes(-1,-2)
	def reshape(self,*shape)->Union['air',torch.Tensor]:
		v=self.v
		sshape=list(self.shape)
		r=list(self.r)
		head=0
		while head<v.ndim and head<len(shape):
			if shape[head]==sshape[head]:
				head+=1
			elif shape[head]==1:
				v=v.unsqueeze(head)
				r.insert(head,None)
				sshape.insert(head,1)
				head+=1
			else:
				raise NotImplementedError('Only reshape to add extra dimensions of size 1 is supported.')
		assert tuple(sshape)==shape
		return self.__class__(v,r).reduce()
	def __getitem__(self,key:Tuple[Iterable[int],int,slice])->Union['air',torch.Tensor]:
		if not isinstance(key,tuple):
			key=[key]
		key=[int(x) if isinstance(x,torch.Tensor) and x.ndim==0 else x for x in key]
		if len(key)>self.ndim:
			raise ValueError('Too many indices.')
		if any(x is Ellipsis for x in key):
			raise NotImplementedError('Ellipsis not supported.')
		if len(list(filter(lambda x:not isinstance(x,(slice,int)),key)))>1:
			raise NotImplementedError('Only one index can be non-slice, non-integer.')
		v=self.v
		r=list(self.r)
		reduce_dims=[]
		for xi in range(len(key)):
			k=key[xi]
			if isinstance(k,slice):
				k=k.indices(self.shape[xi])
				if k[0]==0 and k[1]==self.shape[xi] and k[2]==1:
					continue
				if k[2]==1:
					if r[xi] is None or k[1]<=k[0]:
						v=v.swapaxes(0,xi)[slice(*k)].swapaxes(0,xi)
						r[xi]=None
						continue
					t1=torch.tensor([torch.searchsorted(self.n[xi][1:],k[0],side='right'),torch.searchsorted(self.n[xi][1:],k[1],side='left')],dtype=int,device=self.n[xi].device)
					if t1[0]==t1[1]:
						v=v.swapaxes(0,xi)[t1[0]:t1[0]+1].swapaxes(0,xi)
						r[xi]=torch.ones([1],dtype=r[xi].dtype,device=r[xi].device,requires_grad=False)*(k[1]-k[0])
						assert r[xi].sum()==k[1]-k[0]
						continue
					t2=(self.n[xi][t1[0]+1]-k[0],k[1]-self.n[xi][t1[1]])
					v=v.swapaxes(0,xi)[t1[0]:t1[1]+1].swapaxes(0,xi)
					r[xi]=r[xi][t1[0]:t1[1]+1].clone()
					r[xi][0]=t2[0]
					r[xi][-1]=t2[1]
					assert len(r[xi])==v.shape[xi]
					assert r[xi].sum()==k[1]-k[0]
					continue
				raise NotImplementedError('Step not supported.')
			if isinstance(k,int):
				reduce_dims.append(xi)
				k=[k]
			k=[int(x) for x in k]
			#key is iterable
			if r[xi] is None:
				v=v.swapaxes(xi,0)[k].swapaxes(xi,0)
				continue
			t2=torch.tensor(k,device=self.n[xi].device)%self.shape[xi]
			t1=torch.searchsorted(self.n[xi][1:],t2,side='right')
			t2=t2-self.n[xi][t1]
			v=v.swapaxes(0,xi)[t1].swapaxes(0,xi)
			r[xi]=None
		#Reduce dimensions
		if len(reduce_dims)>0:
			assert all(v.shape[x]==1 for x in reduce_dims)
			assert all(r[x] is None for x in reduce_dims)
			v=v.squeeze(reduce_dims)
			r=[r[x] for x in range(self.ndim) if x not in reduce_dims]
		if v.ndim==0:
			return v
		return self.__class__(v,r).reduce()
	@classmethod
	def op_elem1(cls,func:Callable,op1:'air',*a,**ka)->'air':
		"""
		Elementwise operation with one operand.
		"""
		return cls(func(op1.v,*a,**ka),op1.r)
	@classmethod
	def op_elem2(cls,func:Callable,op1:Union['air',torch.Tensor,int,float],op2:Union['air',torch.Tensor,int,float],*a,**ka)->Union['air',torch.Tensor]:
		"""
		Elementwise operation with two operands.
		"""
		if not any(isinstance(x,cls) for x in [op1,op2]):
			return NotImplemented
		if isinstance(op1,cls) and isinstance(op2,cls):
			o1,o2=op1.tofull_elem(op2)
			if isinstance(o1,torch.Tensor) or isinstance(o2,torch.Tensor):
				return func(o1,o2,*a,**ka)
			if isinstance(o1,cls) and isinstance(o2,cls):
				ans=func(o1.v,o2.v,*a,**ka)
				ans=cls(ans,o1.r)
				return ans.reduce()
			assert False
		if isinstance(op1,cls):
			if isinstance(op2,torch.Tensor):
				return func(op1.tensor(),op2)
			if isinstance(op2,(int,float)):
				return cls(func(op1.v,op2,*a,**ka),op1.r).reduce()
		if isinstance(op2,cls):
			if isinstance(op1,torch.Tensor):
				return func(op1,op2.tensor(),*a,**ka)
			if isinstance(op1,(int,float)):
				return cls(func(op1,op2.v,*a,**ka),op2.r).reduce()
		return NotImplemented
	def toreduce(self,method:str,axis:dict[int,torch.Tensor])->'air':
		"""
		Converts full form to reduced form.
		method:	Method to reduce size. Accepts:
			'mean':	Average
			'sum':	Sum
		axis:	Axes to convert to reduced form as {axis:repeat,...}.
		Return:	Reduced form of self for the given axes.
		"""
		if method not in {'mean','sum'}:
			raise ValueError(f'Unknown method {method}.')
		if len(axis)==0:
			return self.reduce()
		axis=list(zip(*list(axis.items())))
		axis[0]=self._resolve_axis(axis[0])
		axis=dict(zip(*axis))
		v=self.v
		r=list(self.r)
		for xi in axis:
			if r[xi] is None:
				d=torch.zeros(v.shape[:xi]+(len(axis[xi]),)+v.shape[xi+1:],dtype=v.dtype,device=v.device,requires_grad=self.requires_grad)
				t1=torch.repeat_interleave(torch.arange(len(axis[xi]),device=axis[xi].device,requires_grad=False),axis[xi])
				d.index_add_(xi,t1,v)
				if method=='mean':
					d=(d.swapaxes(xi,-1)/axis[xi]).swapaxes(xi,-1)
				v=d
				r[xi]=axis[xi]
				changed=True
			else:
				if (r[xi]!=axis[xi]).any():
					raise NotImplementedError('Different repeat counts not supported.')
				if method=='sum':
					v=(v.swapaxes(xi,-1)*axis[xi]).swapaxes(xi,-1)
					changed=True
		return self.__class__(v,r) if changed else self
	def __add__(self,other:Union['air',torch.Tensor,int,float])->Union['air',torch.Tensor]:
		from operator import add as op
		return self.op_elem2(op,self,other)
	def __mul__(self,other:Union['air',torch.Tensor,int,float])->Union['air',torch.Tensor]:
		from operator import mul as op
		return self.op_elem2(op,self,other)
	def __gt__(self,other:Union['air',torch.Tensor,int,float])->Union['air',torch.Tensor]:
		from operator import gt as op
		return self.op_elem2(op,self,other)
	def __eq__(self,other:Union['air',torch.Tensor,int,float])->Union['air',torch.Tensor]:
		from operator import eq as op
		return self.op_elem2(op,self,other)
	def __pow__(self,other:Union['air',torch.Tensor,int,float])->Union['air',torch.Tensor]:
		from operator import pow as op
		return self.op_elem2(op,self,other)
	def __matmul__(self,other:Union['air',torch.Tensor,'composite'])->Union['air',torch.Tensor,'composite']:
		if isinstance(other,torch.Tensor):
			return self@self.__class__(other,None)
		if not isinstance(other,self.__class__):
			return NotImplemented
		if self.ndim>=2 and other.ndim>=2:
			#Matmul dimension: self[-1] and other[-2]
			if self.shape[-1]!=other.shape[-2]:
				raise TypeError('Matmul dimension must be the same.')
			if any(self.r[x] is not None and other.r[x] is not None and (len(self.r[x])!=len(other.r[x]) or (self.r[x]!=other.r[x]).any()) for x in range(min(self.ndim,other.ndim)-2)):
				raise NotImplementedError('Different forms or repeat counts not supported for uninvolved/broadcasted axes in matmul.')
			if any(self.shape[-x]!=other.shape[-x] for x in range(3,min(self.ndim,other.ndim)+1)):
				raise NotImplementedError('Different sizes not supported for uninvolved/broadcasted axes in matmul.')
			axis_tofull=[x for x in range(min(self.ndim,other.ndim)-2) if (self.r[x] is not None) ^ (other.r[x] is not None)]
			if len(axis_tofull)>0:
				return self.tofull(axis=axis_tofull)@other.tofull(axis=axis_tofull)
			if self.r[-1] is not None and other.r[-2] is None:
				return self@other.toreduce('mean',{-2:self.r[-1]})
			if self.r[-1] is None and other.r[-2] is not None:
				return self.toreduce('mean',{-1:other.r[-2]})@other
			if self.r[-1] is not None and other.r[-2] is not None:
				if len(self.r[-1])!=len(other.r[-2]) or (self.r[-1]!=other.r[-2]).any():
					raise NotImplementedError('Different repeat counts not supported.')
				v=(self.v*self.r[-1])@other.v
			else:
				v=self.v@other.v
			r=([self.r[x] for x in range(self.ndim-2)] if self.ndim>other.ndim else [other.r[x] for x in range(other.ndim-2)])+[self.r[-2],other.r[-1]]
			return self.__class__(v,r).reduce()
		return NotImplemented
	def __radd__(self,other:Union['air',torch.Tensor,float,int])->Union['air',torch.Tensor]:
		return self.__add__(other)
	def __rmul__(self,other:Union['air',torch.Tensor,float,int])->Union['air',torch.Tensor]:
		return self.__mul__(other)
	def __rpow__(self,other:Union['air',torch.Tensor,float,int])->Union['air',torch.Tensor]:
		from operator import pow as op
		return self.op_elem2(op,other,self)
	def __rmatmul__(self,other:Union['air',torch.Tensor])->Union['air',torch.Tensor]:
		if isinstance(other,torch.Tensor):
			return self.__class__(other,None)@self
		if isinstance(other,self.__class__):
			return other.__matmul__(self)
		return NotImplemented
	def svd(self, full_matrices:bool=False, out=None,driver=None)->Union[torch.Tensor, 'air']:
		"""
    	Computes eigenvalues and eigenvectors for the full (per-sample) kinship matrix
    	using PyTorch only.
    	"""
		from functools import partial
		if full_matrices:
			raise NotImplementedError('Full matrices not supported.')
		if out is not None:
			raise NotImplementedError('out not supported.')
		if self.v.ndim != 2:
			raise ValueError('SVD for matrices with !=2 dimensions is not supported.')

		func=partial(torch.linalg.svd,full_matrices=False,driver=driver)
     	#non-compressed matrix can't accelerate
		if all(x is None for x in self.r):
			return func(self.v)

		temp_repeat=[1 if self.r[x] is None else self.r[x] for x in range(len(self.v.shape))]
		#compressed matrix can be accelerated

		scaled = ((self.v * (torch.sqrt(temp_repeat[1]))).T * (torch.sqrt(temp_repeat[0]))).T
		U, S, VT = func(scaled)

		return (
			self.__class__((U.T / (torch.sqrt(temp_repeat[0]))).T,[self.r[0],None]), 
   			S, 
      		self.__class__(VT / (torch.sqrt(temp_repeat[1])), [None, self.r[1]]),
		)

	def sum(self,axis:Union[int,list[int],None]=None)->Union[torch.Tensor,'air']:
		"""
		Sums along axis.
		"""
		axis=sorted(self._resolve_axis(axis),reverse=True)
		if len(axis)==0:
			return self.reduce()
		v=self.v
		for xi in axis:
			if self.r[xi] is None:
				v=v.sum(axis=xi)
			else:
				v=(v.swapaxes(xi,-1)*self.r[xi]).swapaxes(xi,-1).sum(axis=xi)
		r=[self.r[x] for x in filter(lambda y:y not in axis,range(self.ndim))]
		return self.__class__(v,r).reduce()
	@classmethod
	def __torch_function__(cls, func, types, args=(), kwargs={}):
		if len(args)<=len(cls.HANDLED_FUNCTIONS_elem) and func in cls.HANDLED_FUNCTIONS_elem[len(args)-1] and any(isinstance(x, cls) for x in args):
			if len(args)==1:
				#Broadcasted elementwise operation
				return cls.op_elem1(func,*args,**kwargs)
			if len(args)==2:
				#Elementwise operation with two operands
				return cls.op_elem2(func,*args,**kwargs)
		if func in cls.HANDLED_FUNCTIONS_other and any(issubclass(x, cls) for x in args):
			#Other operations
			return cls.HANDLED_FUNCTIONS_other[func](*args, **kwargs)
		return NotImplemented



class composite(base):
	HANDLED_FUNCTIONS={
	}
	def __init__(self,vs:list[Union['composite',air,torch.Tensor]],axis:int)->None:
		"""
		Composite of air and/or torch.Tensor.
		vs:		List of air and/or torch.Tensor.
		axis:	Axis to composite.
		"""
		if len(vs)<1:
			raise ValueError('At least one variable is needed.')
		if any(x.ndim!=vs[0].ndim for x in vs[1:]):
			raise ValueError('All variables must have the same number of dimensions.')
		if axis>=vs[0].ndim or axis<-vs[0].ndim:
			raise ValueError('Axis must be within range.')
		if axis<0:
			axis=vs[0].ndim+axis
			assert axis>=0
		if any(x.shape[:axis]+x.shape[axis+1:]!=vs[0].shape[:axis]+vs[0].shape[axis+1:] for x in vs[1:]):
			raise ValueError('All variables must have the same shape except for the composite axis.')
		if any(x.dtype!=vs[0].dtype for x in vs[1:]):
			raise ValueError('All variables must have the same dtype.')
		if any(x.device!=vs[0].device for x in vs[1:]):
			raise ValueError('All variables must have the same device.')
		if any(x.requires_grad!=vs[0].requires_grad for x in vs[1:]):
			raise ValueError('All variables must have the same requires_grad.')
		self.vs=vs
		self.axis=axis
		self._refresh()
		self.reduce(inplace=True)
	def _refresh(self)->None:
		"""
		(Re)-compute other attributes based on vs and axis.
		"""
		self.dtype=self.vs[0].dtype
		self.device=self.vs[0].device
		self.requires_grad=self.vs[0].requires_grad
		self.ndim=self.vs[0].ndim
		self.shape=self.vs[0].shape[:self.axis]+(sum(x.shape[self.axis] for x in self.vs),)+self.vs[0].shape[self.axis+1:]
		self.sizes=torch.tensor([x.shape[self.axis] for x in self.vs],dtype=torch.int,device=self.device,requires_grad=False)
		self.sizesc=torch.cat([torch.tensor([0],dtype=torch.int,device=self.device,requires_grad=False),self.sizes.cumsum(0)],axis=0)
	def __repr__(self)->str:
		return "AIRComposite(shape={},axis={},N={})".format(self.shape,self.axis,len(self.vs))
	def _reduce(self)->Union[Tuple[list[Union[air,torch.Tensor,'composite']],int],air,torch.Tensor,'composite',None]:
		"""
		Simplifies the composite to a new object
		"""
		import itertools
		if len(self.vs)==1:
			if isinstance(self.vs[0],self.__class__):
				ans=self.vs[0].reduce()
			else:
				ans=self.vs[0]
			return ans
		if all(isinstance(x,torch.Tensor) for x in self.vs):
			return torch.cat(self.vs,dim=self.axis)
		vs=[x.reduce() if isinstance(x,self.__class__) else x for x in self.vs]
		vs=list(itertools.chain.from_iterable([x.vs if isinstance(x,self.__class__) and x.axis==self.axis else [x] for x in vs]))
		ans=[vs[0]]
		for xi in range(1,len(vs)):
			if isinstance(ans[-1],torch.Tensor) and isinstance(vs[xi],torch.Tensor):
				ans[-1]=torch.cat([ans[-1],vs[xi]],dim=self.axis)
			else:
				ans.append(vs[xi])
		return (ans,self.axis)
	def reduce(self,inplace:bool=False)->Union[air,torch.Tensor,'composite',None]:
		"""
		Simplifies the composite to a new object or in place.
		"""
		ans=self._reduce()
		if ans is None:
			return None if inplace else self
		if isinstance(ans,tuple):
			if inplace:
				self.vs,self.axis=ans
				self._refresh()
				return
			return self.__class__(*ans)
		if isinstance(ans,(torch.Tensor,air)):
			if inplace:
				self.vs=[ans]
				self.axis=0
				self._refresh()
				return
			return ans
		if isinstance(ans,self.__class__):
			if inplace:
				self.vs=ans.vs
				self.axis=ans.axis
				self._refresh()
				return
			return ans
		raise TypeError(f'Unsupported type {type(ans)}.')
	def _resolve_axis(self,axis:int)->int:
		"""
		Resolves axis to a valid axis.
		"""
		#Define axis
		if axis<-self.ndim or axis>=self.ndim:
			raise ValueError('Axis must be within range.')
		return axis%self.ndim
	def tofull(self,axis:Union[int,list[int],None]=None)->Union[air,torch.Tensor,'composite']:
		"""
		Converts to full form for the specified axis.
		axis:	Axis or list of axes to convert to full form. If None, converts all axes.
		Return:	Full form of self for the given axes. If all axes are full form, returns torch.Tensor. Otherwise, returns the same class.
		"""
		vs=[x.tofull(axis=axis) if not isinstance(x,torch.Tensor) else x for x in self.vs]
		return self.__class__(vs,self.axis).reduce()
	def tensor(self)->torch.Tensor:
		return torch.cat([x.tensor() if not isinstance(x,torch.Tensor) else x for x in self.vs],dim=self.axis)
	def swapaxes(self,axis0:int,axis1:int)->Union[air,torch.Tensor,'composite']:
		axis0,axis1=[self._resolve_axis(x) for x in [axis0,axis1]]
		if axis0==axis1:
			return self
		vs=[x.swapaxes(axis0,axis1) for x in self.vs]
		return self.__class__(vs,self.axis if self.axis not in {axis0,axis1} else (axis0+axis1-self.axis)).reduce()
	def to(self,*a,**ka)->Union[air,torch.Tensor,'composite']:
		return self.__class__([x.to(*a,**ka) for x in self.vs],self.axis).reduce()
	@property
	def T(self)->'composite':
		"""
		Swaps the last two axes.
		"""
		return self.swapaxes(-1,-2)
	@property
	def mT(self)->'composite':
		"""
		Swaps the last two axes.
		"""
		return self.swapaxes(-1,-2)
	def __getitem__(self,key)->Union[air,torch.Tensor,'composite']:
		if not isinstance(key,tuple):
			key=(key,)
		key=[int(x) if isinstance(x,torch.Tensor) and x.ndim==0 else x for x in key]
		if Ellipsis in key:
			raise NotImplementedError('Ellipsis not supported.')
		if len(key)>self.ndim:
			raise ValueError('Too many indices.')
		if len(list(filter(lambda x:not isinstance(x,(slice,int)),key)))>1:
			raise NotImplementedError('Only one index can be non-slice, non-integer.')
		axis=self.axis-len(list(filter(lambda x:isinstance(x,int),key[:self.axis])))
		if len(key)<=self.axis:
			return self.__class__([x[tuple(key)] for x in self.vs],axis).reduce()
		ans=tuple(key[:self.axis]+[slice(None)]+key[self.axis+1:])
		ans=[x[ans] for x in self.vs]
		k=key[self.axis]
		if isinstance(k,slice):
			t1=k.indices(self.sizesc[-1])
			if t1[2]!=1:
				raise NotImplementedError('Step not supported.')
			if t1[0]==0 and t1[1]==self.sizesc[-1]:
				if all(key[x]==slice(None) for x in range(len(key)) if x!=self.axis):
					return self
				return self.__class__(ans,axis).reduce()
			if t1[1]<=t1[0]:
				t2=list(ans[0].shape)
				t2[axis]=0
				return torch.zeros(t2,device=self.vs[0].device,dtype=self.vs[0].dtype,requires_grad=self.requires_grad)
			t2=torch.tensor([torch.searchsorted(self.sizesc,t1[0],side='right'),torch.searchsorted(self.sizesc,t1[1],side='left')],dtype=int,device=self.sizesc.device)-1
			t1=torch.tensor(t1[:2],dtype=int,device=self.device,requires_grad=False)
			t1-=self.sizesc[t2]
			ans=ans[t2[0]:t2[1]+1]
			if t2[0]==t2[1]:
				# Single entry output
				return ans[0][tuple([slice(None)]*axis+[slice(t1[0],t1[1])])]
			ans[0]=ans[0][tuple([slice(None)]*axis+[slice(t1[0],None)])]
			ans[-1]=ans[-1][tuple([slice(None)]*axis+[slice(t1[1])])]
			return self.__class__(ans,axis).reduce()
		reduce_dim=False
		if isinstance(k,int):
			k=[k]
			reduce_dim=True
		#key is iterable
		k=torch.tensor(k,device=self.sizesc.device,dtype=int)
		k%=self.sizesc[-1]
		t1=torch.searchsorted(self.sizesc,k,side='right')-1
		t1=[k-self.sizesc[t1],t1]
		if reduce_dim:
			return ans[t1[1][0]][tuple([slice(None)]*axis+[t1[0][0]])]
		ans=[ans[t1[1][x]][tuple([slice(None)]*axis+[[t1[0][x]]])] for x in range(t1[1].shape[0])]
		return self.__class__(ans,axis).reduce()
	def op_elem(self,other:Union[air,torch.Tensor,'composite',int,float],op:Callable)->Union[air,torch.Tensor,'composite']:
		"""
		Elementwise operation.
		"""
		if isinstance(other,(int,float)):
			return self.__class__([op(x,other) for x in self.vs],self.axis)
		if self.shape!=other.shape:
			raise TypeError(f'Operation {op} has different shapes: {self.shape} and {other.shape}.')
		if isinstance(other,(torch.Tensor,air)):
			t1=[slice(None)]*self.axis
			return self.__class__([op(self.vs[x],other[tuple(t1+[slice(self.sizesc[x],self.sizesc[x+1])])]) for x in range(len(self.vs))],self.axis).reduce()
		if isinstance(other,self.__class__):
			if other.axis!=self.axis or len(other.vs)!=len(self.vs) or (other.sizes!=self.sizes).any():
				t1=[slice(None)]*self.axis
				return self.__class__([op(self.vs[x],other[tuple(t1+[slice(self.sizesc[x],self.sizesc[x+1])])]) for x in range(len(self.vs))],self.axis).reduce()
			return self.__class__([op(self.vs[x],other.vs[x]) for x in range(len(self.vs))],self.axis).reduce()
		return NotImplemented
	def __add__(self,other:Union[air,torch.Tensor,'composite',int,float])->Union[air,torch.Tensor,'composite']:
		from operator import add as op
		return self.op_elem(other,op)
	def __mul__(self,other:Union[air,torch.Tensor,'composite',int,float])->Union[air,torch.Tensor,'composite']:
		from operator import mul as op
		return self.op_elem(other,op)
	def __gt__(self,other:Union[air,torch.Tensor,'composite',int,float])->Union[air,torch.Tensor,'composite']:
		from operator import gt as op
		return self.op_elem(other,op)
	def __eq__(self,other:Union[air,torch.Tensor,'composite',int,float])->Union[air,torch.Tensor,'composite']:
		from operator import eq as op
		return self.op_elem(other,op)
	def __pow__(self,other:Union[air,torch.Tensor,'composite',int,float])->Union[air,torch.Tensor,'composite']:
		from operator import pow as op
		return self.op_elem(other,op)
	def __matmul__(self,other:Union[air,torch.Tensor,'composite'])->Union[air,torch.Tensor,'composite']:
		#Matmul dimension: self[-1] and other[-2]
		from functools import reduce
		from operator import add
		if self.ndim<2 or other.ndim<2:
			raise NotImplementedError('Matmul requires at least 2D.')
		if self.shape[-1]!=other.shape[-2]:
			raise TypeError('Matmul dimension must be the same.')
		if any(self.shape[-x]!=other.shape[-x] for x in range(3,min(self.ndim,other.ndim)+1)):
			raise NotImplementedError('Different sizes not supported for uninvolved/broadcasted axes in matmul.')
		if self.axis!=self.ndim-1:
			if self.axis==self.ndim-2:
				return self.__class__([x@other for x in self.vs],self.axis).reduce()
			return self.__class__([self.vs[x]@other.swapaxes(self.axis,0)[self.sizesc[x]:self.sizesc[x+1]].swapaxes(self.axis,0) for x in range(len(self.vs))],self.axis).reduce()
		if isinstance(other,self.__class__) and other.axis==other.ndim-2 and len(self.sizes)==len(other.sizes) and (self.sizes==other.sizes).all():
			ans=reduce(add,[x@y for x,y in zip(self.vs,other.vs)])
		else:
			t1=[slice(None)]*(len(other.shape)-2)
			other=[other[tuple(t1+[slice(self.sizesc[x],self.sizesc[x+1])])] for x in range(len(self.vs))]
			ans=reduce(add,[x@y for x,y in zip(self.vs,other)])
		if isinstance(ans,self.__class__):
			ans=ans.reduce()
		return ans
	def __radd__(self,other:Union[air,torch.Tensor,float,int])->Union[air,torch.Tensor,'composite']:
		return self.__add__(other)
	def __rmul__(self,other:Union[air,torch.Tensor,float,int])->Union[air,torch.Tensor,'composite']:
		return self.__mul__(other)
	def __rmatmul__(self,other:Union[air,torch.Tensor,'composite'])->Union[air,torch.Tensor,'composite']:
		if isinstance(other,(torch.Tensor,air)):
			return (self.mT@other.mT).mT
		if isinstance(other,self.__class__):
			return other.__matmul__(self)
		return NotImplemented
	def sum(self,axis:int=None)->Union[air,torch.Tensor,'composite',float,int]:
		"""
		Sums along axis.
		"""
		from functools import reduce
		from operator import add
		if axis is None:
			raise NotImplementedError('Sum over all axes not supported.')
		if not isinstance(axis,int):
			raise NotImplementedError('Sum over multiple axes not supported.')
		axis=self._resolve_axis(axis)
		if axis==self.axis:
			return reduce(add,[x.sum(axis=axis) for x in self.vs])
		return self.__class__([x.sum(axis=axis) for x in self.vs],self.axis-(axis<self.axis)).reduce()


assert __name__ != "__main__"
