#!/usr/bin/python3
# Copyright 2025, Lingfei Wang and Yuhe Wang
#
# This file is part of airqtl.

import numpy as np
import pytest
import torch

from airqtl import air

torch.manual_seed(12345)
np.random.seed(12345)
torch.use_deterministic_algorithms(True)
torch.set_default_device('cpu')
#Default dtypes
dtype_float=torch.float64
dtype_int=torch.int32
ntests=500

pa={
	#Probability of not repeating for each dimension
	'probnrepeat':0.3,
	#List of probability of breaking the repeat for each element for each dimension
	'probbreak':0.3,
}
class Test_air:
	target_cls=air.air
	@staticmethod
	def gen_repeat(shape,p):
		"""
		Generate repeat for air.
		shape:			Shape of full tensor
		probnrepeat:	Probability of not repeating for each dimension
		probbreak:		List of probability of breaking the repeat for each element for each dimension
		Return:
		repeat:		Repeat for air.
		"""
		n=len(shape)
		probnrepeat=p['probnrepeat']
		probbreak=p['probbreak']
		repeat=torch.rand(n)<probnrepeat
		repeat=[None if repeat[x] else torch.cat([torch.tensor([0],dtype=dtype_int,requires_grad=False),(torch.rand(shape[x]-1,dtype=dtype_float,requires_grad=False)<probbreak).nonzero(as_tuple=True)[0]+1,torch.tensor([shape[x]],dtype=dtype_int,requires_grad=False)]) for x in range(n)]
		repeat=[None if x is None else x[1:]-x[:-1] for x in repeat]
		repeat=[y if y is not None or torch.rand(1)<0.5 else torch.ones(shape[x],dtype=dtype_int,requires_grad=False) for x,y in enumerate(repeat)]
		assert all(x is None or x.min()>0 for x in repeat)
		assert all(x is None or x.sum()==shape[i] for i,x in enumerate(repeat))
		return repeat
	@classmethod
	def gen_data(cls,shape,repeat=None,return_repeat=False,p=pa,dtype=dtype_float):
		"""
		Generate air and torch.Tensor containing the same data for testing.
		shape:		Full shape of tensor.
		repeat:		The repeat for air for each dimension. Each entry for one dimension can be:
			True:	Indicating the repeat should be generated with gen_repeat
			None or False:	Indicating the repeat once for each element
			torch.Tensor:	Indicating the number of repeats for each element
			If repeat is None, indicating a list of Trues for each dimension.
		return_repeat:	Indicating whether to return repeat
		p:			Parameters. See pa.
		dtype:		Dtype of tensor.
		Return:
		air:		air object.
		tensor:		torch.Tensor.
		repeat:		Repeat for air (if return_repeat is True)
		"""
		n=len(shape)
		if repeat is None:
			repeat=[True]*n
		assert len(repeat)==n
		assert all(repeat[x] in {None,True} or repeat[x].sum()==shape[x] for x in range(n))
		if any(isinstance(x,(bool,np.bool_)) and x for x in repeat):
			t1=Test_air.gen_repeat(shape,p)
			repeat=[t1[x] if isinstance(repeat[x],(bool,np.bool_)) and repeat[x] else repeat[x] for x in range(n)]
		assert len(repeat)==n
		assert all(repeat[x] is None or repeat[x].sum()==shape[x] for x in range(n))
		shape1=[shape[x] if repeat[x] is None else len(repeat[x]) for x in range(n)]
		d1=torch.randn(*shape1,dtype=dtype_float,requires_grad=False).to(dtype)
		d0=d1
		for xi in filter(lambda x:repeat[x] is not None,range(n)):
			d0=torch.repeat_interleave(d0,repeat[xi],dim=xi)
		assert all(y is None or d1.shape[x]==len(y) for x,y in enumerate(repeat))
		d1=cls.target_cls(d1,repeat)
		assert d0.isnan().sum()==0
		assert d1.v.isnan().sum()==0
		if return_repeat:
			return d1,d0,[x if x is None else x.clone() for x in repeat]
		return d1,d0
	def test_tensor(self,ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests,**ka):
		import itertools
		for ndim,_ in itertools.product(ndims,range(n)):
			shape=np.random.choice(sizes,ndim)
			d1,d0=self.gen_data(shape,**ka)
			assert (d1.tensor()==d0).all()
	def test_getitem(self,ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests,**ka):
		import itertools
		for ndim,_ in itertools.product(ndims,range(n)):
			shape=np.random.choice(sizes,ndim)
			d1,d0=self.gen_data(shape,**ka)
			#Generate random key
			key=np.random.rand(ndim)
			key2=[]
			for xi0,xi in enumerate(key):
				if xi<0.2:
					key2.append(slice(None))
				elif xi<0.4:
					key2.append(np.random.randint(-shape[xi0]+1,shape[xi0]))
				elif xi<0.6:
					key2.append(list(np.random.choice(np.arange(-shape[xi0]+1,shape[xi0]),np.random.randint(1,shape[xi0]+1),replace=True)))
				else:
					key2.append(slice(np.random.randint(-shape[xi0]+1,shape[xi0]),np.random.randint(-shape[xi0]+1,shape[xi0])))
			if len(list(filter(lambda x:not isinstance(x,(slice,int)),key2)))>1:
				continue
			key2=tuple(key2)
			ans=[d0[key2],d1[key2]]
			assert ans[0].shape==ans[1].shape
			if not isinstance(ans[1],torch.Tensor):
				ans[1]=ans[1].tensor()
			assert (ans[0]==ans[1]).all()
	def _test_op_(self,operator,otype='elem',ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests//5,unlimited=False,dtype=dtype_float,**ka):
		"""
		Tests operator on air.
		operator:	Operator to test
		otype:		Operator type. Accepts:
			'none':		Operator accepts no other operand
			'atom':		Operator with single value
			'elem':		Elementwise operator
			'matmul':	Matrix multiplication
		ndims:	Number of dimensions to test
		n:		Number of random tests in each case
		unlimited:	Do not limit the other operand's repeat pattern. Actual limit depends on otype. Only works for elem and matmul.
		"""
		import itertools
		assert otype!='matmul' or 1 not in ndims
		for ndim,_ in itertools.product(ndims,range(n)):
			#Generate data
			if otype=='none':
				shape=np.random.choice(sizes,ndim)
				d1,d0=self.gen_data(shape,dtype=dtype,**ka)
				a=[[d0],[d1]]
			elif otype=='atom':
				shape=np.random.choice(sizes,ndim)
				d1,d0=self.gen_data(shape,dtype=dtype,**ka)
				e0=torch.randn(1,dtype=dtype_float,requires_grad=False).to(dtype)
				a=[[d0,e0],[d1,e0]]
			elif otype=='elem':
				shape=np.random.choice(sizes,ndim)
				if unlimited:
					d1,d0=self.gen_data(shape,dtype=dtype,**ka)
					e1,e0=self.gen_data(shape,dtype=dtype,**ka)
				else:
					d1,d0,repeat=self.gen_data(shape,return_repeat=True,**ka)
					if np.random.rand()<0.5:
						repeat[np.random.choice(len(repeat),1)[0]]=None
					e1,e0=self.gen_data(shape,repeat=repeat,dtype=dtype)
				a=list(itertools.product([d0,d1],[e0,e1]))
			elif otype=='matmul':
				shapes=[np.random.choice(sizes,ndim) for _ in range(2)]
				for xi in range(len(shapes[1])-2):
					shapes[1][xi]=shapes[0][xi]
				shapes[1][-2]=shapes[0][-1]
				if unlimited:
					d1,d0=self.gen_data(shapes[0],dtype=dtype,**ka)
					e1,e0=self.gen_data(shapes[1],dtype=dtype,**ka)
				else:
					d1,d0,repeat=self.gen_data(shapes[0],return_repeat=True,dtype=dtype,**ka)
					repeat=[repeat[x] if x<len(shapes[1])-2 else (repeat[-1] if x==len(shapes[1])-2 else True) for x in range(len(shapes[1]))]
					e1,e0=self.gen_data(shapes[1],repeat=repeat,dtype=dtype,**ka)
				a=[]
				for xi,xj in itertools.product([d0,d1],[e0,e1]):
					if unlimited:
						try:
							a.append([xi,xj])
						except NotImplementedError:
							continue
					else:
						a.append([xi,xj])
			else:
				assert False
			#Test
			assert len(a)>1
			try:
				ans0=operator(*a[0])
				e0=None
			except Exception as err:
				e0=err
			if e0 is not None:
				for xi in a[1:]:
					with pytest.raises(type(e0)):
						operator(*xi)
			else:
				t1=torch.isnan(ans0)
				for xi in a[1:]:
					try:
						ans1=operator(*xi)
					except NotImplementedError:
						continue
					assert ans1.shape==ans0.shape
					if isinstance(ans1,self.target_cls):
						ans1=ans1.tensor()
					assert ans1.shape==ans0.shape
					assert not (t1^torch.isnan(ans1)).any()
					assert (torch.isclose(ans1,ans0)|t1).all()
	def _test_agg_(self,func,ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests//5,dtype=dtype_float,**ka):
		"""
		Tests aggregation function on air.
		func:		Aggregation function to test
		ndims:	Number of dimensions to test
		n:		Number of random tests in each case
		"""
		import itertools
		re_dtype={
			(torch.mean,dtype_int):dtype_float,
		}
		for ndim,_ in itertools.product(ndims,range(n)):
			#Generate data
			shape=np.random.choice(sizes,ndim)
			axis=np.nonzero(np.random.rand(ndim)<0.3)[0]
			if len(axis)>0:
				if func==torch.prod:
					axis=int(np.random.choice(axis))
					axis-=(np.random.rand()<0.3)*ndim
				else:
					axis-=(np.random.rand(len(axis))<0.3)*ndim
					np.random.shuffle(axis)
					if len(axis)==1 and np.random.rand()<0.5:
						axis=int(axis[0])
					else:
						axis=[int(x) for x in axis]
			else:
				axis=None
			d1,d0=self.gen_data(shape,dtype=dtype,**ka)
			if (func,dtype) in re_dtype:
				d0=d0.to(re_dtype[(func,dtype)])
			#Test
			try:
				ans0=func(d0,axis=axis) if func!=torch.prod or axis is not None else func(d0)
				e0=None
			except Exception as err:
				e0=err
			if e0 is not None:
				with pytest.raises(type(e0)):
					func(d1,axis=axis)
			else:
				t1=torch.isnan(ans0)
				ans1=func(d1,axis=axis)
				assert ans1.shape==ans0.shape
				if isinstance(ans1,self.target_cls):
					ans1=ans1.tensor()
				assert ans1.shape==ans0.shape
				if (func,dtype) in re_dtype:
					ans1=ans1.to(re_dtype[(func,dtype)])
				assert not (t1^torch.isnan(ans1)).any()
				assert (torch.isclose(ans1,ans0)|t1).all()
	def test_matmul(self,ndims=[2,3],**ka):
		from operator import matmul
		self._test_op_(matmul,otype='matmul',ndims=ndims,**ka)
	def test_matmul_unlimited(self,**ka):
		return self.test_matmul(unlimited=True,n=ntests//2,**ka)
	def test_svd(self,ndims=[2],sizes=[2,3,5,10,30],n=ntests,**ka):
		import itertools
		for ndim,_ in itertools.product(ndims,range(n)):
			shape = tuple(np.random.choice(sizes,ndim))
			repeat=[x if x else None for x in np.random.rand(ndim)<0.5]
			d1, d0 = self.gen_data(shape,repeat=repeat,**ka) 
			mku, mkl, mkvt = d1.svd() # Perform eigenvalue decomposition on reduced form
			reconstructed_mk = mku@ torch.diag(mkl) @ mkvt
			if isinstance(reconstructed_mk,self.target_cls):
				reconstructed_mk=reconstructed_mk.tensor()
			assert torch.allclose(reconstructed_mk, d0, atol=1e-6), "Reconstructed matrix does not match the original."
	def test_torch_elem1(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test unary pytorch functions
		import itertools
		for xi,dtype in itertools.product(self.target_cls.HANDLED_FUNCTIONS_elem[0],dtypes):
			self._test_op_(xi,otype='none',dtype=dtype,**ka)
	def test_torch_elem2(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test binary pytorch functions
		import itertools
		for xi,dtype,(otype,unlimited) in itertools.product(self.target_cls.HANDLED_FUNCTIONS_elem[1],dtypes,[('atom',False),('elem',False),('elem',True)]):
			self._test_op_(xi,otype=otype,dtype=dtype,unlimited=unlimited,**ka)
	def test_torch_agg(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test pytorch aggregation functions
		import itertools
		for xi,dtype in itertools.product(self.target_cls.HANDLED_FUNCTIONS_agg,dtypes):
			self._test_agg_(xi,dtype=dtype,**ka)
	def test_torch_mean(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test pytorch mean function
		for dtype in dtypes:
			self._test_agg_(torch.mean,dtype=dtype,**ka)
	def test_operator1(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test unary operators
		import itertools
		import operator
		ops=[operator.neg,operator.abs,operator.invert]
		for xi,dtype in itertools.product(ops,dtypes):
			self._test_op_(xi,otype='none',dtype=dtype,**ka)
	def test_operator2(self,dtypes=[dtype_float,dtype_int],**ka):
		#Test binary operators
		import itertools
		import operator
		ops=[operator.add,operator.sub,operator.mul,operator.truediv,operator.floordiv,operator.mod,operator.pow,operator.gt,operator.ge,operator.lt,operator.le,operator.eq,operator.ne]
		for xi,dtype,(otype,unlimited) in itertools.product(ops,dtypes,[('atom',False),('elem',False),('elem',True)]):
			self._test_op_(xi,otype=otype,dtype=dtype,unlimited=unlimited,**ka)

pc={
	#Probability of breaking axis into different components at each element
	'p_break':0.2,
	#Probability of Probability of not repeating for each dimension
	'p_nrepeat':0.3,
	#Probability of breaking repeat at each element inside air
	'p_rbreak':0.3,
	#Probability of next component being tensor
	'p_tensor':0.4,
	#Probability of next component being air
	'p_air':0.3,
	#Probability of next component being composite
	'p_composite':0.3,
}

class Test_composite:
	target_cls=air.composite
	@classmethod
	def gen_data(cls,shape,p=pc,caxis=None,dtype=dtype_float):
		"""
		Generate composite data for testing.
		shape:		Full shape of tensor.
		p:			parameters. See pc.
		caxis:		Axis to break into different components. If None, a random axis will be chosen.
		Return:
		composite:	composite object.
		tensor:		torch.Tensor.
		"""
		#Find axis for composite
		if caxis is None:
			caxis=np.nonzero([x>1 for x in shape])[0]
			if len(caxis)==0:
				raise ValueError('Shape must have at least one element greater than 1')
			caxis=np.random.choice(caxis)
		else:
			assert isinstance(caxis,int)
			assert caxis<len(shape) and caxis>=0
		#Get breaking locations
		t2=torch.nonzero(torch.rand(shape[caxis]-1)<p['p_break'],as_tuple=True)[0]+1
		if len(t2)==0:
			t2=torch.randint(1,shape[caxis],(1,))
		t2=torch.cat([torch.tensor([0],dtype=dtype_int),t2,torch.tensor([shape[caxis]],dtype=dtype_int)])
		#Get type of each component
		t3=np.cumsum([p['p_tensor'],p['p_air'],p['p_composite']])
		assert t3[-1]>0
		t3=t3/t3[-1]
		t3=np.searchsorted(t3,np.random.rand(len(t2)-1))
		#Generate data
		t4=[]
		t5=[]
		for xi in range(len(t2)-1):
			shapenew=tuple(list(shape[:caxis])+[t2[xi+1]-t2[xi]]+list(shape[caxis+1:]))
			if t3[xi]==0 or t2[xi+1]-t2[xi]==1:
				#Generate tensor
				t6=torch.randn(shapenew,dtype=dtype_float,requires_grad=False).to(dtype)
				t4.append(t6)
				t5.append(t6)
			elif t3[xi]==1:
				#Generate air
				t6=Test_air.gen_data(shapenew,dtype=dtype)
				t4.append(t6[0])
				t5.append(t6[1])
			elif t3[xi]==2 and t2[xi+1]-t2[xi]>1:
				#Generate composite
				t6=Test_composite.gen_data(shapenew,p=p,dtype=dtype)
				t4.append(t6[0])
				t5.append(t6[1])
			else:
				assert False
		#Create composite and full tensor
		t4=cls.target_cls(t4,caxis)
		t5=torch.cat(t5,dim=caxis)
		return t4,t5
	def test_tensor(self,ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests):
		import itertools
		for ndim,_ in itertools.product(ndims,range(n)):
			shape=np.random.choice(sizes,ndim)
			if (shape==1).all():
				continue
			d1,d0=self.gen_data(tuple(shape))
			assert (d1.tensor()==d0).all()
	def test_getitem(self,ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests):
		import itertools
		for ndim,nid in itertools.product(ndims,range(n)):
			shape=np.random.choice(sizes,ndim)
			if (shape==1).all():
				continue
			d1,d0=self.gen_data(shape)
			#Generate random key
			key=np.random.rand(ndim)
			key2=[]
			for xi0,xi in enumerate(key):
				if xi<0.2:
					key2.append(slice(None))
				elif xi<0.4:
					key2.append(np.random.randint(-shape[xi0]+1,shape[xi0]))
				elif xi<0.6:
					key2.append(list(np.random.choice(np.arange(-shape[xi0]+1,shape[xi0]),np.random.randint(1,shape[xi0]+1),replace=True)))
				else:
					key2.append(slice(np.random.randint(-shape[xi0]+1,shape[xi0]),np.random.randint(-shape[xi0]+1,shape[xi0])))
			if len(list(filter(lambda x:not isinstance(x,(slice,int)),key2)))>1:
				continue
			key2=tuple(key2)
			ans=[d0[key2],d1[key2]]
			assert ans[0].shape==ans[1].shape
			if not isinstance(ans[1],torch.Tensor):
				ans[1]=ans[1].tensor()
			assert (ans[0]==ans[1]).all()
	def test_sum(self,ndims=[1,2,3],sizes=[1,2,3,5,10,30],unlimited=True,n=ntests):
		import itertools
		if not unlimited:
			raise NotImplementedError
		for ndim,_ in itertools.product(ndims,range(n)):
			shape=np.random.choice(sizes,ndim)
			if (shape==1).all():
				continue
			d1,d0=self.gen_data(tuple(shape))
			#Generate random axes for sum
			a=np.nonzero(np.random.rand(ndim)<0.3)[0]
			if len(a)==0:
				a=None
			elif len(a)==1 and np.random.rand()<0.5:
				a=int(a[0])
			else:
				a=[int(x) for x in a]
			try:
				ans=[d0.sum(axis=a),d1.sum(axis=a)]
			except NotImplementedError:
				continue
			assert not (isinstance(ans[0],float) ^ isinstance(ans[1],float))
			if isinstance(ans[0],float):
				assert ans[0]==pytest.approx(ans[1])
			else:
				assert ans[0].shape==ans[1].shape
				if not isinstance(ans[1],torch.Tensor):
					ans[1]=ans[1].tensor()
				assert torch.allclose(ans[0],ans[1])
	def _test_op_(self,operator,otype='elem',ndims=[1,2,3],sizes=[1,2,3,5,10,30],n=ntests//2,unlimited=True,**ka):
		"""
		Tests operator on air.
		operator:	Operator to test
		otype:		Operator type. Accepts:
			'atom':		Operator with single value
			'elem':		Elementwise operator
			'matmul':	Matrix multiplication
		ndims:	Number of dimensions to test
		probbreak:	Probability of breaking the repeat for each element
		n:		Number of random tests in each case
		"""
		import itertools
		assert otype!='matmul' or 1 not in ndims
		for ndim,_ in itertools.product(ndims,range(n)):
			if otype=='atom':
				shape=tuple(np.random.choice(sizes,ndim))
				if all(x==1 for x in shape):
					continue
				d1,d0=self.gen_data(shape,**ka)
				e0=e1=np.random.randn()
				ans=[operator(d0,e0),operator(d1,e1)]
			elif otype=='elem':
				shape=tuple(np.random.choice(sizes,ndim))
				if all(x==1 for x in shape):
					continue
				d1,d0=self.gen_data(shape,**ka)
				e1,e0=self.gen_data(shape,**ka)
				ans=[]
				for xi,xj in itertools.product([d0,d1],[e0,e1]):
					if unlimited:
						try:
							ans.append(operator(xi,xj))
						except NotImplementedError:
							continue
					else:
						ans.append(operator(xi,xj))
			elif otype=='matmul':
				shapes=[np.random.choice(sizes,ndim) for _ in range(2)]
				for xi in range(len(shapes[1])-2):
					shapes[1][xi]=shapes[0][xi]
				shapes[1][-2]=shapes[0][-1]
				shapes=[tuple(x) for x in shapes]
				if any(all(y==1 for y in x) for x in shapes):
					continue
				d1,d0=self.gen_data(shapes[0],**ka)
				e1,e0=self.gen_data(shapes[1],**ka)
				ans=[]
				for xi,xj in itertools.product([d0,d1],[e0,e1]):
					if unlimited:
						try:
							ans.append(operator(xi,xj))
						except NotImplementedError:
							continue
					else:
						ans.append(operator(xi,xj))
			else:
				assert False
			ans+=[x.reduce() for x in ans if not isinstance(x,torch.Tensor)]
			ans=[x if isinstance(x,torch.Tensor) else x.tensor() for x in ans]
			assert all(x.shape==ans[0].shape for x in ans[1:])
			for xi in range(1,len(ans)):
				assert torch.isfinite(ans[xi]).all()
				assert ans[xi]==pytest.approx(ans[0])
	def test_add(self,**ka):
		from operator import add
		self._test_op_(add,otype='atom',**ka)
		self._test_op_(add,otype='elem',**ka)
	def test_mul(self,**ka):
		from operator import mul
		self._test_op_(mul,otype='atom',**ka)
		self._test_op_(mul,otype='elem',**ka)
	def test_matmul(self,ndims=[2,3],**ka):
		from operator import matmul
		self._test_op_(matmul,otype='matmul',ndims=ndims,**ka)