#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

import airqtl
import numpy as np
import pytest
import torch

from . import test_air as tair

device='cpu'
dtype=torch.float64
torch.manual_seed(12345)
np.random.seed(12345)
torch.use_deterministic_algorithms(True)
torch.set_default_device(device)

class Test_multi:
	@staticmethod
	def gen_data(nx,ny,nc,ncov,maf=None,na=2):
		"""
		Generate data for association testing.
		nx:		Number of predictor variables.
		ny:		Number of target variables.
		nc:		Number of cells or samples.
		ncov:	Number of covariates.
		maf:	Minor allele frequency. If None, generate continuous data for predictor variables.
		na:		Numer of alleles. Only 2 is supported.
		Return:
		dx:		Predictor variable in shape (nx,nc).
		dy:		Target variable in shape (ny,nc).
		dc:		Covariates in shape (ncov,nc).
		ncs:	Number of cells in each sample in shape (n_donor).
		"""
		import numpy as np
		from scipy.stats import rankdata
		if na!=2:
			raise NotImplementedError
		pcnew=dict(tair.pc)
		pcnew['p_composite']=0
		r=[]
		while len(r)<3:
			dx,_,r=tair.Test_air.gen_data((nx,nc),repeat=[None,True],return_repeat=True)
			assert dx.shape==(nx,nc)
			dx=dx.v.numpy()
			assert len(r)==2 and r[0] is None
			if r[1] is not None:
				r=r[1].numpy()
			else:
				r=np.ones(nc,dtype=int)
			assert r.sum()==nc
		if maf is not None:
			#Convert to integer predictor
			assert maf>0 and maf<=0.5
			t0=dx.shape
			t1=np.array([maf**2,2*maf*(1-maf),(1-maf)**2]).cumsum()
			t2=np.array([(rankdata(x)-1)/(dx.shape[1]-1) for x in dx]).T
			t2[np.nonzero(t2==t2.min(axis=0))]=0
			dx=np.array([np.searchsorted(t1,x) for x in t2.T])
			dx[dx>na]=na
			dx=na-dx
			assert dx.shape==(nx,len(r))
			assert (dx>=0).all() and (dx<=na).all()
			assert all(len(set(x))>1 for x in dx)
			if np.random.rand()<0.2:
				t1=np.random.randint(na+1)
				dx2=dx.copy()
				dx2[dx2==t1]=(t1+1)%(na+1)
				if all(len(set(x))>1 for x in dx2):
					dx=dx2
			dx=dx.astype(float)
		dy=np.random.randn(ny,nc)
		dc=np.random.randn(ncov,nc)
		return (dx,dy,dc,r)
	@staticmethod
	def cmp_output(anss,nonan=True,nol0=False):
		"""
		Compare the outputs.
		anss:	List of outputs to check consistency.
		nonan:	Whether to check the absence of NaN. If False, checks the consistency of NaN.
		nol0:	Whether to skip checking the l0 column of the output (second column).
		"""
		import io

		import numpy as np
		import pandas as pd
		d=[]
		for xi in anss:
			with io.StringIO(xi) as f:
				d.append(pd.read_csv(f,sep='\t',header=None,index_col=[0,1]))
		assert all(len(x.index)==len(d[0].index) for x in d[1:])
		assert all(len(x.columns)==len(d[0].columns) for x in d[1:])
		assert all(set(x.index)==set(d[0].index) for x in d[1:])
		assert all((x.columns==d[0].columns).all() for x in d[1:])
		cols=d[0].columns
		if nol0:
			cols=[y for x,y in enumerate(cols) if x!=1]
		for xi in cols:
			v=[x.loc[d[0].index][xi].values for x in d]
			assert all(x.shape==v[0].shape for x in v[1:])
			v=np.array(v)
			t1=np.isnan(v)
			if nonan:
				assert not t1.any()
			else:
				assert (t1[1:]==t1[0]).all()
			v=v[:,~t1[0]]
			assert all(x==pytest.approx(v[0]) for x in v[1:])
	def _test1_consistency(self,nx,ny,nc,ncov,maf=None,l0var=None,bslist=[[1,5],[3,99999]]):
		"""
		Individual test case for consistency of multi.
		nx:		Number of predictor variables.
		ny:		Number of target variables.
		nc:		Number of cells or samples.
		ncov:	Number of covariates.
		maf:	Minor allele frequency. If None, generate continuous data for predictor variables.
		l0:		Relative strength of random effect in linear mixed model. If None, uses linear model.
		bslist:	List of batch sizes for predictor and target variables.
		"""
		import io
		from functools import partial

		import numpy as np
		dx,dy,dc,ncs=self.gen_data(nx,ny,nc,ncov,maf=maf)
		if ncov>0:
			dc=np.concatenate([dc,np.ones_like(dc[[0]])],axis=0)
		else:
			dc=np.ones((1,nc))
		
		if l0var is None:
			l0,mkl,mku=None,None,None
		else:
			assert l0var>0 and l0var<100
			l0=np.abs(np.random.randn(ny)*np.sqrt(l0var))
			mkl,mku=airqtl.kinship.eigen(np.eye(len(ncs)),ncs)
		names=[[str(x) for x in range(nx)],[str(x) for x in range(ny)]]
		fmt=partial(airqtl.association.fmt1,None,names,cis=None)
		ans=[]
		for xi in bslist:
			with io.StringIO() as f:
				# Approach 1 with constant predictors and covariates
				airqtl.association.multi(dx,dy,None,dc,ncs,mkl,mku,l0,None,lambda x,c:x.reshape(x.shape[0],1,x.shape[1]),1,f,fmt,device=device,bsx=xi[0],bsy=xi[1])
				ans.append(f.getvalue())
			with io.StringIO() as f:
				# Approach 1 with functional predictors and covariates
				airqtl.association.multi(dx,dy,dc[:-1],dc[[-1]],ncs,mkl,mku,l0,lambda x,c:c.reshape(1,c.shape[0],c.shape[1]).expand(x.shape[0],c.shape[0],c.shape[1]),lambda x,c:x.reshape(x.shape[0],1,x.shape[1]),1,f,fmt,device=device,bsx=xi[0],bsy=xi[1])
				ans.append(f.getvalue())
		self.cmp_output(ans,nol0=l0 is None)
	def test_1(self,n=500):
		import itertools

		import numpy as np
		for nid in range(n):
			print(nid)
			nx=np.random.randint(5,10)
			ny=np.random.randint(3,15)
			nc=np.random.randint(30,40)
			ncov=np.random.randint(5)
			maf=np.random.rand()
			if maf>0.5:
				maf=None
			l0var=np.random.rand()*5
			if l0var>4:
				l0var=None
			bslist=[1,2,3,5,7,11,99999]
			bslist=list(itertools.product(np.random.choice(bslist,2),np.random.choice(bslist,2)))
			print(nx,ny,nc,ncov,maf,l0var,bslist)
			self._test1_consistency(nx,ny,nc,ncov,l0var=l0var,maf=maf,bslist=bslist)
