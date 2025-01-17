#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

from typing import Optional, Tuple, Union

from .utils.importing import torch


def multi(dx,dy,dc0,dc1,ncs,mkl,mku,l0,f0,f1,nxd,out,fmt,bsx:int=128,bsy:int=32768,dimreduce:float=0,dtype_f=torch.float64,dtype_i=torch.int32,device:str='cuda:0',h0:str='chi2',subset:Optional[list[set]]=None):
	"""Fast association testing between trait and single-variate with linear models or linear mixed models in single-cell cohort setting.
	Outputs to a single file object directly with each line being a statistical test result. This function does not output the header row.

	Inputs:
	dx:		numpy.ndarray(shape=(n_x,n_donor)). Genotype matrix. Overloadable.
	dy:		numpy.ndarray(shape=(n_y,n_cell)). Target variables. Overloadable.
	dc0:	numpy.ndarray(shape=(n_cov0,n_cell)). Intermediate covariates used to compute null/alternative fixed effects as input of f0 and f1 below. If no dc0 is needed, set to None.
	dc1:	numpy.ndarray(shape=(n_cov1,n_cell)). Standalone constant covariates used as fixed effects independent of f0 and f1. Should not be empty because intercept term is always needed.
	ncs:	numpy.ndarray(shape=(n_donor,),dtype='uint'). Number of cells from each donor.
			dy and dc should be ordered to contain ns[i] samples for i=0 to n_donor-1 sequentially.
	mkl:	np.array(shape=(n_k,)). Eigenvalues of K.
	mku:	np.array(shape=(n_k,n_donor)). Reduced eigenvectors of K.
			Full eigenvector x is np.concat([np.repeat(mku[x][y],ns[y]) for y in range(nd)])
	l0:		np.array(shape=(n_y,)). Gene null heritability.
	f0,
	f1:		Function to compute null/alternative fixed effects respectively. f0 should depend on x or set to None. If f0 is not None and is independent of x, putting f0 into c can speed up computation. f1 can be dependent or independent of x, and should not be None. See sections below.
	nxd:	n_xd below. Number of dimensions of f1 return.
	out:	file object for output in tsv format.
	fmt:	Formatter and filter function that take parameters (id_x,id_y,s0,l0,b,s,r,p) and returns list of strings without end-of-line as the output. Parameter dimensions are (n_x) for id_x; (n_y) for id_y, l0; (n_x,n_y) for s0, s, r, p; (n_x,n_y,nxd) for b.
	bsx,
	bsy:	Batch size for X or Y
	dimreduce:	np.array(shape=(n_y,),dtype='uint') or uint. If Y doesn't have full rank in the first place, this parameter allows to specify the loss for accurate p-value computation.
	dtype_f,
	dtype_i:Data type for float and integer in pytorch
	device:	Device for pytorch
	h0:		Null distribution used to test hypothesis and compute p-value. Accepts:
		chi2:	Chi-square distribution. Fast fro pytorch but less accurate
		beta:	Beta distribution. Slow from scipy but more accurate
	subset:	Subset of SNP-gene pairs to test as a list of sets whose element is the indices of the SNPs to test for each gene. If None, test all SNP-gene pairs.
	
	Statistical model for each frequentist test (SNP-gene pair):
	Alternative:			y = a*c1 + b*f1(x,c0) + d*f0(x,c0) + e, 
	Null:					y = a0*c1 + d0*f0(x,c0) + e0
							i.e. b=0
	For linear mixed model:	e ~ MVN(0, s**2 * (I+l0*K)), e0 ~ MVN(0, s0**2 * (I+l0*K))
	For linear model:		e ~ iid N(0, s**2), e0 ~ iid N(0, s0**2)
	Model fitting & test:	restricted maximum likelihood
	Default model is linear mixed model. To use linear model, set mkl, mku, and l0 to None.

	Output matrix definitions and shapes:
	b:		numpy.ndarray(shape=(n_xd,n_x,n_y)). eQTL b, effect size.
	p:		numpy.ndarray(shape=(n_x,n_y)). eQTL p-value.
	r:		numpy.ndarray(shape=(n_x,n_y)). Pearson R between genotype and gene expression after conditional on covariates and transformation into iid unexplained variance. Signed if n_xd==1. Otherwise simply is the unsigned square root of proportion of variance explained.
	s:		numpy.ndarray(shape=(n_x,n_y)). gene alternative scale.
	s0: 	numpy.ndarray(shape=(n_x,n_y)). gene null scale. All rows are identical if f0 is None.
	
	Inputs of f0 or f1:
	x:		air(shape=(n_x,n_cell)). Genotype matrix.
	c0:		numpy.ndarray(shape=(n_cov0,n_cell)). Intermediate covariates.

	Return of f0 or f1:
	dxd:	air(shape=(n_x,n_cov0 or n_xd,n_cell)): Dynamic variables for dx. n_xd is the dimension that matches dimxd. 

	Examples of f0 & f1:
	1. f0(x,c0)=None, f1(x,c0)=x to test linear effect of x.
	2. f0(x,c0)=x, f1(x,c0)=x>0 to test dominant allele effect of x.
	3. f0(x,c0)=[x,x*c0], f1(x,c0)=x**2 to test quadratic effects of x.
	4. f0(x,c0)=x, f1(x,c0)=x*c0[0] to test effect of x*c0[0].

	Definitions and shapes for each test:
		y:		gene expression of shape (1, n_cell)
		x:		genotype (=0,1,2) of shape (1, n_cell)
		c:		covariates of shape (n_cov, n_cell)
		a0: 	covariate null contribution of shape (n_cov,1)
		l0: 	gene null heritability (scaled)
		s0: 	gene null scale
		a: 		covariate alternative contribution of shape (1, n_cov)
		b:		eQTL beta of shape (n_xd)
		s:		gene alternative scale
		K:		kinship matrix of shape (n_cell, n_cell), derived from mkl and mku
		mkl:	eigenvalues of K of shape (n_k,), n_k <= n_donor
		mku:	reduced eigenvectors of K of shape (n_k,n_donor)
	Statistic for frequentist test (all equivalent): |b|, Pearson R^2, likelihood ratio

	Optimization over existing methods:
		* K is a low rank matrix, allowing for low-rank optimizations
		* Many variables are only donor dependent but constant across cells from the same donor. This allows for acceleration with reduced form for most variables.
		* GPU acceleration
		* Optional out-of-core computation. See section "Overloading".

	Overloading:
		You may overload inputs dx and/or dy for out-of-core computation. See airqtl.io for existing overloading classes.
		The following methods are needed for dx and dy to follow the same interface with numpy.ndarray:
		* shape:	Shape of the matrix
		* __getitem__(self, index)-> numpy.ndarray:
			Accessing a subset of the matrix. This function is designed to get item slices sequentially along the first dimension, with one round for dy and possibly multiple rounds for dx. Your overloaded function should be optimized towards this.
			
	Note:
		This function only does not return other coefficient such as a0, d0, a, d. To obtain their values, you can put c1 and f0 into f1 but note that the obtained p-values are changed. Also note that l0 is estimated using c1 but not f0, so moving c1 into f1 can change l0 estimation.

	Return:	None
	"""
	import logging
	from functools import reduce
	from operator import or_
	from os import linesep

	import numpy as np
	from scipy.stats import beta

	from .air import air
	from .op import (mmatmul, mmatmul1, mmatmul2, mmmatmul, mmsquare, msquare,msquare1)
	if mkl is None:
		if mku is not None or l0 is not None:
			raise TypeError('Set mkl, mku, l0 all to None for linear model or all to not None for linear mixed model.')
		#Use linear model
		lm=True
	else:
		if mku is None or l0 is None:
			raise TypeError('Set mkl, mku, l0 all to None for linear model or all to not None for linear mixed model.')
		#Use linear mixed model
		lm=False
		
	# Dimensions
	nd = len(ncs)
	nx0 = dx.shape[0]
	ny0 = dy.shape[0]
	ns = ncs.sum()
	if dc0 is None:
		dc0=np.zeros([0,ns],dtype=float)
	nc1 = dc1.shape[0]
	# Validity checks
	if nd == 0:
		raise ValueError('Empty sample count.')
	if (ncs <= 0).any():
		raise ValueError('nc must be all positive.')
	if bsy <= 0:
		raise ValueError('bsy must be positive.')
	if len(dy.shape) != 2:
		raise ValueError('dy must have 2 dimensions.')
	if dc0.ndim != 2:
		raise ValueError('dc0 must have 2 dimensions.')
	if dc1.ndim != 2:
		raise ValueError('dc1 must have 2 dimensions.')
	if nx0 <= 0:
		raise ValueError('dx must be non-empty.')
	if ny0 <= 0:
		raise ValueError('dy must be non-empty.')
	if nc1 <= 0:
		raise ValueError('dc1 must be non-empty, such as including the intercept term.')
	if dx.shape[1]!=nd:
		raise ValueError(f'dx must have {nd} donors.')
	if dy.shape[1]!=ns:
		raise ValueError(f'dy must have {ns} samples.')
	if dc0.shape[1]!=ns:
		raise ValueError(f'dc0 must have {ns} samples.')
	if dc1.shape[1]!=ns:
		raise ValueError(f'dc1 must have {ns} samples.')
	if np.isscalar(dimreduce):
		dimreduce = np.repeat(int(dimreduce), ny0)
	if dimreduce.shape != (ny0,):
		raise ValueError('Incorrect shape for dimreduce.')
	if nxd == 0:
		raise ValueError('nxd must be positive.')
	if h0=='chi2':
		dimreduce = torch.tensor(dimreduce,device=device,requires_grad=False)
	if not lm:
		nk = mkl.shape[0]
		if (mkl <= 0).any():
			raise ValueError('mkl (eigenvalues) must all be positive.')
		if mku.shape != (nk, nd):
			raise ValueError('Incorrect shape for mku.')
	if h0 not in {'chi2','beta'}:
		raise ValueError('h0 must be either chi2 or beta.')
	if nxd!=1:
		raise NotImplementedError('nxd!=1 is not implemented.')
	if subset is not None:
		assert len(subset)==ny0
		assert all(len(x)==0 or min(x)>=0 for x in subset)
		assert all(len(x)==0 or max(x)<nx0 for x in subset)
		counts=[0,0]
	ans_b=[[] for _ in range(nxd)]
	ans_p=[]
	ans_r=[]
	ans_s=[]
	ans_s0=[]

	ncs=torch.tensor(ncs,dtype=dtype_i,device=device,requires_grad=False)
	dx0=dx
	dy0=dy
	dc0=torch.tensor(dc0,dtype=dtype_f,device=device,requires_grad=False)
	dc1=torch.tensor(dc1,dtype=dtype_f,device=device,requires_grad=False)
	#[n_cov1,n_cov1]
	pcc=dc1@dc1.T
	if not lm:
		du=air(torch.tensor(mku,dtype=dtype_f,device=device,requires_grad=False),repeat=[None,ncs])
		ds=torch.tensor(mkl,dtype=dtype_f,device=device,requires_grad=False)
		dl0=torch.tensor(l0,dtype=dtype_f,device=device,requires_grad=False)
		tc=dc1@du.T
	if f0 is None:
		#[1,1,n_cov1,n_cov1]
		pi=torch.linalg.inv(pcc).view(1,1,*pcc.shape)
	if subset is None:
		xid=np.arange(nx0)
		dx1=dx0
	for vy in range(0,ny0,bsy):
		for xi in range(nxd):
			ans_b[xi].append([])
		ans_p.append([])
		ans_r.append([])
		ans_s.append([])
		ans_s0.append([])
		dy=torch.tensor(dy0[vy:vy+bsy],dtype=dtype_f,device=device,requires_grad=False)
		if not lm:
			dl=dl0[vy:vy+bsy]
		ny=dy.shape[0]
		#Matrix products, original and transformed
		#Original
		#[n_y]
		p0yy=(dy**2).sum(axis=1)
		assert p0yy.shape==(ny,)
		#[n_y,n_cov1]
		pyc=dy@dc1.T
		assert pyc.shape==(ny,nc1)
		#Transformed/Modified
		if lm:
			#[n_y]
			mp0yy=p0yy
			assert mp0yy.shape==(ny,)
			#[1,n_cov1,n_cov1]
			mpcc=pcc.view(1,*pcc.shape)
			assert mpcc.shape==(1,nc1,nc1)
			#[n_y,n_cov1]
			mpyc0=pyc
			assert mpyc0.shape==(ny,nc1)
			if f0 is None:
				mpi=pi
				ncov=mpi.shape[-1]
		else:
			ty=dy@du.T
			ts=1-(1+ds*dl.view(-1,1).expand(-1,ds.shape[0]))**(-1)
			#[n_y]
			mp0yy=msquare1(ty,p0yy,ts)
			assert mp0yy.shape==(ny,)
			#[n_y,n_cov1,n_cov1]
			mpcc=mmatmul(tc,tc,pcc,ts)
			assert mpcc.shape==(ny,nc1,nc1)
			#TODO: Allow reduced rank covariates
			#[n_y,n_cov1]
			mpyc0=mmatmul1(ty,tc,pyc,ts)
			assert mpyc0.shape==(ny,nc1)
			if f0 is None:
				#[n_y,1,n_cov1,n_cov1] (2 lines)
				mpi=torch.linalg.inv(mpcc)
				mpi=mpi.reshape(mpi.shape[0],1,*mpi.shape[1:])
				assert mpi.shape==(ny,1,nc1,nc1)
				ncov=mpi.shape[-1]
		#Double modified
		if f0 is None:
			#[n_y,1]
			mmp0yy=mmsquare(mp0yy.reshape(*mp0yy.shape,1,1,1),mpyc0.reshape(*mpyc0.shape[:-1],1,1,mpyc0.shape[-1]),mpi)[:,:,0,0]
			assert mmp0yy.shape==(ny,1)
		if subset is not None:
			t1=sum(len(x) for x in subset[vy:vy+ny])
			if t1==0:
				del dy,p0yy,pyc,mp0yy,mpcc,mpyc,mmp0yy
				if not lm:
					del ty,ts
				continue
			xid=np.array(sorted(list(reduce(or_,subset[vy:vy+ny]))))
			dx1=dx0[xid]
			counts[0]+=t1
			counts[1]+=dx1.shape[0]*ny
			t1=np.concatenate([np.repeat(x,len(subset[x+vy])) for x in range(ny)])
			t1=[np.concatenate([np.array(list(x),dtype=int) for x in subset[vy:vy+ny]]),t1]
			subsetmat=np.zeros([nx0,ny],dtype=bool)
			subsetmat[t1[0],t1[1]]=True
			subsetmat=subsetmat[xid]
			logging.debug('Intermediate subset mode efficiency report. Target pairs: {}. Computed pairs: {}. Efficiency: {}'.format(*counts,counts[0]/counts[1]))
		for vx in range(0,dx1.shape[0],bsx):
			dx=air(torch.tensor(dx1[vx:vx+bsx],dtype=dtype_f,device=device,requires_grad=False),repeat=[None,ncs]).reduce()
			nx=dx.shape[0]
			if f0 is not None:
				#[n_x,n_f0,n_cell]
				df0=f0(dx,dc0)
				assert df0.ndim==3
				nf0=df0.shape[1]
				#[n_x,n_f0,n_f0]
				p0f0f0=(df0@df0.mT)
				assert p0f0f0.shape==(nx,nf0,nf0)
				#[n_x,n_f0,n_cov1]
				pf0c=df0@dc1.T
				assert pf0c.shape==(nx,nf0,nc1)
				#[n_y,n_x,n_f0]
				pyf0=(dy.view(1,*dy.shape).expand(df0.shape[0],*dy.shape)@df0.mT).swapaxes(0,1)
				assert pyf0.shape==(ny,nx,nf0)
				if lm:
					#[1,n_x,n_f0,n_f0]
					mp0f0f0=p0f0f0.view(1,*p0f0f0.shape)
					assert mp0f0f0.shape==(1,nx,nf0,nf0)
					#[1,n_x,n_f0,n_cov1]
					mpf0c=pf0c.view(1,*pf0c.shape)
					assert mpf0c.shape==(1,nx,nf0,nc1)
				else:
					#[n_x,n_f0,n_k]
					tf0=df0@du.T
					assert tf0.shape==(nx,nf0,nk)
					#[n_y,n_x,n_f0,n_f0]
					mp0f0f0=mmatmul2(tf0.view(1,*tf0.shape),tf0.view(1,*tf0.shape),p0f0f0.view(1,*p0f0f0.shape),ts.view(ts.shape[0],1,1,ts.shape[1]))
					assert mp0f0f0.shape==(ny,nx,nf0,nf0)
					#[n_y,n_x,n_f0,n_cov1]
					mpf0c=mmatmul2(tf0.view(1,*tf0.shape),tc.view(1,1,*tc.shape),pf0c.view(1,*pf0c.shape),ts.view(ts.shape[0],1,1,ts.shape[1]))
					assert mpf0c.shape==(ny,nx,nf0,nc1)
				#[n_y or 1,n_x,n_f0+n_cov1,n_f0+n_cov1]
				mpi=torch.linalg.inv(torch.cat([torch.cat([mp0f0f0,mpf0c.swapaxes(-1,-2)],axis=-2),torch.cat([mpf0c,mpcc.view(mpcc.shape[0],1,*mpcc.shape[1:]).expand(mpcc.shape[0],nx,*mpcc.shape[1:])],axis=-2)],axis=-1))
				assert mpi.shape==(ny,nx,nf0+nc1,nf0+nc1) or mpi.shape==(1,nx,nf0+nc1,nf0+nc1)
				ncov=mpi.shape[-1]
			df1=f1(dx,dc0)
			assert df1.ndim==3 and df1.shape[1]==nxd
			#Matrix products, original and transformed
			#Original
			#[n_x,n_f1,n_f1]
			p0xx=df1@df1.mT
			assert p0xx.shape==(nx,nxd,nxd)
			#[n_y,n_x,n_f1]
			pyx=(dy@df1.mT).swapaxes(0,1)
			assert pyx.shape==(ny,nx,nxd)
			if f0 is not None:
				#[n_x,n_f1,n_f0+n_cov1]
				pxc=torch.cat([df1@df0.mT,df1@dc1.T],axis=2)
				assert pxc.shape==(nx,nxd,nf0+nc1)
			else:
				#[n_x,n_f1,n_cov1]
				pxc=df1@dc1.T
				assert pxc.shape==(nx,nxd,nc1)
			#Transformed/Modified
			if lm:
				#[1,n_x,n_f1,n_f1]
				mp0xx=p0xx.view(1,*p0xx.shape)
				assert mp0xx.shape==(1,nx,nxd,nxd)
				mpyx=pyx
				if f0 is not None:
					#[1,n_x,n_f1,n_f0+n_cov1]
					mpxc=pxc.view(1,*pxc.shape)
					assert mpxc.shape==(1,nx,nxd,nf0+nc1)
					#[n_y,n_x,n_f0+n_cov1]
					mpyc=torch.cat([pyf0,mpyc0.reshape(mpyc0.shape[0],1,mpyc0.shape[1]).expand(mpyc0.shape[0],pyf0.shape[1],mpyc0.shape[1])],axis=-1)
					assert mpyc.shape==(ny,nx,nf0+nc1)
				else:
					#[1,n_x,n_f1,n_cov1]
					mpxc=pxc.view(1,*pxc.shape)
					assert mpxc.shape==(1,nx,nxd,nc1)
					mpyc=mpyc0.view(mpyc0.shape[0],1,mpyc0.shape[1])
			else:
				tx=df1@du.T
				mp0xx=msquare(tx,p0xx,ts)
				mpyx=mmatmul1(ty,tx,pyx,ts)
				if f0 is not None:
					#[1,n_x,n_f0+n_cov1,n_k]
					tca=torch.cat([tf0,tc.view(1,*tc.shape).expand(tf0.shape[0],-1,-1)],axis=1).view(1,tf0.shape[0],tf0.shape[1]+tc.shape[0],tf0.shape[2])
					#[n_y,n_x,n_f1,n_f0+n_cov1]
					mpxc=mmatmul2(tx.view(1,*tx.shape),tca,pxc.view(1,*pxc.shape),ts.view(ts.shape[0],1,1,ts.shape[1]))
					assert mpxc.shape==(ny,nx,nxd,nf0+nc1)
					#[n_y,n_x,n_f0+n_cov1]
					mpyc=torch.cat([mmatmul2(ty.view(ty.shape[0],1,1,ty.shape[1]),tf0.view(1,*tf0.shape),pyf0.view(*pyf0.shape[:-1],1,pyf0.shape[-1]),ts.view(ts.shape[0],1,1,ts.shape[1]))[:,:,0],mpyc0.view(mpyc0.shape[0],1,mpyc0.shape[1]).expand(-1,nx,-1)],axis=2)
					assert mpyc.shape==(ny,nx,nf0+nc1)
				else:
					mpxc=mmatmul(tx,tc,pxc,ts)
					mpyc=mpyc0.reshape(mpyc0.shape[0],1,1,mpyc0.shape[1])
			#Double modified
			#[1 or n_y,n_x,n_f1,n_f1]
			mmp0xx=mmsquare(mp0xx,mpxc,mpi.reshape(mpi.shape[0],1,mpi.shape[1],mpi.shape[2]) if mpi.ndim<4 else mpi)
			assert mmp0xx.shape==(ny,nx,nxd,nxd) or mmp0xx.shape==(1,nx,nxd,nxd)
			#[n_y,n_x,n_f1]
			mmpyx=mmmatmul(mpyx,mpyc,mpxc,mpi)
			assert mmpyx.shape==(ny,nx,nxd)
			if f0 is not None:
				#[n_y,n_x]
				mmp0yy=mmsquare(mp0yy.reshape(*mp0yy.shape,1,1,1),mpyc.reshape(*mpyc.shape[:-1],1,mpyc.shape[-1]),mpi)[:,:,0,0]
				assert mmp0yy.shape==(ny,nx)

			#s0
			#[n_y,n_x] (3 lines)
			ans1_s0=torch.sqrt(mmp0yy/ns)
			if f0 is None:
				ans1_s0=ans1_s0.expand(-1,mmp0xx.shape[1])
			assert ans1_s0.shape==(ny,nx)
			#b
			#[n_y,n_x,n_f1]
			ans1_b=mmpyx/torch.clamp(torch.diagonal(mmp0xx,dim1=-2,dim2=-1),min=1E-30)
			assert ans1_b.shape==(ny,nx,nxd)
			#Conditional Pearson R (proportion of explained variance)
			#Simpler version only working for nxd==1
			#[n_y,n_x]
			ans1_r=mmpyx[:,:,0]/torch.clamp(torch.sqrt(mmp0xx[:,:,0,0]*ns)*ans1_s0,min=1E-30)
			assert ans1_r.shape==(ny,nx)
			#Reset outputs for single-valued f1
			#[n_y,n_x,n_f1]
			t1=torch.diagonal(mmp0xx,dim1=-2,dim2=-1)>0
			t2=t1.any(axis=-1).to(df1.dtype)
			t1=t1.to(df1.dtype)
			ans1_b*=t1
			ans1_r*=t2
			#s
			#[n_y,n_x]
			ans1_s=ans1_s0*torch.sqrt(1-ans1_r**2)
			assert ans1_s.shape==(ny,nx)
			assert ans1_r.isfinite().all()
			assert ans1_s0.isfinite().all()
			assert ans1_s.isfinite().all()
			assert ans1_b.isfinite().all()
			assert all(x.isfinite().all() for x in [ans1_r,ans1_s,ans1_b])
			assert (ans1_r>=-2).all() and (ans1_r<=2).all()
			ans1_r=ans1_r.clamp(-1,1)
			assert (ans1_s>=0).all()
			ans1_s,ans1_s0,ans1_b=[x.cpu().numpy() for x in [ans1_s,ans1_s0,ans1_b]]
			if h0=='beta':
				ans1_r=ans1_r.cpu().numpy()
				ans1_p=beta.cdf(1 - ans1_r.T**2, (ns - 1 - ncov - dimreduce[vy:vy+ny]) / 2, 0.5).T
			elif h0=='chi2':
				ans1_p=torch.special.gammaincc(torch.tensor(nxd/2,device=device,requires_grad=False),(ans1_r.T**2)*((ns-ncov-dimreduce[vy:vy+ny])/2)).T.cpu().numpy()
				ans1_r=ans1_r.cpu().numpy()
			else:
				assert False
			assert (ans1_p>=0).all() and (ans1_p<=1).all()
			#Output
			assert ans1_s0.shape==(ny,nx)
			assert ans1_b.shape==(ny,nx,nxd)
			assert all(x.shape==(ny,nx) for x in [ans1_s,ans1_r,ans1_p])
			ka={} if subset is None else {'subset':subsetmat[vx:vx+nx]}
			out1=fmt(xid[vx:vx+nx],np.arange(vy,vy+ny),ans1_s0,l0[vy:vy+ny] if l0 is not None else None,ans1_b,ans1_s,ans1_r,ans1_p,**ka)
			if len(out1)>0:
				out.write(linesep.join(out1)+linesep)
			del dx,p0xx,pxc,pyx,mp0xx,mpxc,mpyx,mmp0xx,mmpyx,ans1_r,ans1_s,ans1_b,ans1_s0,t1,t2
			if not lm:
				del tx
		del dy,p0yy,pyc,mp0yy,mpcc,mpyc,mmp0yy
		if not lm:
			del ty,ts
	if subset is not None:
		logging.info('Final subset mode efficiency report. Target pairs: {}. Computed pairs: {}. Efficiency: {}'.format(*counts,counts[0]/counts[1]))

def fmt1(locs:Tuple[list[str],list[str]],names:Tuple[list,list],id_x,id_y,s0,l0,b,s,r,p,pcut=2,cis:Union[None,Tuple[int,int],int]=(-1000000,1000000),float_fmt:str="%.8G",sep:str='\t',subset=None)->list[str]:
	"""
	Filtering and formatting function for multi based on P-values and cis relation between SNP and gene
	names:		Name of SNPs and genes as list of np.array
	locs:		Locations of SNPs and genes as [np.array(shape=(n_snp,2)) for chr and location,np.array(shape=(n_gene,3)) for start and stop]
	pcut:		Cutoff for p-value. Trans-eQTL entries with p-values above pcut will be ignored. Possible values:
		* 0:		No output
		* 0 to 1:	Only p-values below this will be output
		* >1:		No filtering with p-values
	cis:		Distance bound from SNP to TSS of gene on the gene's strand to be considered cis. Cis relations are output regardless of p-value filters Possible values:
		* None:			No cis exception. All eQTL entries are subject to pcut filtering.
		* int:			Distance bound for cis relations.
		* [int,int]:	Use different bounds for upstream (first) and downstream (second) distances 
	float_fmt:	Format string for float numbers.
	subset:		Subset of SNP-gene pairs to output as numpy.ndarray(dtype=bool,shape=(len(id_x),len(id_y))). If None, output all pairs.
	Return:		List of string where each is an output eQTL mapping entry.
	"""
	import numpy as np

	from .utils.eqtl import find_cis
	if pcut<0:
		raise ValueError('pcut must be non-negative.')
	#Filter cis
	assert all(len(x)>0 for x in [id_x,id_y])
	ans=np.zeros((len(id_x),len(id_y)),dtype=bool)
	if cis is not None:
		t1=find_cis([locs[0][id_x],locs[1][id_y]],cis)
		ans[t1[0],t1[1]]=True
	#Filter p
	if pcut>0:
		ans[p.T<pcut]=True
	if subset is not None:
		assert subset.shape==ans.shape
		ans&=subset
	#Format output
	ans=np.array(np.nonzero(ans)).T
	ans=[sep.join([names[0][id_x[x]],names[1][id_y[y]]]+[float_fmt % z if z is not None else '' for z in [s0[y,x],l0[y] if l0 is not None else None]+list(b[y,x])+[s[y,x],r[y,x],p[y,x]]]) for x,y in ans]
	return ans

def fmt1_header(names_b,sep:str='\t')->str:
	"""
	Header for fmt1
	"""
	assert len(names_b)==len(set(names_b))
	return sep.join(['SNP','Gene','s0','l0']+['b_%s' % x if x is not None else 'b' for x in names_b]+['s','r','p'])

def multi_linear(dx,dy,dc1,ncs,mkl,mku,l0,out,fmt,**ka):
	"""
	Tests linear effect of genotype on gene expression.
	See multi for the framework.
	"""
	return multi(dx,dy,None,dc1,ncs,mkl,mku,l0,None,lambda x,c0:x.reshape(x.shape[0],1,x.shape[1]),1,out,fmt,**ka)

def multi_dominant(dx,dy,dc1,ncs,mkl,mku,l0,out,fmt,**ka):
	"""
	Tests dominant effect of genotype on gene expression.
	See multi for the framework.
	"""
	return multi(dx,dy,None,dc1,ncs,mkl,mku,l0,lambda x,c0:x.reshape(x.shape[0],1,x.shape[1]),lambda x,c0:(x.reshape(x.shape[0],1,x.shape[1])>0).to(x.dtype),1,out,fmt,**ka)

def multi_gxc_f0(nxd,dx,dc0,dom:bool=False):
	from airqtl.air import composite
	t1=[dx.reshape(dx.shape[0],1,dx.shape[1])]
	if dom:
		t1.append((t1[0]>0).to(dtype=t1[0].dtype))
	t1.append(dx.reshape(dx.shape[0],1,dx.shape[1])*dc0[nxd:].reshape(1,dc0.shape[0]-nxd,dc0.shape[1]))
	return composite(t1,1)

def multi_gxc_f1(nxd,dx,dc0):
	return dx.reshape(dx.shape[0],1,dx.shape[1])*dc0[:nxd].reshape(1,nxd,dc0.shape[1])

def multi_gxc(dx,dy,dc0,dc1,ncs,mkl,mku,l0,nxd,out,fmt,dom:bool=False,**ka):
	"""
	Tests genotype * context (covariate) effect on gene expression.
	Treats the first nxd covariates as interactions into f1 and the rest as interactions into f0.
	nxd:	The initial number of covariates to be treated as interactions into f1.
	dom:	Whether to include dominant genotype as an extra covariate
	"""
	from functools import partial
	if nxd>dc0.shape[0] or nxd<=0:
		raise ValueError('nxd must be greater than zero and less than or equal to the number of intermediate covariates (dc0.shape[0]).')
	return multi(dx,dy,dc0,dc1,ncs,mkl,mku,l0,partial(multi_gxc_f0,nxd,dom=dom),partial(multi_gxc_f1,nxd),nxd,out,fmt,**ka)


assert __name__ != "__main__"
