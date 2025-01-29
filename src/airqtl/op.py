#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

from .utils.importing import torch

# Variable naming convention:
# dx: original variable
# tx: transformed variable. tx=dx@du.T
# pxy: variable matmul products. pxy=dx@dy.T
# p0xx: variable self-products for same row/column. p0xx=diag(pxx)
# mpxy: modified variable matmul products after transformation into iid unexplained variance. mpxy=tx@(I+ds*dl0)**(-1)@ty.T
# mp0xx: modified variable self-products. mp0xx=diag(mpxx)
# mmpxy: double modified variable matmul products after further removing covariates. mmpxy=mpxy-mpxc@(mpcc)**(-1)@mpyc.T
# mmp0xx: double modified variable self-products. mmp0xx=diag(mmpxx)

def mmatmul(ta,tb,pab,ts):
	"""
	Modified matmul for transformed data.
	"""
	return pab.expand(ts.shape[0],*([-1]*pab.ndim))-((ta.reshape(-1,ta.shape[-1]).expand(ts.shape[0],-1,-1)*ts.view(ts.shape[0],1,ts.shape[1]))@tb.reshape(-1,tb.shape[-1]).expand(ts.shape[0],-1,-1).swapaxes(1,2)).reshape(ts.shape[0],*pab.shape)

def mmatmul1(ta,tb,pab,ts):
	"""
	Modified matmul for transformed data.
	ta and l are always follow the same axis[0].
	"""
	return pab-((ta.reshape(-1,ta.shape[-1])*ts)@tb.reshape(-1,tb.shape[-1]).T).reshape(pab.shape)
	
def mmatmul2(ta,tb,pab,ts):
	"""
	Modified matmul for transformed data.
	"""
	return pab-(ta*ts@tb.swapaxes(-1,-2))

def msquare(ta,p0aa,ts):
	"""
	Modified square for transformed data.
	"""
	ta=ta.reshape(-1,ta.shape[-1])
	t1=torch.sqrt(ts)
	t1=t1.view(t1.shape[0],1,t1.shape[1]).expand(-1,ta.shape[0],-1)
	t1=ta.expand(ts.shape[0],-1,-1)*t1
	return torch.clamp(p0aa.expand(ts.shape[0],*p0aa.shape)-torch.linalg.norm(t1,axis=-1,ord=2).reshape(ts.shape[0],*p0aa.shape)**2,min=0)

def msquare1(ta,p0aa,ts):
	"""
	Modified square for transformed data.
	ta and l are always follow the same axis[0].
	"""
	t1=ta*torch.sqrt(ts)
	return torch.clamp(p0aa-torch.linalg.norm(t1,axis=-1,ord=2)**2,min=0)

def mmmatmul(mpab,mpac,mpbc,mpcci):
	"""
	Double modified matmul for transformed data after covariate removal.
	"""
	mpac,mpbc=[x.reshape(x.shape[0],*([] if x.ndim==2 else [-1]),x.shape[-1]) for x in [mpac,mpbc]]
	reduce=[x.ndim<mpcci.ndim for x in [mpac,mpbc]]
	if reduce[0]:
		mpac=mpac.view(*mpac.shape[:-1],1,mpac.shape[-1])
	if reduce[1]:
		mpbc=mpbc.view(*mpbc.shape[:-1],1,mpbc.shape[-1])
	t1=mpac@mpcci@mpbc.swapaxes(-1,-2)
	if reduce[0]:
		t1=t1.squeeze(-2)
	if reduce[1]:
		t1=t1.squeeze(-1)
	return mpab-t1.reshape(*mpab.shape)

def mmsquare_v1(mp0aa,mpac,mpcci):
	"""
	Double modified square for transformed data after covariate removal.
	"""
	mpac=mpac.reshape(mpac.shape[0],*([] if mpac.ndim==2 else [-1]),mpac.shape[-1])
	reduce=mpac.ndim<mpcci.ndim
	if reduce:
		mpac=mpac.view(*mpac.shape[:-1],1,mpac.shape[-1])
	t1=((mpac@mpcci)*mpac).sum(axis=-1)
	if reduce:
		t1=t1.squeeze(-1)
	return torch.clamp(mp0aa-t1.reshape(*mp0aa.shape),min=0)

def mmsquare(mp0aa,mpac,mpcci):
	"""
	Double modified square for transformed data after covariate removal.
	"""
	return torch.clamp(mp0aa-((mpac@mpcci)@mpac.swapaxes(-1,-2)),min=0)


assert __name__ != "__main__"
