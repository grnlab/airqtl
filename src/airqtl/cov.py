#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

def c2d(dc,ncs,names):
	"""Convert cell-level covariates to donor level. 
	dc: 	pandas.DataFrame of cell-level covariates of shape (n_cell,n_covariate). Cells should be order by donor.
	ncs:	Number of cells for each donor
	names:	Names of donors in matching order of ncs
	Return:	pandas.DataFrame of donor-level covariates of shape (n_donor,n_covariate)
	"""
	import numpy as np
	import pandas as pd
	assert (ncs>0).all()
	n=len(ncs)
	t1=np.r_[0,ncs].cumsum()
	assert all(ncs[x]==1 or (dc.values[t1[x]+1:t1[x+1]]==dc.values[t1[x]]).all() for x in range(n))
	dcnew=pd.DataFrame(dc.values[t1[:-1]],columns=dc.columns,index=names)
	return dcnew

def d2o(dc):
	"""Convert discrete covariates to one-hot representation. Only n-1 categories for each discrete covariates are used to avoid redundancy, where n is the number of different values for this covariate.
	dc: 	pandas.DataFrame of discrete covariates of shape (n_sample,n_covariate_discrete)
	Return:	
	dcnew:	pandas.DataFrame of one-hot covariates of shape (n_sample,n_covariate_onehot)
	grp:	numpy.ndarray(shape=(n_covariate_onehot,n_covariate_onehot),dtype=bool) of whether new covariate pairs are from the same group
	"""
	import numpy as np
	import pandas as pd
	dv=dc.values
	dcnew=[]
	cols=[]
	for xj0 in range(dv.shape[1]):
		xj=dv[:,xj0]
		#Not taking the last value to avoid redundancy and maintain the right degree of freedom
		t2=set(list(xj))
		t2=sorted(list(t2))[:-1] if 'Others' not in t2 else sorted(list(t2-{'Others'}))
		dcnew.append([xj==x for x in t2])
		cols+=['{}={}'.format(dc.columns[xj0],x) for x in t2]
	if len(cols)==0:
		#Account for empty covariates
		dcnew=np.zeros((dv.shape[0],0),dtype=dv.dtype)
	else:
		dcnew=np.concatenate(dcnew,axis=0).T
	assert dcnew.shape==(dv.shape[0],len(cols))
	dcnew=pd.DataFrame(dcnew,columns=cols,index=dc.index)
	grp=np.array([x.split('=')[0] for x in cols])
	grp=np.repeat(grp[:,None],len(grp),axis=1)==np.repeat(grp[None,:],len(grp),axis=0)
	return (dcnew,grp)

def o2d(dc,missing={}):
	"""Convert one-hot covariates to discrete representation.
	dc: 		pandas.DataFrame of one-hot covariates of shape (n_sample,n_covariate_onehot)
	missing:	The category name missing for each covariate because they are the last category. If not provided, they are called "Others".
	Return:		pandas.DataFrame of discrete covariates of shape (n_sample,n_covariate_discrete)
	"""
	from collections import defaultdict

	import numpy as np
	import pandas as pd
	assert ((dc.values==0)|(dc.values==1)).all()
	if dc.shape[1]==0:
		#Account for empty covariates
		return pd.DataFrame(np.zeros((dc.shape[0],0)),index=dc.index)
	dc=dc.copy()
	#Get covariate names
	covs=defaultdict(list)
	for xi in dc.columns:
		if not (dc[xi]!=0).any():
			continue
		t1=xi.split('=')
		assert len(t1)==2
		covs[t1[0]].append(t1[1])
	assert all((dc[[x+'='+y for y in covs[x]]].values.sum(axis=1)<=1).all() for x in covs)
	for xi in covs:
		t1=missing[xi] if xi in missing else 'Others'
		dc[xi+'='+t1]=1-dc[[xi+'='+y for y in covs[xi]]].values.sum(axis=1)
		covs[xi]=sorted(covs[xi]+[t1])

	dcnew=[]
	cols=[]
	for xi in covs:
		t1=dc[[xi+'='+y for y in covs[xi]]].values
		t1=np.argmax(t1,axis=1)
		t1=[covs[xi][x] for x in t1]
		dcnew.append(t1)
		cols.append(xi)
	dcnew=pd.DataFrame(np.array(dcnew).T,columns=cols,index=dc.index)
	assert dcnew.shape==(dc.shape[0],len(cols))
	#Drop single-valued covariates
	dcnew=dcnew.loc[:,dcnew.nunique()!=1].copy()
	return dcnew


assert __name__ != "__main__"
