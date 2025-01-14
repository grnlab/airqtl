#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

from typing import Callable, Optional, Tuple, Union

import scipy
from numpy.typing import NDArray


def sim1_sample_cells(selectc:NDArray,dg:NDArray,cid:NDArray[int],dimd:NDArray,ncs:NDArray[int],K:Optional[NDArray],nd0:int,nc0:int,ncs1:Optional[NDArray[int]]=None,nocheck_ncs:bool=False):
	"""
	Down or up-sample cells in the dataset.
	selectc:	Index or boolean of cells to keep.
	dg,...,nc0:	Original dataset to select cells.
	ncs1:		Number of cells for each donor in the new dataset. Useful to specify more or fewer donors than original. If None, use the same donors and compute cell count based on boolean selectc.
	"""
	import logging

	import numpy as np
	assert len(selectc)==nc0 or (np.issubdtype(selectc.dtype,np.integer) and selectc.max()<nc0)
	assert cid.shape==(nc0,)
	assert ncs1 is not None or ncs.shape==(nd0,)
	assert K is None or K.shape==(nd0,nd0)
	assert (ncs>=0).all()
	assert nocheck_ncs or ncs.sum()==nc0
	#Update cell variables
	cid=cid[selectc]
	nc=len(cid)
	if ncs1 is None:
		#Must be downsampling cells with boolean index
		assert np.issubdtype(selectc.dtype,np.bool_)
		nd=nd0
		ncsc=np.r_[0,ncs.cumsum()]
		ncs1=np.array([selectc[ncsc[x]:ncsc[x+1]].sum() for x in range(nd)])
	else:
		assert ncs1.sum()==nc
		assert ncs1.shape==(nd0,)
		nd=len(ncs1)
	#Remove empty donors
	selectd=ncs1>0
	if not selectd.all():
		logging.warning('Actual number of donors ({}) is smaller than requested due to cell downsampling.'.format(selectd.sum()))
		assert (ncs1[~selectd]==0).all()
		ans=sim1_sample_donors(selectd,dg,cid,dimd,ncs1,K,nd,nc)
	else:
		ans=(dg,cid,dimd,ncs1,K,nd,nc)
	sim1_sample_check(*ans)
	return ans

def sim1_sample_donors(selectd:NDArray,dg:NDArray,cid:NDArray[int],dimd:NDArray,ncs:NDArray[int],K:Optional[NDArray],nd0:int,nc0:int):
	"""
	Down or up-sample donors in the dataset.
	selectd:	Index or boolean of donors to keep.
	"""
	import numpy as np
	assert len(selectd)==nd0 or (np.issubdtype(selectd.dtype,np.integer) and selectd.max()<nd0)
	assert cid.shape==(nc0,)
	assert all(x.shape[-1]==nd0 for x in [dg,dimd,ncs])
	assert K is None or K.shape==(nd0,nd0)
	assert (ncs>=0).all()
	if np.issubdtype(selectd.dtype,np.bool_):
		selectd=np.nonzero(selectd)[0]

	#Update donor variables
	if K is not None:
		K=K[selectd,:][:,selectd]
	dg=dg[:,selectd]
	dimd=dimd[selectd]
	nd1=len(dimd)
	ncsc=np.r_[0,ncs.cumsum()]
	selectc=np.concatenate([np.arange(ncsc[x],ncsc[x+1]) for x in selectd])
	ncs1=ncs[selectd]
	ans=sim1_sample_cells(selectc,dg,cid,dimd,ncs,K,nd1,nc0,ncs1=ncs1)
	sim1_sample_check(*ans)
	return ans

def sim1_sample_check(dg:NDArray,cid:NDArray[int],dimd:NDArray,ncs:NDArray[int],K:Optional[NDArray],nd:int,nc:int):
	"""
	Check the validity of up/downsampled dataset
	"""
	import numpy as np
	ng=dg.shape[0]
	assert dg.shape[1]==nd and cid.shape==(nc,)
	assert dimd.shape==(nd,)
	assert ncs.shape==(nd,) and (ncs>0).all() and ncs.sum()==nc
	assert K is None or K.shape==(nd,nd)

def sim_net_struct_none(l:float,n:int)->NDArray:
	"""
	Generate binary network with no edge
	"""
	import numpy as np
	assert l==0
	return np.zeros((n,n),dtype=bool)

def sim_net_struct_randomp(l:float,n:int)->NDArray[bool]:
	"""
	Generate random binary network with n nodes. Each node has x~Poisson(l) incoming edges (if allowed) from preceeding nodes with equal probability.
	l:	Mean number of incoming edges per node
	n:	Number of nodes
	"""
	import numpy as np
	e0=np.zeros((n,n),dtype=bool)
	ne=np.random.poisson(l,size=n-1)
	for xi in range(1,n):
		e0[np.random.choice(xi,ne[xi-1],replace=False) if ne[xi-1]<xi else np.arange(xi),xi]=True
	t1=np.arange(n)
	np.random.shuffle(t1)
	return e0[t1,:][:,t1]

def sim_net_struct_bap(l:float,n:int,const:float=None)->NDArray[bool]:
	"""
	Generate barabasi-albert binary network with n nodes. Each node has x~Poisson(l) incoming edges (if allowed) from preceeding nodes with probability proportional to existing outgoing edges + const.
	l:		Mean number of incoming edges per node
	n:		Number of nodes
	const:	Constant added to existing outgoing edges. Defaults to l.
	"""
	import numpy as np
	if const is None:
		const=l
	e0=np.zeros((n,n),dtype=bool)
	ne=np.random.poisson(l,size=n-1)
	prob=np.repeat(const,n-1)
	for xi in range(1,n):
		t1=prob[:xi]/prob[:xi].sum()
		t2=np.random.choice(xi,ne[xi-1],p=t1,replace=False) if ne[xi-1]<xi else np.arange(xi)
		e0[t2,xi]=True
		prob[t2]+=1
	t1=np.arange(n)
	np.random.shuffle(t1)
	return e0[t1,:][:,t1]

def sim_net_strength_normal(mean:float,var:float,net:NDArray[bool])->NDArray:
	"""
	Generate random network with strength from binary network.
	mean:	Mean of normal distribution of strength
	var:	Variance of normal distribution of strength
	net:	Binary network
	"""
	import numpy as np
	ne=net.sum()
	ans=np.zeros(net.shape,dtype=float)
	if ne>0:
		t1=np.random.randn(ne)*np.sqrt(var)+mean
		ans[net]=t1
	return ans

def compute_nettot(net:NDArray[bool])->NDArray:
	"""
	Computes total effect network from direct effect network.
	net:	Direct effect network
	Return:	Total effect network
	"""
	import networkx as nx
	import numpy as np

	assert net.ndim==2
	n=net.shape[0]
	assert net.shape[1]==n
	# Test DAG
	net2=nx.DiGraph(list(zip(*np.nonzero(net))))
	assert nx.is_directed_acyclic_graph(net2)
	# Get order
	order=list(nx.topological_sort(net2))
	# Adding genes with no edge
	order=np.array(sorted(list(set(range(n))-set(order)))+order)
	# Compute total effect
	ans=np.zeros(net.shape,dtype=float)
	for xj in range(1,n):
		ans[order[:xj],order[xj]]=net[order[:xj],order[xj]]+ans[order[:xj],order[:xj]]@net[order[:xj],order[xj]]
		# for xi in range(xj):
		# 	ans[order[xi],order[xj]]=net[order[xi],order[xj]]+ans[order[xi],order[:xi]]@net[order[:xi],order[xj]]
	# Validation
	assert ((ans!=0)^(net==0)).all()
	return ans

def sim1_b_prob(locs:Tuple[NDArray,NDArray],bp_a:float=0.1,bp_b:float=1E5,bp_c:float=1,bp_bound:int=1000000,distdep:bool=False):
	"""
	Generate b_prob_{ij} based on distance
	b_prob_{ij}~Bern(b_probraw_{ij})
	b_probraw_{ij}=bp_a*exp(-|dist|^{bp_c}/bp_b) if |dist|<bp_bound on the same chromosome and 0 otherwise
	If distdep is True: b_probraw_{ij} is multiplied by the distance between left and right nearest SNPs (or double of one-sided distance of no SNP rests on the other side).

	locs:	[location of SNPs, location of genes]
			Each location has shape (n_x,3), where n_x is the number of SNPs or genes.
			Each column:
				0:	Chromosome
				1:	Start position
				2:	Stop position
	Return:
		b_prob:	Probability of SNP i affecting gene j as coo_array
	"""
	from functools import reduce
	from operator import or_

	import numpy as np
	from scipy.sparse import coo_array
	if bp_a==0:
		return coo_array(([],([],[])),shape=(locs[0].shape[0],locs[1].shape[0]))

	#(row or SNP,col or gene)
	ans=[[] for _ in range(3)]
	chrs=sorted(list(reduce(or_,[set(x[:,0]) for x in locs])))
	for chr in chrs:
		ids=[np.nonzero(x[:,0]==chr)[0] for x in locs]
		if any(len(x)==0 for x in ids):
			continue
		#ordered SNP locations
		sids0=[np.argsort(locs[x][ids[x],1]) for x in range(2)]
		sids=[ids[x][sids0[x]] for x in range(2)]
		svals=locs[0][sids[0],1]
		if distdep:
			#Compute for each SNP the distance between left and right nearest SNPs or double of single-sided distance if other side is not available
			if len(svals)>2:
				dist_between=np.r_[(svals[1]-svals[0])*2,svals[2:]-svals[:-2],(svals[-1]-svals[-2])*2]
			else:
				dist_between=np.ones(svals.size,dtype=int)
			assert dist_between.size==svals.size
			assert dist_between.min()>0
		#Find windows of each gene
		window_l=np.searchsorted(svals,locs[1][ids[1],1]-bp_bound)
		window_r=np.searchsorted(svals,locs[1][ids[1],2]+bp_bound)
		window=np.array([window_l,window_r-window_l]).T
		#Find SNP-gene pairs
		ans[0].append(np.concatenate([sids[0][x[0]:x[0]+x[1]] for x in window]))
		ans[1].append(np.repeat(ids[1],window[:,1]))
		ans[2].append(np.concatenate([sids0[0][x[0]:x[0]+x[1]] for x in window]))
		assert len(ans[0][-1])==len(ans[1][-1])
		#Compute distance
		dist=(locs[0][ans[0][-1],1]-locs[1][ans[1][-1],1])*np.sign(locs[1][ans[1][-1],2]-locs[1][ans[1][-1],1]).astype(int)
		#Compute b_probraw
		bprobraw=np.exp(-(np.abs(dist)**bp_c)/bp_b)*bp_a
		if distdep:
			bprobraw*=dist_between[ans[2][-1]]/dist_between[ans[2][-1]].mean()
		#Compute b_prob
		bprob=np.random.rand(bprobraw.size)<bprobraw
		#Remove pairs with b_prob=0
		ans[0][-1]=ans[0][-1][bprob]
		ans[1][-1]=ans[1][-1][bprob]
	#Merge
	if len(ans[0])==0:
		return coo_array(([],([],[])),shape=(locs[0].shape[0],locs[1].shape[0]),dtype=bool)
	ans=np.array([np.concatenate(x) for x in ans[:2]])
	ans=coo_array((np.ones(ans.shape[1],dtype=bool),(ans[0],ans[1])),shape=(locs[0].shape[0],locs[1].shape[0]))
	return ans

def sim1_b(dg1,de1,dc1,ncs,bprob,scalea:Union[float,NDArray[float],None]=None,scaleb:float=1,dbprob1:float=0,dbprob2:float=0,dbc:Optional[NDArray[bool]]=None)->Tuple[Optional[NDArray[float]],scipy.sparse.coo_array,scipy.sparse.coo_array]:
	"""
	Generate b_{ij} based on b_prob_{ij}
	dg1:		Genotype of shape (n_SNP,n_donor)
	de1:		Expression of shape (n_gene,n_cell) as normalized read counts
	dc1:		Measured confounder of shape (n_cov,n_cell)
	ncs:		Number of cells for each donor of shape (n_donor,)
	bprob:		Probability of SNP i affecting gene j as coo_array
	scalea:		Scale of effect size of measured confounder on gene expression compared to inferred from reference level. This can be set to 0 for certain technical confounders to disable their effect on true gene expression level. If None, will not estimate confounder effect.
	scaleb:		Scale of effect size of SNPs on gene expression compared to inferred from reference level. This can be set higher when positive SNPs are dense, i.e. their estimated effect sizes are smaller than actual ones.
	dbprob1:	Probability of having a confounder dependent effect size for each b
	dbprob2:	Probability for each confounder-dependent b to depend on each confounder
	dbc:		Whether each confounder can affect b of shape (n_cov,). If None, assume all confounders affect b.
	Return:
		a:	Effect size of measured confounder on gene expression as sparse matrix of shape (n_cov,n_gene), or None if scalea is None
		b:	Effect size of SNPs on gene expression as sparse matrix of shape (n_SNP,n_gene).
		db:	Confounder dependent effect size of SNPs on gene expression as sparse matrix of shape (n_cov*n_SNP,n_gene).
	"""
	import numpy as np
	from scipy.sparse import coo_array

	assert dg1.shape[0]==bprob.shape[0]
	assert de1.shape[0]==bprob.shape[1]
	assert dbprob1>=0 and dbprob2>=0
	if scalea is not None:
		if isinstance(scalea,float):
			scalea=np.repeat(scalea,dc1.shape[0])
		if scalea.shape!=(dc1.shape[0],):
			raise ValueError('scalea must have shape (n_cov,).')
		if (scalea<0).any():
			raise ValueError('Confounder effect size scaling must be nonnegative.')
	if dbc is None:
		dbc=np.ones(dc1.shape[0],dtype=bool)
	if dbc.shape!=(dc1.shape[0],):
		raise ValueError('dbc must have shape (n_cov,).')
	dbc=np.nonzero(dbc)[0]

	ansa=np.zeros((dc1.shape[0],de1.shape[0]),dtype=float)
	ansb=[[] for x in range(3)]
	ansdb=[[] for x in range(3)]
	for xi in range(de1.shape[0]):
		t1=bprob[:,[xi]].nonzero()[0]
		if t1.size==0:
			continue
		#Linear fit
		dg2=dg1[t1,:]
		dg2=dg2[:,np.repeat(np.arange(dg2.shape[1]),ncs)]
		shapes=[x.shape[0] for x in [dg2,dc1]]
		dg2=np.concatenate([dg2,dc1],axis=0)
		if dbprob1>0:
			tprob1=np.nonzero(np.random.rand(shapes[0])<dbprob1)[0]
		else:
			tprob1=[]
		if len(tprob1)>0 and dbprob2>0:
			#Confounder dependent effect size
			tprob2=np.nonzero(np.random.rand(len(dbc),len(tprob1))<dbprob2)
			if len(tprob2[0])>0:
				tprob2=[dbc[tprob2[0]],tprob1[tprob2[1]]]
				t3=dg2[shapes[0]:][tprob2[0]]*dg2[:shapes[0]][tprob2[1]]
				dg2=np.concatenate([dg2,t3],axis=0)
				shapes.append(t3.shape[0])
		#Linear model fit
		t2=np.linalg.lstsq(dg2.T,de1[xi,:],rcond=None)[0]
		ansa[:,xi]=t2[shapes[0]:shapes[0]+shapes[1]]
		ansb[0].append(t1)
		ansb[1].append(np.repeat(xi,t1.size))
		ansb[2].append(t2[:shapes[0]]*scaleb)
		assert all(len(x[-1])==len(ansb[0][-1]) for x in ansb[1:])
		if len(tprob1)>0 and len(tprob2[0])>0:
			ansdb[0].append(tprob2[0]*dg1.shape[0]+t1[tprob2[1]])
			ansdb[1].append(np.repeat(xi,len(ansdb[0][-1])))
			ansdb[2].append(t2[shapes[0]+shapes[1]:].ravel())
	if len(ansb[0])==0:
		ansb=coo_array(([],([],[])),shape=(dg1.shape[0],de1.shape[0]))
	else:
		ansb=[np.concatenate(x) for x in ansb]
		ansb=coo_array((ansb[2],(ansb[0],ansb[1])),shape=(dg1.shape[0],de1.shape[0]))
	if len(ansdb[0])==0:
		ansdb=coo_array(([],([],[])),shape=(dc1.shape[0]*dg1.shape[0],de1.shape[0]))
	else:
		ansdb=[np.concatenate(x) for x in ansdb]
		ansdb=coo_array((ansdb[2],(ansdb[0],ansdb[1])),shape=(dc1.shape[0]*dg1.shape[0],de1.shape[0]))
	return ((ansa.T*scalea).T if scalea is not None else None,ansb,ansdb)

def sim1_resample(de0,de,dg,dc,ne,ng,nc,nd,ne0,ng0,nc0,nd0,ncs,locs,dime,dimg,dimc,dimd,cisonly,bp_bound,upsample,K):
	"""Resampling data to the desired size. Includes up and downsampling as needed.
	"""
	import logging
	from collections import Counter, defaultdict

	import numpy as np

	from .utils.eqtl import find_cis

	# Downsampling
	locs1=[None,None]
	if ne is None:
		de1=de
		ne=ne0
		locs1[1]=locs[1]
		dime1=dime
	elif ne<ne0:
		t1=np.sort(np.random.choice(np.arange(ne0),ne,replace=False))
		de1,de0,locs1[1],dime1=[x[t1] for x in [de,de0,locs[1],dime]]
	else:
		raise NotImplementedError('Upsampling genes not implemented.')
	if cisonly:
		#Filter SNPs within the cis region of any gene
		t1=find_cis([locs[0],locs1[1]],bp_bound)
		t1=np.unique(t1[0])
		dg1,locs1[0],dimg1=[x[t1] for x in [dg,locs[0],dimg]]
	else:
		dg1=dg
		locs1[0]=locs[0]
		dimg1=dimg
	if ng is None:
		#Use all SNPs after optional cis filtering
		ng=len(dimg1)
	elif ng>ng0:
		raise NotImplementedError('Upsampling genotypes not implemented.')
	elif ng>len(dimg1):
		raise NotImplementedError('Upsampling genotypes after filtering cis-SNPs is not implemented.')
	elif ng<len(dimg1):
		t1=np.sort(np.random.choice(len(dimg1),ng,replace=False))
		dg1,locs1[0],dimg1=[x[t1] for x in [dg1,locs1[0],dimg1]]
	if nd is None:
		nd=nd0
	if nc is None:
		nc=nc0
	#Donor sampling
	cid=np.arange(nc0)
	if nd!=nd0:
		if nd<nd0:
			ud=False
			#Determine if upsampling cells is necessary
			t1=np.argsort(ncs)[::-1]
			if ncs[t1[:nd]].sum()<nc:
				if not upsample:
					raise RuntimeError('Upsampling not allowed.')
				uc=True
			else:
				uc=False
			#Initial downsample donor
			t2=np.random.choice(nd0,nd,replace=False)
			niter=0
			while not uc and ncs[t2].sum()<nc and niter<nd0:
				#Keep removing the smallest donor and add a larger one
				t3=np.argmin(ncs[t2])
				t4=list(set(np.nonzero(ncs>ncs[t2[t3]])[0])-set(t2))
				t2=np.concatenate([t2[:t3],np.random.choice(t4,1),t2[t3+1:]])
				niter+=1
			assert niter<nd0
			selectd=np.sort(t2)
		elif nd>nd0:
			ud=True
			if not upsample:
				raise RuntimeError('Upsampling not allowed.')
			if K is not None:
				raise ValueError('Donor upsampling not allowed with K.')
			logging.warning('Upsampling donors.')
			selectd=np.r_[np.arange(nd0),np.random.choice(nd0,nd-nd0,replace=True)]
	else:
		ud=False
		selectd=np.arange(nd0)
	dg1,cid,dimd1,ncs1,K,nd1,nc1=sim1_sample_donors(selectd,dg1,cid,dimd,ncs,K,nd0,nc0)
	assert nd1==nd
	if ud:
		#Randomize genotypes for new donors
		t1=np.argsort(np.random.rand(dg1.shape[0],nd-nd0),axis=1)
		t2=np.repeat(np.arange(dg1.shape[0]).reshape(-1,1),nd-nd0,axis=1)
		dg1=np.concatenate([dg1[:,:nd0],dg1[:,nd0:][t2,t1]],axis=1)
	#Cell sampling
	if nc<nc1:
		#Downsample cells
		uc=False
		t1=np.sort(np.random.choice(nc1,nc,replace=False))
		selectc=np.zeros(nc1,dtype=int)
		selectc[t1]=1
		assert (selectc<=1).all()
	elif nc>nc1:
		#Upsample cells
		uc=True
		if not upsample:
			raise RuntimeError('Upsampling not allowed.')
		logging.warning('Upsampling cells.')
		selectc=Counter(np.r_[np.arange(nc1),np.random.choice(nc1,nc-nc1,replace=True)])
		selectc=np.array(list(selectc.items())).T
		selectc=selectc[1,np.argsort(selectc[0])]
		assert (selectc>=1).all()
	else:
		uc=False
		selectc=np.ones(nc1,dtype=int)
	assert len(selectc)==nc1
	#Update cell variables
	ncsc1=np.r_[0,ncs1.cumsum()]
	assert ncsc1[-1]==nc1
	ncs2=np.array([selectc[ncsc1[x]:ncsc1[x+1]].sum() for x in range(nd)])
	selectc2=np.repeat(np.arange(len(selectc)),selectc)
	assert len(selectc2)==nc
	dg1,cid,dimd1,ncs1,K,nd2,nc2=sim1_sample_cells(selectc2,dg1,cid,dimd1,ncs1,K,nd1,nc1,ncs1=ncs2)
	assert nc2==nc and nd2<=nd
	de1,de0,dc1,dimc1=[x[...,cid] for x in [de1,de0,dc,dimc]]

	#Remove donors with no cells
	t1=ncs1>0
	if not t1.all():
		dg1,ncs1,dimd1=[x[...,t1] for x in [dg1,ncs1,dimd1]]
		K=K[t1,:][:,t1]
		nd2=t1.sum()

	#Ensure unique names with suffix
	t1=defaultdict(int)
	t2=[]
	for xi in dimd1:
		t1[xi]+=1
		t2.append(f'{xi}-{t1[xi]}')
	if all(x.endswith('-1') for x in t2):
		t2=[x[:-2] for x in t2]
	dimd1=np.array(t2)
	t1=defaultdict(int)
	t2=[]
	for xi in dimc1:
		t1[xi]+=1
		t2.append(f'{xi}-{t1[xi]}')
	if all(x.endswith('-1') for x in t2):
		t2=[x[:-2] for x in t2]
	dimc1=np.array(t2)
	assert len(dimd1)==len(set(dimd1))
	assert len(dimc1)==len(set(dimc1))

	return (de0,de1,dg1,dc1,ne,ng,nc,nd2,ncs1,locs1,dime1,dimg1,dimc1,dimd1,K)

def sim1(
	#Data
	dg:NDArray,de:NDArray,de0:NDArray[int],dc:NDArray[float],ncs:NDArray[int],locs:Tuple[NDArray,NDArray],
	#GRN parameters
	grnmeth_struct:Callable[[int],NDArray[bool]],grnmeth_effect:Callable[NDArray[bool],NDArray],
	#Dimension labels
	dimg,dime,dimc,dimd,
	#Data resampling: sizes
	ng:Optional[int]=None,ne:Optional[int]=None,nd:Optional[int]=None,nc:Optional[int]=None,
	#Data resampling: other parameters
	upsample:bool=True,cisonly:bool=False,permg:bool=False,
	#SNP effect parameters
	bp_a:float=0.1,bp_b:float=1E5,bp_c:float=1,bp_bound:int=1000000,distdep:bool=False,
	#Noise parameters
	sigma0:float=0,sigma1:float=0,sigma2:float=0,K:Optional[NDArray]=None,
	#Confounding effect parameters
	scalea:Union[float,NDArray[float],None]=None,scaleb:float=1,
	#Confounding*SNP effect parameters
	dbprob1:float=0,dbprob2:float=0,dbc:Optional[NDArray[bool]]=None,
	#Misc
	na:Optional[int]=None,forcena:bool=False,scaleumi:float=0,scaleprop:float=0,seed:int=None
	):
	"""
	Simulate null and non-null datasets for one cell type based on real datasets.
	Using real genotype to simulate cis-eQTL and expression.
	Downsamples genes, genotypes, cells, and donors if needed.	
	Genotype j is named g_j. Gene i has single-cell transcriptomic read count t_i
	t_i~B(n,softmax(t1_i)), where B is binomial distribution and
	t0_i=(sum_j b_{ij}*g_j)+(sum_{jk} deltab_{ijk}*g_j*c_k)+(sum_j<i net_{ij}*t0_j)+e0_i+e1_i+e2_i,
	t1_i=t0_i+(sum_j a_{ij}*c_j)
	is true log expression.

	Contributions to transriptomic levels:
	* From confounders:
		c_j:		Value of measured confounder j with mean=0 except for intercept term. Only uses intercept term as all-one if scalea is None. Can use select confounders by setting scalea to 0 for others.
		a_{ij}:		Effect size of measured confounder j on gene i. Fitted by linear regression on normalized data.
	* From genotypes:
		g_j:		Value of genotype j with mean=0. Uses real genotype data.
		b_{ij}:		Effect size of genotype j (or of a nearby unmeasured genotype which has j as the best proxy) on gene i. b_{ij}=b_prob_{ij}*b_value_{ij}.
			b_prob_{ij}~Bern(b_probraw_{ij})
			b_probraw_{ij}=bp_a*exp(-|dist|^{bp_c}/bp_b) if |dist|<bp_bound on the same chromosome and 0 otherwise
			If distdep is True: b_probraw_{ij} is multiplied by the distance between left and right nearest SNPs (or double of one-sided distance of no SNP rests on the other side).
			b_value_{ij}: Fitted by linear regression of normalized t_i on all g_ij with nonzero b_prob_{ij}.
			Hyperparameter estimation outside this function:
			1. Determine b_probraw_{ij} as the distribution of cis-eQTL frequency by distance.
			2. Determine bp_a based on the distibution of Pearson R between cis-genotype and normalized expression. If observing stronger but fewer Pearson R, increase bp_a.
	* From genotypes*confounders:
		deltab_{ijk}:	Effect size of genotype j (or of a nearby unmeasured genotype which has j as the best proxy) * confounder k on gene i.
		deltab_{ijk}=deltab_prob1_{ij}*deltab_prob2_{ijk}*deltab_value_{ijk}.
			deltab_prob1_{ij}~Bern(dbprob1)
			deltab_prob2_{ijk}~Bern(dbprob2)
			deltab_value_{ijk}: Fitted by linear regression along with genotypes and confounders.
		Disabled (by default) when setting dbprob1=0.
	* From GRN:
		net_{ij}:	DAG GRN effect size of gene j on gene i
			Structure: depends on grnmeth_struct
				randomp:	use a random order for genes and create random sparse DAG with a Poisson in-degree.
			Effect size: depends on grnmeth_effect
				normal:		~ iid N(mean,var)
			Hyperparameter estimation: Determine sigman with distribution of co-expression strength. Needs sigman<1.88736 to avoid divergence.
	* From noise:
		e0_i~MVN(0,sigma0_i^2):		Cell level noise of gene i with mean=0. If not specified, sigma0_i is determined by cell level variance below precomputed from normalized transcriptome.
		e1_i~MVN(0,sigma1_i^2*K_d):		Donor level noise of gene i with mean=0. If not specified, sigma1_i is determined by donor level variance below precomputed from normalized transcriptome.
		e2_i~iid N(0,sigma2_i^2*K_g)):	Genetic level noise of gene i with mean=0. If not specified, sigma2_i is determined by genetic level variance below precomputed from normalized transcriptome.
	
	Total variance contributions:
	* Total network definition
		nettot_{ij}=net_{ij}+sum_{k=j+1,..,i-1} nettot_{ik}net_{kj}
	* At cell level (varc_i):
		varc_i=sigma0_i^2+sum_j<i (nettot_{ij}^2*varc_j)
	* At donor level (vard_i):
		vard_i=sigma1_i^2+sum_j<i (nettot_{ij}^2*vard_j)
	* At genetic random effect level (varg_random_i):
		varg_random_i=sigma2_i^2+sum_j<i (nettot_{ij}^2*varg_random_j)
	* At genetic level (varg_i):
		varg_i=var(sum_j ((b_{ij}+sum_k<i nettot_{ik}*b_{kj})*g_j))

	Benchmarking controls:
		Positive controls: genotype i and gene j with b_{ij}!=0
		Negative controls:
			Normal:	genotype i and gene j with b_{ik}=0 for all k in gene j, its parents, grandparents, etc according to net.
			Strict:	genotype i and gene j with genotype i and gene j & k on different chromosomes for all k in gene j's parents, grandparents, etc according to net.
		Effect size:	Positive controls after permuting genotypes to remove linkage disequilibrium.
	
	Dimension up/downsampling:
		Dimensions include genotypes, genes, cells, and donors. When requested dimension size is lower than the original, downsampling is performed by selecting a subset at random. When requested dimension size is higher than the original, upsampling is performed by selecting all elements and additional elements at random. Exception: when donor is downsampled, cell upsampling will be avoided if possible by forcing non-random donor downsampling. Upsampling for some dimensions may be not implemented. Upsampled datasets are suggested to be used only for time and memory complexity benchmarking.

	Parameters:
	dg:		Genotype of shape (n_SNP,n_donor)
	de:		Expression of shape (n_gene,n_cell) as normalized read counts
	de0:	Expression of shape (n_gene,n_cell) as raw read counts
	dc:		Measured confounder of shape (n_cov,n_cell). Only used to estimate hyperparameter b.
	ncs:	Number of cells for each donor
	locs:	[location of SNPs, location of genes]
			Each location has shape (n_x,3), where n_x is the number of SNPs or genes.
			Each column:
				0:	Chromosome
				1:	Start position
				2:	Stop position
	grnmeth_struct:		GRN structure generation method as function(node_count). See contribution section.
	grnmeth_effect:		GRN effet size generation method as function(binary_net). See contribution section.
	dimg,
	dime,
	dimc,
	dimd:		Names of genotype, expression, cell, and donor.
	ng:			Number of genotypes in simulated dataset. Default: same as original dataset.
	ne:			Number of genes in simulated dataset. Default: same as original dataset.
	nd:			Number of donors in simulated dataset. Default: same as original dataset.
	nc:			Number of cells in simulated dataset. Default: same as original dataset.
	upsample:	Whether to allow upsampling genotypes/genes/donors/cells. Only some are implemented. Raises a RuntimeError if upsampling is necessary but not allowed.
	cisonly:	Whether to filter cis-SNPs of simulated genes only.
	permg:		Whether to permute every genotype among donors to remove linkage disequilibrium.
	bp_a,bp_b,bp_c,bp_bound,distdep:	Model hyperparameters. See contribution section.
	sigma0,
	sigma1,
	sigma2:		Cell, donor, and genetic level noise of single float or of shape (n_gene,).
	scalea:		Scale of effect size of measured confounder on gene expression compared to inferred from reference level. This can be set to 0 for certain technical confounders to disable their effect on true gene expression level. If None, will ignore confounder effect and only use mean expression level estimated from pseudobulk CPM.
	scaleb:		Scale of effect size of SNPs on gene expression compared to inferred from reference level. This can be set higher if positive SNPs are dense so their estimated effect sizes are smaller than actual ones.
	dbprob1:	Probability of having a confounder dependent effect size for each b
	dbprob2:	Probability for each confounder-dependent b to depend on each confounder
	dbc:		Whether each confounder can affect b of shape (n_cov,). If None, assume all confounders affect b.
	na:			Number of alleles in simulated dataset. Default: same as original dataset.
	forcena:	Whether to force all possible allele values to exist in each genotype.
	scaleumi:	Scale of all UMI counts per cell compared to reference level. If 0 and not resampling cells, use real UMI counts.
	scaleprop:	Exponential scale of proportion of gene expressions from reference level. If 0 and not resampling genes, use real proportions.
	seed:		Random seed. If None, use current time. This value is added with other parameters to form the final random seed.

	Return:
	dg,
	de0,
	dc,
	ncs,
	locs:	Data. See input format.
	de_true:True expression proportion of shape (n_gene,n_cell)
	da:		Effect size of confounders on gene expression of shape (n_cov,n_gene).
	db:		Effect size of SNPs on gene expression as sparse matrix of shape (n_SNP,n_gene).
	ddeltab:Confounder dependent effect size of SNPs on gene expression as sparse matrix of shape (n_cov*n_SNP,n_gene).
	dnet:	GRN effect size matrix of shape (n_gene for source,n_gene for target).
	dnettot:Total GRN effect size matrix of shape (n_gene for source,n_gene for target).
	dimg,
	dime,
	dimc,
	dimd:	Names of genotype, expression, cell, and donor.
	"""
	def calcprops(dt):
		"""
		dt: (n_gene,n_cell) as raw read counts
		"""
		def tocdf(d):
			import numpy as np
			assert len(d.shape) == 1
			d = d.copy()
			d.sort()
			d = np.array([np.arange(d.size) / (d.size - 1), d])
			return d

		ntot = dt.sum(axis=0)
		nprop = dt.sum(axis=1)
		nprop = nprop / nprop.sum()
		t1 = dt > 0
		pcsig = t1.mean(axis=1)
		pgsig = t1.mean(axis=0)
		ntot = tocdf(ntot)
		nprop = tocdf(nprop)
		pcsig = tocdf(pcsig)
		pgsig = tocdf(pgsig)
		return ntot, nprop, pcsig, pgsig

	import logging
	from collections import Counter

	import numpy as np
	import scipy.sparse

	# from scipy.stats import beta as betadist

	ng0,nd0=dg.shape
	ne0,nc0=de.shape
	if any(x==0 for x in [ng0,ne0,nd0,nc0]):
		raise ValueError('Input data must be non-empty.')
	if any(len(x)!=y for x,y in [(dimg,ng0),(dime,ne0),(dimc,nc0),(dimd,nd0)]):
		raise ValueError('Input data must have correct dimension.')
	if any(x is not None for x in [na]):
		raise NotImplementedError('na must be None.')
	if na is not None and na > 255:
		raise ValueError('At most 255 alleles allowed.')
	if forcena:
		raise NotImplementedError('forcena=True not implemented.')
	if scaleumi < 0:
		raise ValueError('Invalid UMI scaling.')
	if scaleprop < 0:
		raise ValueError('Invalid expression proportion scaling.')
	if scaleb <= 0:
		raise ValueError('Genotype effect size scaling must be nonnegative.')
	assert all(x>=0 for x in [sigma0,sigma1,sigma2])
	assert any(x>0 for x in [sigma0,sigma1,sigma2])
	assert sigma2==0 or K is not None
	if K is not None:
		assert K.shape==(nd0,nd0)
	if seed is not None:
		seed+=4174712 + (ng0+(ng if ng is not None else 0))*(nd0+(nd if nd is not None else 0)) + (ne0+(ne if ne is not None else 0))*(nc0+(nc if nc is not None else 0))
		seed+=(1624*cisonly+int(5214*bp_a)+int(12815*np.log(bp_b))+int(81453*np.log(min(bp_c,1)))+int(58765*np.log(bp_bound))) % (2**32)
		seed+=(5284*distdep+int(915435*sigma0)+int(161234*sigma1)+int(746512*sigma2)+6142*forcena+int(917235*scaleumi)+int(72364*scaleprop)+81549*((scalea is None)-1)+(0 if scalea is None else int(541449*np.sum(scalea)))+int(8154934*(scaleb-1))) % (2**32)
		seed+=(int(dbprob1*71529461)+int(dbprob2*9461721)) % (2**32)
		seed=seed % (2**32)
		np.random.seed(seed)

	# Data resampling
	de0,de1,dg1,dc1,ne,ng,nc,nd2,ncs1,locs1,dime1,dimg1,dimc1,dimd1,K=sim1_resample(de0,de,dg,dc,ne,ng,nc,nd,ne0,ng0,nc0,nd0,ncs,locs,dime,dimg,dimc,dimd,cisonly,bp_bound,upsample,K)

	if permg:
		#Permute genotypes to remove LD
		if K is not None:
			raise NotImplementedError
		dg1=np.array([np.random.permutation(x) for x in dg1],dtype=dg1.dtype)
	#Compute UMI count and expression proportion
	ntot, nprop, pcsig, pgsig = calcprops(de0)
	# total read counts, scaled with gene count
	if scaleumi==0 and nc<=nc0:
		#Use real values
		ntotn = de0.sum(axis=0)
	elif scaleumi==0:
		raise ValueError('Cannot downsample cells without UMI scaling.')
	else:
		#Use randomly sampled values
		ntotn = (np.interp(np.random.rand(nc), ntot[0], ntot[1]) * scaleumi).astype(int, copy=False)
	ntotn=np.clip(ntotn,1,None)
	# gene read proportions or average expression level
	if scaleprop==0 and ne<=ne0:
		#Use real values
		npropn = de0.sum(axis=1)
		npropn = npropn / npropn.sum()
	else:
		#Use randomly sampled values
		npropn = np.interp(np.random.rand(ne), nprop[0], nprop[1]**(scaleprop))
		npropn /= npropn.sum()
	dmean=np.log(npropn).reshape(1,-1)

	# Simulate genotype effect
	bprob=sim1_b_prob(locs1,bp_a=bp_a,bp_b=bp_b,bp_c=bp_c,bp_bound=bp_bound,distdep=distdep)
	logging.info('Average number of cis-eQTLs for each gene: {}'.format(bprob.sum()/bprob.shape[1]))
	a,b,deltab=sim1_b(dg1,de1,dc1,ncs1,bprob.tocsr(),scalea=scalea,scaleb=scaleb,dbprob1=dbprob1,dbprob2=dbprob2,dbc=dbc)
	if dbprob1==0:
		logging.info('Average effect size of cis-eQTLs: {}'.format(b.sum()/(b.count_nonzero()+1E-100)))
		logging.info('Average absolute effect size of cis-eQTLs: {}'.format(abs(b).sum()/(b.count_nonzero()+1E-100)))

	# Simulate network
	net=grnmeth_struct(ne)
	net=grnmeth_effect(net)
	nettot=compute_nettot(net)

	# Simulate true expression
	de_c=np.random.randn(ne,nc)*sigma0
	de_d=np.random.randn(ne,nd2)*sigma1
	if sigma2>0:
		de_gr=np.random.multivariate_normal(np.zeros(nd2),K,size=ne)*sigma2
	de_g=(dg1.T@b.tocsc()).T
	t1=np.repeat(np.arange(nd2),ncs1)
	if dbprob1>0:
		# Covariate dependent SNP effects, currently in compressed form
		#[n_cov,n_gene,n_donor]
		de_g2=np.array([(dg1.T@deltab.tocsr()[x*dg1.shape[0]:(x+1)*dg1.shape[0]].tocsc()).T for x in range(dc1.shape[0])])

	#Propagate effects on GRN
	for xi in range(ne):
		de_c[xi]+=de_c[:xi].T@nettot[:xi,xi]
		de_d[xi]+=de_d[:xi].T@nettot[:xi,xi]
		if sigma2>0:
			de_gr[xi]+=de_gr[:xi].T@nettot[:xi,xi]
		de_g[xi]+=de_g[:xi].T@nettot[:xi,xi]
		if dbprob1>0:
			de_g2[:,xi]+=de_g2[:,:xi].swapaxes(1,2)@nettot[:xi,xi]		
	
	de2=de_c+de_d[:,t1]+de_g[:,t1]
	if dbprob1>0:
		for xi in range(dc1.shape[0]):
			de2+=dc1[xi]*de_g2[xi][:,t1]
	if sigma2>0:
		de2+=de_gr[:,t1]
	if scalea is not None:
		#Use covariates to estimate expression expectation
		de2+=a.T@dc1
	else:
		#Use mean expression to estimate expression expectation
		de2=(de2.T+dmean).T

	# Simulate scRNA-seq read count
	deprop=np.exp(de2)
	deprop=deprop/deprop.sum(axis=0)
	de2=[np.array(list(Counter(np.random.choice(np.arange(ne),ntotn[x],p=deprop[:,x])).items())).T for x in range(nc)]
	de2=[np.concatenate([[np.zeros_like(x[0])],x],axis=0) for x in de2]
	de2=[np.asarray(scipy.sparse.coo_matrix((x[2],x[:2]),shape=(1,ne)).todense())[0] for x in de2]
	de2=np.array(de2).T.astype(de0.dtype,copy=False)
	assert (de2.sum(axis=0)==ntotn).all()
	assert dg1.shape == (ng, nd2)
	assert de2.shape == (ne, nc)
	assert dc1.shape == (dc.shape[0], nc)
	assert ncs1.shape == (nd2,)
	assert locs1[0].shape == (ng, 3)
	assert locs1[1].shape == (ne, 3)
	assert deprop.shape == (ne, nc)
	if scalea is None:
		assert a is None
	else:
		assert a.shape == (dc.shape[0], ne)
	assert b.shape == (ng, ne)
	assert deltab.shape == (ng * dc.shape[0], ne)
	assert net.shape == (ne, ne)
	assert nettot.shape == (ne, ne)
	assert dimg1.shape == (ng,)
	assert dime1.shape == (ne,)
	assert dimc1.shape == (nc,)
	assert dimd1.shape == (nd2,)
	assert all(len(x) == len(set(x)) for x in [dimg1, dime1, dimc1, dimd1])
	assert all(np.isfinite(x).all() for x in [dg1,de2,dc1,ncs1,deprop,net,nettot])
	assert (ncs1>0).all()
	assert all((x>=0).all() for x in [dg1,de2,deprop])
	return (dg1,de2,dc1,ncs1,locs1,deprop,a,b,deltab,net,nettot,dimg1,dime1,dimc1,dimd1)


assert __name__ != "__main__"
