#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

def estimate(dg,loc=None,method='raw',norm=True):
	"""Estimate genetic relatedness from genotype matrix using realised relatedness matrix.
	See Hayes, Visscher, and Goddard (2009). “Increased accuracy of artifi- cial selection by using the realized relationship matrix.”
	dg:		numpy.array(shape=(n_snp,n_donor)) Genotype matrix
	loc:	numpy.array(shape=(...,n_snp),dtype=int) Location of each genotype. Needed when method!='raw'. Row definitions:
			0:	Chromosome ID. Must starts with 1 and all chromosomes present.
			1:	Chromosome start position. Starts with 0. Optional and currently unused.
			2:	Chromosome end position. Starts with 0. Optional and currently unused.
	method:	Method to compute relatedness matrix. Accepts:
			'raw':	Computes one relatedness matrix from all genotypes
			'loco':	Leave one chromosome out. Computes one relatedness matrix when leaving out each chromosome. Extra return dimension: (n_chrom)
			'ltco':	Leave two chromosome out. Computes one relatedness matrix when leaving out each chromosome pair. Extra return dimension: (n_chrom,n_chrom). Diagonals are the same as 'loco'.
	norm:	Whether to normalize dg.
	Return:	numpy.array(shape=(...,n_donor,n_donor)) Estimated relatedness matrix. See parameter 'method' for preceeding dimensions.
	"""
	import numpy as np
	if method not in {'raw','loco','ltco'}:
		raise ValueError('Unknown method {}.'.format(method))
	if norm:
		dg=(dg.T-dg.mean(axis=1)).T
	if method=='raw':
		return np.corrcoef(dg.T)
	import itertools
	if loc is None:
		raise ValueError('Location of each genotype is required when method is not "raw".')
	if loc.ndim!=2:
		raise ValueError('Location of each genotype must have 2 dimensions.')
	if loc.shape[1]!=dg.shape[0]:
		raise ValueError('Number of SNPs in genotype matrix and location matrix must match.')
	if loc.shape[0]<1:
		raise ValueError('Location of each genotype must have at least 1 row.')
	s=np.sort(np.unique(loc[0]))
	ns=len(s)
	if not (s==np.arange(1,ns+1)).all():
		raise ValueError('Chromosome IDs must start with 1 and be sequential (containing no missing chromosome). If not, rename your chromosomes.')
	if method=='loco':
		nd=1
	elif method=='ltco':
		nd=2
	if ns<nd+1:
		raise ValueError('Need at least {} chromosomes when method is "{}".'.format(nd+1,method))
	sit=range(ns)
	if method=='loco':
		sit=map(lambda x:(x,),sit)
	elif method=='ltco':
		sit=filter(lambda x,y:x<=y,itertools.product(*itertools.tee(sit)))
	ans=np.zeros(([ns]*nd)+([dg.shape[1]]*2),dtype=float)
	for xi in sit:
		t1=(loc[0].reshape(-1,1).repeat(nd,axis=1)!=xi).all(axis=1)
		ans1=estimate(dg[t1],method='raw',norm=False)
		if method=='loco':
			ans[xi[0]]=ans1
		elif method=='ltco':
			ans[xi[0],xi[1]]=ans1
			if xi[0]!=xi[1]:
				ans[xi[1],xi[0]]=ans1
	return ans

def eigen(mk, nc, tol=1E-8):
	"""Computes eigenvalues and eigenvectors for full (per sample) kinship matrix.
	Uses reduced (per donor) representation of kinship matrix as input.
	mk:		numpy.array(shape=(nd,nd)) Kinship matrix between donors
	nc:		numpy.array(shape=(nd,)) Number of cells/samples from each donor.
			Must match every row/column of mk.
	tol:	Tolerance level of small eigenvalues. When any eigenvalue is smaller than tol
			times the largest eigenvalue, it's regarded as 0 and therefore mk is low rank.

	Return:	(mkl,mku). Full kinship matrix mk_{full}=mku_{full}^T*mkl*mku_{full}
	mkl:	numpy.array(shape=(nk,)) Eigenvalues
	mku:	numpy.array(shape=(nk,nd)) Reduced eigenvectors
			Each full eigenvector x is:
			mku_{full}[x]=np.concatenate([np.repeat(y,(dd==y).sum()) for y in mku[x]])

	Dimensions: 
	nd:		Number of Donors
	"""
	import logging

	import numpy as np
	from scipy.linalg import svd
	if mk.ndim != 2:
		raise ValueError('kinship matrix must have 2 dimensions.')
	nd = mk.shape[0]
	if mk.shape[1] != nd:
		raise ValueError('kinship matrix must be a square matrix.')
	if nc.shape != (nd, ):
		raise ValueError(
			'ncell should have shape ({},) instead of {}. Check sizes of kinship matrix and ncell.'
			.format(nd, nc.shape))
	if (nc < 0).any():
		raise ValueError('ncell must be all positive integers.')
	if tol < 0 or tol >= 1:
		raise ValueError('tol must be between 0 and 1')

	t1 = svd((mk.T * np.sqrt(nc)).T * np.sqrt(nc))
	mkl = t1[1]
	mku = t1[2] / np.sqrt(nc)
	t2 = mkl.min()
	if t2 < 0:
		raise ValueError('Kinship matrix is not semi-positive definite.')
	t3 = tol * mkl.max()
	if t2 <= t3:
		t2 = mkl > t3
		logging.warning(
			'Kinship matrix is low rank. Removed {}/{} eigenvectors.'.format(
				mkl.size - t2.sum(), mkl.size))
		mkl = mkl[t2]
		mku = mku[t2]

	return (mkl, mku)


assert __name__ != "__main__"
