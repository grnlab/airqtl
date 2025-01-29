#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

def prod(p1, dar, dbr, mdiag):
	"""Product of two matrices A,B using pre-computed values.
	But not containing sigma (variance) multiplier.
	p1: 	A^T*B
	dar,
	dbr:	A^T*Q1^T, B^T*Q1^T
	mdiag:	Diagonal matrix to be put in the middle
	"""
	import numpy as np
	ans = p1 - np.matmul(dar, (mdiag * dbr).T)
	return ans


def nll2(vx, mkl, ns):
	"""Analytical computation of negative log likelihood given the MLEs"""
	import numpy as np
	sx0, sx = vx[:2]

	n = ns.sum()
	beta = sx**2
	ans = np.log(np.abs(sx0)) * n + (np.log(1 + beta * mkl)).sum() / 2
	return float(ans)


def nll3_est(sx, mkl, dtr, dcr, dpt, dpc, dptc, ns):
	"""MLE estimator of other variables, given sx"""
	import numpy as np
	from normalisr.association import inv_rank

	beta = sx**2
	t0 = (1 - 1 / (1 + beta * mkl))
	pcc = prod(dpc, dcr, dcr, t0)
	ptc = prod(dptc, dcr, dtr, t0)
	ptt = prod(dpt, dtr, dtr, t0)

	assert np.isfinite(pcc).all()
	t1 = inv_rank(pcc)[0]
	xb = np.matmul(t1, ptc)
	assert xb.shape == (dcr.shape[0], )
	sx0 = ptt - 2 * np.dot(ptc, xb) + np.dot(xb, np.matmul(pcc, xb))
	sx0 = np.sqrt(sx0 / ns.sum())

	ans = np.concatenate([[sx0, sx], xb])
	assert ans.shape == (dcr.shape[0] + 2, )
	return ans


def nll3(vx, mkl, dtr, dcr, dpt, dpc, dptc, ns):
	"""Analytical computation of negative log likelihood with internal optimization for sx0 and xb
	Note results are negative due to 'minimize' in scipy."""
	import numpy as np

	# Compute MLEs
	t1 = nll3_est(vx.ravel()[0], mkl, dtr, dcr, dpt, dpc, dptc, ns)
	t2 = nll2(t1, mkl, ns)
	sx0, sx = t1[:2]
	xb = t1[2:]
	assert sx == vx.ravel()[0]
	beta = sx**2
	# Compute dL/dbeta
	t1 = dtr - np.matmul(xb, dcr)
	ans = np.dot(t1, t1 * mkl / ((1 + beta * mkl)**2))
	ans = (mkl / (1 + beta * mkl)).sum() / 2 - 0.5 * (sx0**(-2)) * ans
	# Compute dL/dsx
	ans *= 2 * sx
	ans = [t2, np.array([ans])]
	return ans


def estimate(dt, dc, ns, mkl, mku, tol=1E-6, onerror=0,**ka):
	"""Linear time estimator of heritability.
	Fit each of dt with non-Identity covariance matrix to Identity covariance
	with MLEs. Assume samples are sorted by donor.
	WARNING: assumes full-rank covariates.
	WARNING: constant 1 should be already included in the covariate dc as intercept.
	Null hypothesis:
		dt=alpha*dc+epsilon
		epsilon~N(0,sigma**2*(I+beta*K))
		K=any (positive definite) square-block matrix. Square block sizes match donor sample counts

	Arguments:
	dt:	numpy.array(shape=(nt,nsa)) Transcriptome. Each row is a gene and is estimated separately.
		Each column is a cell/sample, which should be ordered sequentially according to donor in ns.
		Can be the output from method gen.
	dc:	numpy.array(shape=(nc,nsa)) Covariates. Each row is a covariate.
		Each column is a cell/sample, which should be ordered sequentially according to donor in ns.
		Can be the output from method gen.
	ns:	numpy.array(shape=(nd,),dtype='uint') Number of cells from each donor.
	mkl,
	mku:Eigenvalue decomposition of kinship matrix K. Can be the output from method kinship_eigen.
	tol:Error tolerance level for scipy.optimize.minimize.
	onerror:		What to do when scipy.optimize.minimize fails.
		0:			Ignore and continue with heritability=0
		'raise':	Raise RuntimeError
	ka:	Args passed to scipy.optimize.minimize.

	Return: (sigma,beta,alpha,issuccess)
	sigma,
	beta:	np.array(shape=(nt,)) each. MLE of model above.
	alpha:	np.array(shape=(nt,nc)). MLE of model above.
	issuccess: np.array(shape=(nt,)). Whether estimation is successful for each gene.

	Dimensions:
	nt:	Number of genes
	nc:	Number of covariates
	nsa:Total number of cells/samples
	"""
	import logging

	import numpy as np
	from scipy.optimize import minimize
	if dt.ndim != 2:
		raise ValueError('Transcriptome must have 2 dimensions.')
	if dc.ndim != 2:
		raise ValueError('Covariates must have 2 dimensions.')
	if mku.ndim != 2:
		raise ValueError('Eigenvector matrix must have 2 dimensions.')
	if ns.ndim != 1:
		raise ValueError('ncell must have 1 dimension.')
	if mkl.ndim != 1:
		raise ValueError('Eigenvalues must have 1 dimension.')
	nt, nsa = dt.shape
	# Number of samples from each donor
	nsc = np.concatenate([[0], ns.cumsum()])
	nsa = nsc[-1]
	nn = len(ns)
	nc = dc.shape[0]
	if nsc[-1] != nsa:
		raise ValueError(
			'Unmatching number of cells/samples between transcriptome and ncell.')
	if dc.shape[1] != nsa:
		raise ValueError(
			'Unmatching number of cells/samples between transcriptome and covariates.'
		)
	if mkl.shape[0] != mku.shape[0]:
		raise ValueError('Unmatching number of eigenvalues and eigenvectors.')
	if mku.shape[1] != nn:
		raise ValueError('Unmatching number of donors in eigenvectors and ncell.')
	if (mkl <= 0).any():
		raise ValueError('Non-positive eigenvalue detected.')
	if tol <= 0:
		raise ValueError('Non-positive tol detected.')

	# dt & dc summed by donor [nx,nn]
	assert (ns>0).all()
	dts = np.array([dt[:, nsc[x]:nsc[x + 1]].sum(axis=1) for x in range(nn)]).T
	dcs = np.array([dc[:, nsc[x]:nsc[x + 1]].sum(axis=1) for x in range(nn)]).T
	# dt & dc reduced by kinship eigen vectors [nx,nn]
	dtr = np.matmul(dts, mku.T)
	dcr = np.matmul(dcs, mku.T)
	del dts, dcs
	# Precomputed naive products between dc [nc,nc]
	dpc = np.matmul(dc, dc.T)

	e_sx0s = []
	e_sxs = []
	e_xbs = []
	issuccess = []
	for xi in range(nt):
		# MLE
		if xi % 500 == 0:
			logging.debug('Fitted transcriptome: {}/{}'.format(xi, nt))
		dx = dt[xi]
		dxr = dtr[xi]

		# Naive products within dt
		dpt = np.dot(dx, dx)
		# Naive products between dt & dc
		dptc = np.matmul(dc, dx)

		try:
			ans4 = minimize(nll3,
							np.array([nn / nsc[-1]]),
							args=(mkl, dxr, dcr, dpt, dpc, dptc, ns),
							jac=True,
							options={
								'gtol': tol * nsa,
								'maxiter': 100},
							**ka)
			success=ans4.success
		except AssertionError:
			success=False
		if success:
			ans4=ans4.x[0]
		else:
			if onerror == 0:
				ans4=0
				success=True
			elif onerror == 'raise':
				print(ans4)
				raise RuntimeError('scipy.optimize.minimize failed for gene {}.'.format(xi))
			else:
				raise ValueError('Invalid onerror value.')
		issuccess.append(success)
		# Obtain MLEs
		t1 = nll3_est(ans4, mkl, dxr, dcr, dpt, dpc, dptc, ns)
		e_sx0, e_sx = t1[:2]
		e_xb = t1[2:]
		e_sx0 = np.abs(e_sx0)
		e_sx = np.abs(e_sx)
		e_sx0s.append(e_sx0)
		e_sxs.append(e_sx)
		e_xbs.append(e_xb)
	e_sx0s, e_sxs, e_xbs = [
		np.array(x, dtype=dt.dtype) for x in [e_sx0s, e_sxs, e_xbs]]
	# Recover original variance without removing covariates
	e_sx0s *= np.sqrt(nsa / (nsa - nc))
	issuccess = np.array(issuccess, dtype=bool)
	assert onerror!='raise' or issuccess.all()

	assert np.all([x.size == nt for x in [e_sx0s, e_sxs]])
	assert e_xbs.shape == (nt, nc)
	return (e_sx0s, e_sxs**2, e_xbs, issuccess)


assert __name__ != "__main__"
