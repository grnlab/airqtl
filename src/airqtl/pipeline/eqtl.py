#!/usr/bin/python3
# Copyright 2025, Lingfei Wang, Matthew Funk
#
# This file is part of airqtl.

"""
Single-cell eQTL mapping.
"""

from typing import Optional


def _logdims(d:dict,prefix:str)->None:
	import logging
	logging.info(prefix+', current dimensions: cell: {}, gene: {}, donor: {}, genotype: {}, raw donor: {}.'.format(*[len(d[x]) for x in ['dimc','dime','dimd','dimg']],d['dg'].shape[1]))

def subset(diri:str,diro:str,covc:str,covd:str,vals:str,rmcov:bool=False)->None:
	"""
	Subset dataset by discrete covariates. Saves subset dataset (only select variables) in output directory.

	Parameters
	------------
	diri:
		Path of input directory of pre-subset dataset. Accepts de.tsv.gz or de.mtx.gz for expression read count matrix.
	diro:
		Path of output directory of post-subset dataset
	covc:
		List of discrete cell covariates used to subset cells, separated by ','. Each must be a column in dccd.tsv.gz. Set to '' if none used.
	covd:
		List of discrete donor covariates used to subset cells, separated by ','. Each must be a column in dcdd.tsv.gz. Set to '' if none used.
	vals:
		One allowed value for each discrete cell or donor covariate separated by ',' as specified in covc or covd. Must have matching length and order as covc and covd concatenated.
	rmcov:
		Whether to remove unmentioned covariates.
	"""

	import numpy as np
	import scipy.sparse

	from . import dataset as plib

	#Preprocess arguments
	covc,covd=[list(filter(lambda y:len(y)>0,x.split(','))) for x in [covc,covd]]
	vals=vals.split(',')
	assert len(vals)==len(covc)+len(covd),'Number of values must match number of covariates.'
	assert all(len(x)>0 for x in vals),'Each covariate must have at least one value.'

	#Load dataset
	d=plib.load_dataset(diri,noslim=True)
	assert all(x in d['dccd'].columns for x in covc),'At least one discrete cell covariate not found in dataset.'
	assert d['dccd'][covc].isna().sum().sum()==0,'At least one discrete cell covariate contains missing values.'
	assert all(x in d['dcdd'].columns for x in covd),'At least one discrete donor covariate not found in dataset.'
	assert d['dcdd'][covd].isna().sum().sum()==0,'At least one discrete donor covariate contains missing values.'

	#Determine subset
	covs=[d['dccd'][x].values for x in covc]+[d['dcdd'][x].values[d['dd']] for x in covd]
	nc=len(covs)
	assert len(vals)==nc
	vals=[int(x) if np.issubdtype(y.dtype,np.integer) else (float(x) if np.issubdtype(y.dtype,np.floating) else x) for x,y in zip(vals,covs)]
	coveq=np.prod([x==y for x,y in zip(covs,vals)],axis=0).astype(bool)
	if not coveq.any():
		#No cell, return empty dataset
		d2={
			'dimc':d['dimc'][[]],
			'de':np.zeros((d['de'].shape[0],0),dtype=d['de'].dtype),
			'dd':d['dd'][[]],
			'dccc':d['dccc'].iloc[[]][[]],
			'dccd':d['dccd'].iloc[[]][[]],
			'dcdc':d['dcdc'].iloc[[]][[]],
			'dcdd':d['dcdd'].iloc[[]][[]],
		}
		plib.save_dataset(d2,diro,check_full='none')
		return

	#Remove selected covariates
	dccd=d['dccd'].iloc[coveq].drop(columns=np.array(covc))
	dcdd=d['dcdd'].drop(columns=np.array(covd))
	#Remove unmentioned covariates
	if rmcov:
		dccc=d['dccc'].iloc[coveq][[]]
		dccd=dccd.drop(columns=list(filter(lambda x:x not in covc,dccd.columns)))
		dcdd=dcdd.drop(columns=list(filter(lambda x:x not in covd,dcdd.columns)))
	else:
		dccc=d['dccc'].iloc[coveq]
	#Check and save dataset
	d2={
		'dimc':d['dimc'][coveq],
		'dd':d['dd'][coveq],
		'dccc':dccc,
		'dccd':dccd,
		'dcdd':dcdd,
		'dcdc':d['dcdc'],
	}
	if isinstance(d['de'],np.ndarray):
		d2['de']=d['de'][:,coveq]
	elif isinstance(d['de'],scipy.sparse.sparray):
		d2['de']=d['de'].tocsc()[:,coveq].toarray()
	else:
		raise ValueError(f'Unknown expression matrix type: {type(d["de"])}')
	
	if rmcov:
		d2['dcdc']=d2['dcdc'][[]]
	d.update(d2)
	plib.check_dataset(d)
	plib.save_dataset(d2,diro,check_full='none')

def qc(diri:str,diro:str,diri_raw:Optional[str]=None,cezp:int=100,cen:int=500,eezdn:int=0,eezdp:float=0.1,eezn:int=100,eezp:float=0.02,een:int=0,dcn:int=5,dgp:float=0,gp:float=0,gdp:float=0,gdn:int=0,scn:int=500,sdn:int=40,sgn:int=10000,sen:int=500,na:int=2)->None:
	"""
	Quality control on sceQTL mapping data.

	Parameters
	------------
	diri:
		Path of input directory of pre-QC dataset
	diro:
		Path of output directory of post-QC dataset
	diri_raw:
		Optional path of additional input directory of raw dataset. If specified, diri only needs to contain files specifying cell, donor, and covariate inclusion.
	cezp:
		Expressed (read count>0) gene count lower bound for cells
	cen:
		Read count lower bound for cells
	eezdn:
		Expressed donor count lower bound for genes
	eezdp:
		Expressed donor proportion lower bound for genes
	eezn:
		Expressed cell count lower bound for genes
	eezp:
		Expressed cell proportion lower bound for genes
	een:
		Read count lower bound for genes
	dcn:
		Cell count lower bound for donors
	dgp:
		Genotype missing rate upper bound for donors
	gp:
		Missing rate upper bound for genotypes. Genotype value -1 indicates missing.
	gdp:
		Minor allele frequency lower bound for genotypes
	gdn:
		Minor allele count lower bound for genotypes
	scn:
		Cell count lower bound for dataset
	sdn:
		Donor count lower bound for dataset
	sgn:
		Genotype count lower bound for dataset
	sen:
		Gene count lower bound for dataset
	na:
		Number of alleles. Only needed if gdp>0 or gdn>0.

	Method:
	------------
	1 Iteratively QC cells and genes.
	2 QC donors by dcn.
	3 QC genotypes.
	4 QC donors by dgp.
	5 Go to 1 if any donor is removed in steps 2 or 4.
	6 Filter raw donors.
	7 QC dataset.
	"""
	import logging

	import numpy as np

	from ..utils.numpy import groupby
	from . import dataset as plib
	assert all(x>=0 for x in [cezp,cen,eezdn,eezn,een,dcn,scn,sdn,sgn,sen,gdp,gdn]),'All parameters must be non-negative.'
	assert na>0,'Number of alleles must be positive.'
	assert all(0<=x<=1 for x in [eezdp,eezp,dgp,gp]),'All proportions must be in [0,1].'
	assert all(0<=x<=0.5 for x in [gdp]),'Minor allele frequency must be in [0,0.5].'

	#Load dataset
	try:
		if diri_raw is None:
			d=plib.load_dataset(diri)
		else:
			t1=['dimc','de','dd','dccc','dccd','dcdd']
			d=plib.load_dataset(None,select={diri_raw:list(set(x[0].split('.')[0] for x in plib.datasetfiles_data)-set(t1)),diri:t1},check=False)
	except plib.EmptyDatasetError:
		logging.info('Empty dataset. Writing empty file '+diro)
		plib.save_dataset(plib.empty_dataset(),diro)
		return

	assert (gdp==0 and gdn==0) or (d['dg'].max()<=na and d['dg'].min()>=-1),f'Genotype values must be in [-1,{na}] (-1 for missing)'

	#QC
	_logdims(d,'Input dataset')
	changed1=True
	n1=1
	while changed1 and all(len(d['dim'+x])>0 for x in 'cdeg') and d['dg'].shape[1]>0:
		changed2=True
		n2=1
		while changed2 and all(len(d['dim'+x])>0 for x in 'cdeg') and d['dg'].shape[1]>0:
			#1. Iteratively QC cells and genes
			selectc=((d['de']>0).sum(axis=0)>=cezp)&(d['de'].sum(axis=0)>=cen)
			donorgrp=groupby(d['dd'])
			selecte=np.array([(d['de'][:,donorgrp[x]]>0).any(axis=1) if x in donorgrp else np.zeros(d['de'].shape[0],dtype=bool) for x in range(max(donorgrp)+1)])
			selecte=(selecte.sum(axis=0)>=eezdn)&(selecte.mean(axis=0)>=eezdp)
			selecte&=((d['de']>0).sum(axis=1)>=eezn)&((d['de']>0).mean(axis=1)>=eezp)&(d['de'].sum(axis=1)>=een)
			changed2=(~selectc).any() or (~selecte).any()
			d=plib.filter_genes(plib.filter_cells(d,selectc,check=None),selecte,check=None)
			_logdims(d,f'Round {n1}, step 1.{n2}')
			n2+=1
		changed1=False
		#2. QC donors by dcn
		donorgrp=groupby(d['dd'])
		selectd=set(filter(lambda x:len(donorgrp[x])>=dcn,donorgrp))
		if len(selectd)!=len(donorgrp):
			changed1=True
			selectd=np.array([x in selectd for x in range(len(d['dimd']))])
			d=plib.filter_donors(d,selectd)
			_logdims(d,f'Round {n1}, step 2')
		#3. QC genotypes
		selectg=(d['dg']==-1)[:,d['dgmap']].mean(axis=1)<=gp
		if gdp>0 or gdn>0:
			ref_allele_count = np.clip(d['dg'],min=0).sum(axis=1)
			non_missing_count = (d['dg']>=0).sum(axis=1)*na
			if gdn>0:
				selectg&=(ref_allele_count>=gdn) & (ref_allele_count<=non_missing_count-gdn)
			if gdp>0:
				ref_allele_freq = ref_allele_count/non_missing_count
				selectg&=(ref_allele_freq>=gdp) & (ref_allele_freq<=1-gdp)
		if (~selectg).any():
			d=plib.filter_genotypes(d,selectg,check=None)
			_logdims(d,f'Round {n1}, step 3')
		#4. QC donors by dgp
		selectd=((d['dg']==-1).mean(axis=0)<=gp)[d['dgmap']]
		if (~selectd).any():
			changed1=True
			d=plib.filter_donors(d,selectd)
			_logdims(d,f'Round {n1}, step 4')
		n1+=1
	#6. Filter raw donors
	d=plib.filter_rawdonors(d,np.unique(d['dgmap']),check=None)
	_logdims(d,'Step 6')
	#7. QC dataset
	if len(d['dimc'])<scn or len(d['dimd'])<sdn or len(d['dimg'])<sgn or len(d['dime'])<sen:
		#Set to 0 size for too small datasets so pipeline will not proceed
		d=plib.filter_genotypes(plib.filter_genes(plib.filter_cells(plib.filter_rawdonors(plib.filter_donors(d,np.arange(0)) if len(d['dd'])>0 else d,np.arange(0)),np.arange(0)),np.arange(0)),np.arange(0))
		_logdims(d,'Step 7')

	#Postprocessing: order samples by donor
	t1=np.argsort(d['dd'])
	d['dd']=d['dd'][t1]
	d['de']=d['de'][:,t1]
	d['dccc']=d['dccc'].iloc[t1]
	d['dccd']=d['dccd'].iloc[t1]
	d['dimc']=d['dimc'][t1]
	# assert set(d['dd'])==set(range(nd)),f'Donor ID contains values not in [0,{nd})'

	#Remove donors with no cells
	assert (d['dd'][1:]>=d['dd'][:-1]).all(),'Donor ID not sorted.'
	ncs=np.searchsorted(d['dd'],np.arange(len(d['dimd'])+1))
	ncs=ncs[1:]-ncs[:-1]
	t1=ncs>0
	if not t1.all():
		d['dgmap']=d['dgmap'][t1]
		d['dimd']=d['dimd'][t1]
		d['dcdc']=d['dcdc'].iloc[t1]
		d['dcdd']=d['dcdd'].iloc[t1]
		t1=np.nonzero(t1)[0]
		t2=-np.ones(t1[-1]+1,dtype=int)
		t2[t1]=np.arange(len(t1))
		d['dd']=t2[d['dd']]

	#Remove single-valued covariates
	d['dccc'],d['dccd'],d['dcdc'],d['dcdd']=[x[x.columns[x.nunique(dropna=False)>1]] for x in [d['dccc'],d['dccd'],d['dcdc'],d['dcdd']]]

	#Save dataset
	plib.save_dataset(d,diro)

def association(diri_data:str,fo:str,diri_meta:Optional[str]=None,effect:str='linear',rand:str='d',cisbound:int=1000000,pcut:float=1E-4,device:str='cpu',h0:str='chi2',ndccc:Optional[str]=None,ndccd:Optional[str]=None,ndcdc:Optional[str]=None,ndcdd:Optional[str]=None,ngpc:int=0,fi_subset:Optional[str]=None,na:int=2,bsx:int=None,bsy:int=None)->None:
	"""
	SceQTL mapping.

	Alternative model:	y = a*c + b*x + e, e ~ MVN(0, s**2 * (I+l0*K)).
	Null model:			y = a0*c + e0, e0 ~ MVN(0, s0**2 * (I+l0*K)), i.e. b=0.

	See paper for method details.

	Parameters
	----------
	diri_data:
		Path of input directory for dataset after preprocessing (e.g. QC)
	fo:
		Path of output tsv.lz4 file for eQTL mapping result
	diri_meta:
		Path of input directory for metadata. Can be directory of raw data before preprocessing. Uses *diri_data* if unspecified.
	effect:
		SNP effect type to test. Accepts:
		* "linear":		Linear SNP effect.
		* "dominant":	Dominant SNP effect in the presence of linear effect. Not tested.
		* "linear-gxc":	Interaction between linear SNP and context effect.
	rand:
		Random effect terms used in linear mixed model. Accepts a single str containing 0, 1, or more of below separated by ',':
		* "d":	Donor identity matrix.
		* "g0":	Genetic relatedness matrix estimated using all SNPs.
		Examples: ""; "d"; "d,g0".
		Use "" to indicate linear models without random effects.
	cisbound:
		Maximum distance between SNP and gene to be considered as cis-eQTL. Cis-eQTLs are not subject to p-value filtering by `pcut` in the output file.
	pcut:
		Maximum p-value cutoff to filter trans-eQTL mapping results. Accepts:
		* >0:	Standard filtering. Trans-eQTLs with P<pcut are retained.
		* =0:	No trans-eQTL will be output.
		* <0:	No cis- and trans-eQTL will be output. Mainly used to debug or test running speed.
	device:
		Device to perform association on, such as "cpu" or "cuda:0". See PyTorch documentation for details.
	h0:	
		Null distribution used to test hypothesis and compute p-value. Accepts:
		* "chi2":	Chi-square distribution. Fast from pytorch but less accurate. Usually sufficient given sceQTL cell count.
		* "beta":	Beta distribution. Slow from scipy but more accurate.
	ndccc:
		Cell-level continuous covariate to use, as defined by column name in dccc.tsv.gz. Use "*" for all covariates.
	ndccd:
		Cell-level discrete covariate to use, as defined by column name in dccd.tsv.gz. Use "*" for all covariates.
	ndcdc:
		Donor-level continuous covariate to use, as defined by column name in dcdc.tsv.gz. Use "*" for all covariates.
	ndcdd:
		Donor-level discrete covariate to use, as defined by column name in dcdd.tsv.gz. Use "*" for all covariates.
	ngpc:
		Number of genotype principal components to use as covariates
	fi_subset:
		Path of input tsv file to restrict QTL mapping to a subset of SNP-gene pairs. Headerless file should contain two columns indicating SNPs and genes by name respectively. If not provided, all pairs are tested.
	na:
		Number of alleles
	bsx:
		Batch size for genotype
	bsy:
		Batch size for gene expression
	"""
	import logging
	from functools import partial
	from os import linesep

	import lz4.frame
	import normalisr.normalisr as norm
	import numpy as np
	import pandas as pd
	import scipy
	from sklearn.decomposition import TruncatedSVD

	from .. import cov, heritability, kinship
	from ..association import fmt1 as fmt
	from ..association import fmt1_header as fmt_header
	from ..utils.eqtl import compute_locs, find_cis
	from ..utils.numpy import groupby
	from . import dataset as plib
	from .dataset import load_dataset

	#Raw heritability cutoffs (min,max)
	heritability_cuts=[0,10]
	#Only use top genes/SNPs if specified. For debugging only.
	topg=tope=None
	# topg=tope=512
	# topg=8192
	# tope=None
	if diri_meta is None:
		diri_meta=diri_data
	assert cisbound>=0,'cisbound must nonnegative.'
	#Preprocess arguments
	ndccc,ndccd,ndcdc,ndcdd=[list(filter(lambda y:len(y)>0,x.split(','))) if x is not None else [] for x in [ndccc,ndccd,ndcdc,ndcdd]]
	if effect=='linear':
		from ..association import multi_linear as model
	elif effect=='dominant':
		from ..association import multi_dominant as model
	elif effect=='linear-gxc':
		from ..association import multi_gxc as model
	else:
		raise ValueError('Unknown effect to test.')
	rand=set(list(filter(lambda x:len(x)>0,rand.split('-'))))
	assert len(rand-{'d','g0'})==0,'Unknown random effect term(s) in rand.'
	if len(rand)>1:
		raise NotImplementedError('Multiple random effects not implemented.')
	if ngpc<0:
		raise ValueError('ngpc (Number of genotype principal components used as covariates) must be non-negative.')
	#Batch size for genotype and gene expression
	if (bsx is None)^(bsy is None):
		raise ValueError('bsx and bsy must be both specified or unspecified.')
	if bsx is None:
		if effect=='linear-gxc' and len(rand)>0:
			if fi_subset is None:
				bsx=64
				bsy=32768
			else:
				bsx=2048
				bsy=32
		else:
			bsx=256
			bsy=32768
		logging.info(f'Using default batch sizes: bsx={bsx}, bsy={bsy}.')

	#Load dataset
	try:
		d=load_dataset(diri_data,meta=diri_meta)
	except plib.EmptyDatasetError as e:
		if effect in {'linear','dominant'}:
			logging.info('Empty dataset. Writing empty file '+fo)
			with lz4.frame.open(fo,mode='wt') as f:
				f.write(fmt_header([None])+linesep)
			return
		raise NotImplementedError('Empty dataset not supported for effect "{}".'.format(effect)) from e
	ndccc,ndccd,ndcdc,ndcdd=[x if x!=['*'] else list(d['dc'+y].columns) for x,y in zip([ndccc,ndccd,ndcdc,ndcdd],'cc,cd,dc,dd'.split(','))]
	assert all(all(y in d[f'dc{x[0]}'].columns for y in x[1]) for x in {'cc':ndccc,'cd':ndccd,'dc':ndcdc,'dd':ndcdd}.items()),'At least one covariate not found in dataset.'

	#Generate genotype PCs as continuous donor covariates
	if ngpc>0:
		if ngpc>=d['dg'].shape[1]:
			raise ValueError('ngpc must be less than number of raw donors.')
		t2=d['dg'][:,d['dgmap']]
		t2=t2.T-t2.mean(axis=1)
		t2=t2/(t2.std(axis=0)+1E-300)
		t2=TruncatedSVD(n_components=ngpc).fit_transform(t2)
		assert t2.shape[1]==ngpc,'Number of genotype PCs does not match requested number.'
		assert np.isfinite(t2).all(),'Genotype PCs contain NaN or Inf.'
		t2-=t2.mean(axis=0)
		t2/=t2.std(axis=0)+1E-300
		t2=pd.DataFrame(t2,index=d['dcdc'].index,columns=[f'gpc_{x+1}' for x in range(ngpc)])
		assert len(t2)==len(d['dcdc']),f'Number of genotype PCs ({len(t2)}) does not match number of donors ({len(d["dcdc"])}).'
		d['dcdc']=pd.concat([d['dcdc'],t2],axis=1)
		assert np.isfinite(d['dcdc'].values).all(),'Continuous donor covariates contain NaN or Inf.'
		ndcdc+=t2.columns.tolist()

	#Load subset file
	if fi_subset is not None:
		subset=pd.read_csv(fi_subset,sep='\t',header=None,index_col=None)
		assert subset.shape[1]==2 and subset.shape[0]>0,'Subset file must have two columns and at least one row.'
	else:
		subset=None

	#Whether covariates are from the same group. Useful to avoid multiplication among those in the same group
	grp=[None]*4
	#Filter covariates
	d['dccc']=d['dccc'][ndccc]
	grp[0]=np.eye(len(ndccc)).astype(bool)
	d['dcdc']=d['dcdc'][ndcdc]
	grp[2]=np.eye(len(ndcdc)).astype(bool)
	#Convert discrete to one-hot covariates
	d['dccd'],grp[1]=cov.d2o(d['dccd'][ndccd])
	d['dcdd'],grp[3]=cov.d2o(d['dcdd'][ndcdd])
	#Convert donor to cell covariates
	for xi in 'cd':
		d[f'dcd{xi}2']=d[f'dcd{xi}'].values[d['dd']]

	#Run normalisr
	#Compute Bayesian logCPM and cellular summary covariates
	dt,_,_,dc0=norm.lcpm(d['de'])
	#Normalize covariates and add constant-1 covariate
	dc=np.concatenate([d['dcc'+x].values for x in 'cd']+[d[f'dcd{x}2'] for x in 'cd'],axis=1).T
	ncov0=dc.shape[0]
	dc=np.concatenate([dc,dc0],axis=0)
	grp.append(np.array([[1,0,0,1],[0,0,0,1],[0,0,0,1],[1,1,1,1]],dtype=bool))
	dc=norm.normcov(dc)
	#Compute variance normalization factors for each gene and each cell
	sf=norm.scaling_factor(d['de'])
	weight=norm.compute_var(dt,dc)
	#Normalize gene expression at mean and variance levels and covariates at variance level
	dt,dc=norm.normvar(dt,dc,weight,sf,cat=0)
	#Use subset for debugging
	if tope is not None:
		dt=dt[:tope]
		d['dime']=d['dime'][:tope]
	if topg is not None:
		d['dg']=d['dg'][:topg]
		d['dimg']=d['dimg'][:topg]

	#Run normalisr cohort part
	assert (d['dd'][1:]>=d['dd'][:-1]).all(),'Donor ID not sorted.'
	#Number of cells per donor
	ncs=np.r_[0,np.searchsorted(d['dd'],np.arange(1,d['dd'][-1]+1)),d['dd'].size]
	ncs=ncs[1:]-ncs[:-1]
	if len(rand)==0:
		s0,l0,mkl,mku=None,None,None,None
		select_e=np.ones(dt.shape[0],dtype=bool)
	elif rand=={'d'} or rand=={'g0'}:
		k=np.eye(len(ncs)) if rand=={'d'} else kinship.estimate(d['dg'][:,d['dgmap']])
		mkl,mku=kinship.eigen(k,ncs)
		s0,l0,a0,select_e=heritability.estimate(dt,dc,ncs,mkl,mku)
		select_e&=(l0/(1+l0)>=heritability_cuts[0])&(l0/(1+l0)<=heritability_cuts[1])
		assert select_e.sum()>1,'No donors with heritability in range.'
		s0,l0,a0,dt=[x[select_e] for x in [s0,l0,a0,dt]]
	else:
		raise NotImplementedError('Requested random effect term (combination) {} not implemented.'.format(rand))

	#Obtain SNP and gene locations
	t1=d['dmeta_e'].reindex(index=d['dime'][select_e])
	t1['start']=t1['start'].fillna(-1).astype(int)
	t1['stop']=t1['stop'].fillna(-1).astype(int)
	t1['strand']=t1['strand'].fillna('+')
	locs=compute_locs(d['dmeta_g'].reindex(index=d['dimg'],fill_value=-1),t1)
	#Logging the number of cis and trans SNP-gene pairs
	t1=find_cis(locs,cisbound,sizeonly=True)
	logging.info('Found {}/{} cis-/trans- SNP-gene pairs between {} SNPs and {} genes:'.format(t1,d['dimg'].shape[0]*d['dime'].shape[0]-t1,d['dimg'].shape[0],d['dime'].shape[0]))

	#Association testing
	dc0=dc
	ncs0=ncs
	dx=d['dg'][:,d['dgmap']]
	dy=dt
	dl0=l0
	ka1={'bsx':bsx,'bsy':bsy,'device':device,'h0':h0}
	ka2={'pcut':max(pcut,0),'cis':cisbound}
	if pcut<0:
		ka2['cis']=None
	fmt1=partial(fmt,locs,(d['dimg'][:dx.shape[0]],d['dime'][select_e][:dy.shape[0]]),**ka2)
	#Prepare subset
	if subset is not None:
		t1=[dict(zip(x,range(len(x)))) for x in [d['dimg'],d['dime'][select_e]]]
		t2=subset.shape[0]
		subset=np.array(list(filter(lambda x:all(x[y] in t1[y] for y in range(2)),subset)))
		logging.info('Number of SNP-gene pairs in subset after/before filtering: {}/{}'.format(subset.shape[0],t2))
		subset=np.array([[t1[x][y] for y in subset[:,x]] for x in range(2)])
		t1=groupby(subset[1])
		subset=[set(subset[0,t1[x]]) if x in t1 else set() for x in range(len(d['dime'][select_e]))]
	logging.info('Writing file '+fo)
	with lz4.frame.open(fo,mode='wt') as f:
		#Write header
		if effect in {'linear','dominant'}:
			f.write(fmt_header([None])+linesep)
			model(dx,dy,dc0,ncs0,mkl,mku,dl0,f,fmt1,subset=subset,**ka1)
		elif effect=='linear-gxc':
			#Normal covariates
			dc1=[dc0]
			#Determine covariate multiplications
			grp=scipy.linalg.block_diag(*grp)
			#Intercept never multiplies
			grp[-1]=True
			grp[:,-1]=True
			grp=np.array(np.nonzero(~grp))
			grp=grp[:,grp[0]>=grp[1]]
			dc1.append(dc0[grp[0]]*dc0[grp[1]])
			dc1=np.concatenate(dc1,axis=0)
			f.write(fmt_header(np.concatenate([d['dc'+x].columns for x in 'cc/cd/dc/dd'.split('/')]))+linesep)
			if na!=2:
				raise NotImplementedError('Number of alleles !=2 not implemented.')
			t1=np.sum([(dx==x).any(axis=1) for x in range(na+1)],axis=0)
			for xi in range(2,na+2):
				t2=np.nonzero(t1==xi)[0]
				if len(t2)==0:
					continue
				if subset is not None:
					t3=dict(zip(t2,range(len(t2))))
					t4=set(t2)
					tsubset=[set(t3[y] for y in x&t4) for x in subset]
				else:
					tsubset=None
				fmt1=partial(fmt,[locs[0][t2],locs[1]],(d['dimg'][:dx.shape[0]][t2],d['dime'][select_e][:dy.shape[0]]),**ka2)
				model(dx[t2],dy,dc0[:-1],dc1,ncs0,mkl,mku,dl0,ncov0,f,fmt1,dom=xi>2,subset=tsubset,**ka1)

def qvalue(diri_data:str,fi_result:str,fo_cis:str,fo_trans:str,qcut_cis:float=1,qcut_trans:float=1,isfull:bool=False,filter_trans:Optional[str]="has_cis",cisbound:int=1000000)->None:
	"""
	Computing q-values for sceQTL mapping with Benjamini Hochberg procedure.

	Parameters
	----------
	diri_data:
		Path of input directory for post-QC dataset
	fi_result:
		Path of input file for sceQTL mapping result
	fo_cis:
		Path of output file for cis associations with q-value 
	fo_trans:
		Path of output file for trans associations with q-value
	qcut_cis:
		Optional cutoff for q-value of cis-associations before output
	qcut_trans:
		Optional cutoff for q-value of trans-associations before output
	isfull:
		Whether trans p-value data was full (i.e. without p-value cutoff). Warning: this parameter and its related parameters are crucial for correct q-value calculation. If False, filter_trans must be set properly and qcut_trans might not be reached.
	filter_trans:
		Optional filters for trans-associations before q-value computation. Accepts any combination of below separated by ',':
		* "has_cis": The candidate gene/SNP has at least one cis-association candidate SNP/gene (within region set by `cisbound`). Required for correct trans-sceQTL q-value calculation if isfull is False.
	cisbound:
		Maximum distance between SNP and gene to be considered as cis
	"""
	import logging

	import lz4.frame
	import numpy as np
	import pandas as pd

	from ..utils.eqtl import compute_locs, find_cis
	from ..utils.qv import bh
	from . import dataset as plib

	assert qcut_cis>=0 and qcut_cis<=1,'Cis q-value cutoff must be in [0,1].'
	if qcut_cis==1:
		qcut_cis=None
	assert qcut_trans>=0 and qcut_trans<=1,'Trans q-value cutoff must be in [0,1].'
	if qcut_trans==1:
		qcut_trans=None
	filter_trans={} if filter_trans is None else set(list(filter(lambda x:len(x)>0,filter_trans.split(','))))
	assert len(filter_trans-{'has_cis'})==0,'Unknown filter(s) in filter_trans'
	if 'has_cis' not in filter_trans and not isfull:
		logging.warning('Filter "has_cis" not specified but is required for correct q-value calculation when isfull is False. Results may be incorrect.')
	with lz4.frame.open(fi_result,'r') as f:
		d1=pd.read_csv(f,sep='\t',index_col=None,header=0,na_values=[""],keep_default_na=False)
	try:
		d0=plib.load_dataset(diri_data,meta=diri_data,select=['dime','dimg','dmeta_e','dmeta_g'],check=False)
	except plib.EmptyDatasetError:
		logging.info('Empty dataset. Writing empty files '+fo_cis+' and '+fo_trans)
		d1=d1.iloc[[]].copy()
		d1['q']=np.nan
		d1.iloc[[]].to_csv(fo_cis,sep='\t',header=True,index=False)
		d1.iloc[[]].to_csv(fo_trans,sep='\t',header=True,index=False)
		return

	#Separate cis and trans relations
	t1=d0['dmeta_e'].reindex(index=d0['dime'])
	t1['chr']=t1['chr'].fillna(-1)
	t1['start']=t1['start'].fillna(-1).astype(int)
	t1['stop']=t1['stop'].fillna(-1).astype(int)
	t1['strand']=t1['strand'].fillna('+')
	t1=[d0['dmeta_g'].reindex(index=d0['dimg'],fill_value=-1),t1]
	del d0
	t2=[dict(zip(x.index,range(len(x.index)))) for x in t1]
	assert len(set(d1['SNP'].tolist())-set(t2[0]))==0
	assert len(set(d1['Gene'].tolist())-set(t2[1]))==0
	t2=list(zip([t2[0][x] for x in d1['SNP'].tolist()],[t2[1][x] for x in d1['Gene'].tolist()]))
	cis=find_cis(compute_locs(*t1),cisbound)
	cis=set(tuple(x) for x in cis[:2].T)
	cis=np.array([x in cis for x in t2])
	assert len(cis)==d1.shape[0]
	d1cis=d1.iloc[cis].copy()
	d1trans=d1.iloc[~cis].copy()
	del d1

	t1=[set(d1cis[x].tolist()) for x in ['SNP','Gene']]
	sizes=[len(x) for x in t1]
	if 'has_cis' in filter_trans:
		#Restrict trans to SNPs and genes with at least one cis candidate	
		t1=[d1trans[x].isin(t1[i]) for i,x in enumerate(['SNP','Gene'])]
		d1trans=d1trans[t1[0]&t1[1]].copy()

	#Compute q-values with BH procedure separately for cis and trans associations
	if len(d1cis)>0:
		d1cis['q']=bh(d1cis['p'].values)
	else:
		d1cis['q']=np.nan
	if len(d1trans)>0:
		ka={} if isfull else {'size':sizes[0]*sizes[1]-d1cis.shape[0]}
		d1trans['q']=bh(d1trans['p'].values,**ka)
	else:
		d1trans['q']=np.nan

	#Applying q-value cutoff
	d1cis_cut=d1cis[d1cis['q']<qcut_cis].copy() if qcut_cis is not None else d1cis
	d1trans_cut=d1trans[d1trans['q']<qcut_trans].copy() if qcut_trans is not None else d1trans

	#Output
	logging.info('Writing file '+fo_cis)
	d1cis_cut.to_csv(fo_cis,sep='\t',header=True,index=False)
	logging.info('Writing file '+fo_trans)
	d1trans_cut.to_csv(fo_trans,sep='\t',header=True,index=False)


subset._da=True
qc._da=True
association._da=True
qvalue._da=True

assert __name__ != "__main__", "This module is not meant to be run directly."
