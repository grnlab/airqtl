#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

#Essential dataset files
datasetfiles_data=[
	#Donor names, one line each
	"dimd.txt.gz",
	#Cell names, one line each
	"dimc.txt.gz",
	#Gene names, one line each
	"dime.txt.gz",
	#Genotype names, one line each
	"dimg.txt.gz",
	#Genotype matrix, genotype x raw donor
	"dg.tsv.gz",
	#Expression matrix, gene x cell
	"de.tsv.gz",
	#Donor ID for each cell, one line each
	"dd.tsv.gz",
	#Continuous cell covariates, cell x covariate
	"dccc.tsv.gz",
	#Discrete cell covariates, cell x covariate
	"dccd.tsv.gz",
	#Continuous donor covariates, donor x covariate
	"dcdc.tsv.gz",
	#Discrete donor covariates, donor x covariate
	"dcdd.tsv.gz",
	#Donor to raw donor map, one line for each donor
	"dgmap.tsv.gz",
]
#Cell subsetting files
datasetfiles_subset=['dimc.txt.gz','de.tsv.gz','dd.tsv.gz','dccc.tsv.gz','dccd.tsv.gz','dcdd.tsv.gz']

#Metadata files for each dataset not directly used in pipeline
datasetfiles_meta=[
	#Metadata matrix
	"dmeta.tsv.gz",
	#Metadata JSON
	"meta.json",
	#Metadata for genotypes
	"dmeta_g.tsv.gz",
	#Metadata for genes
	"dmeta_e.tsv.gz",
]
#Groundtruth dataset files
datasetfiles_truth=[
	#True expression proportion
	"te.tsv.gz",
	#True covariate effect size
	"ta.tsv.gz",
	#True genotype effect size
	"tb.mtx.gz",
	#True covariate*genotype effect size
	"tdb.mtx.gz",
	#True network
	"tnet.tsv.gz",
	#True total effect network
	"tnettot.tsv.gz",
]
assert len(set([x.split('.')[0] for x in datasetfiles_data+datasetfiles_meta+datasetfiles_truth]))==len(datasetfiles_data)+len(datasetfiles_meta)+len(datasetfiles_truth),f'Duplicate dataset file names found'
assert len(set(datasetfiles_subset)-set(datasetfiles_data))==0,f'Unknown dataset files: {set(datasetfiles_subset)-set(datasetfiles_data)}'

class EmptyDatasetError(ValueError):
	pass

def empty_dataset(meta=False):
	"""
	Generate empty dataset.
	meta:	Whether to include empty metadata
	"""
	import numpy as np
	import pandas as pd
	ans={
		'dimd':np.array([]),
		'dimc':np.array([]),
		'dime':np.array([]),
		'dimg':np.array([]),
		'de':np.array([]).reshape(0,0),
		'dg':np.array([]).reshape(0,0),
		'dd':np.array([]),
		'dccc':pd.DataFrame(np.array([]).reshape(0,0)),
		'dccd':pd.DataFrame(np.array([]).reshape(0,0)),
		'dcdc':pd.DataFrame(np.array([]).reshape(0,0)),
		'dcdd':pd.DataFrame(np.array([]).reshape(0,0)),
		'dgmap':np.array([])
	}
	if meta:
		ans.update({
			'dmeta':pd.DataFrame(np.array([]).reshape(0,0)),
			'meta':{},
			'dmeta_g':pd.DataFrame(np.array([]).reshape(0,0)),
			'dmeta_e':pd.DataFrame(np.array([]).reshape(0,0)),
		})
	return ans

def check_dataset(d,check_full='data'):
	#Check fullness
	tsetd,tsetm=[set(x.split('.')[0] for x in y) for y in [datasetfiles_data,datasetfiles_meta]]
	dset=set(d)
	assert len(dset-tsetd-tsetm)==0,f'Unknown dataset files: {dset-tsetd-tsetm}'
	if check_full=='data':
		assert len(tsetd-dset)==0,f'Missing dataset files: {tsetd-dset}'
	elif check_full=='data_only':
		assert len(tsetd-dset)==0 and len(dset-tsetd)==0,f'Missing/unknown dataset files: {(dset-tsetd)|(tsetd-dset)}'
	elif check_full=='meta':
		assert len(tsetm-dset)==0,f'Missing dataset files: {tsetm-dset}'
	elif check_full=='meta_only':
		assert len(tsetm-dset)==0 and len(dset-tsetm)==0,f'Missing/unknown dataset files: {(dset-tsetm)|(tsetm-dset)}'
	elif check_full=='all':
		assert len((tsetd|tsetm)-dset)==0,f'Missing dataset files: {(tsetd|tsetm)-dset}'
	elif check_full=='none':
		pass
	else:
		raise ValueError(f'Unknown check_full value: {check_full}')
	
	#Check individual variables
	nd,nc,ne,ng=[len(d['dim'+x]) if 'dim'+x in d else None for x in ['d','c','e','g']]
	if any(x==0 for x in [nd,nc,ne,ng]) or ('dg' in d and d['dg'].shape[1]==0):
		#Empty dataset
		return
	for xi in set(['dimd','dimc','dime','dimg'])&dset:
		assert len(d[xi])==len(set(d[xi])),f'Duplicate names found for {xi}'
		assert "" not in d[xi],f'Empty name found for {xi}'
	assert any(x not in d for x in ['de','dime','dimc']) or d['de'].shape==(ne,nc),f'Expression matrix shape mismatch: {d["de"].shape} vs {(ne,nc)}'
	assert any(x not in d for x in ['dg','dimg']) or d['dg'].shape[0]==ng,f'Genotype matrix shape[0] mismatch: {d["dg"].shape[0]} vs {ng}'
	assert any(x not in d for x in ['dd','dimc']) or d['dd'].shape==(nc,),f'Donor ID shape mismatch: {d["dd"].shape} vs {(nc,)}'
	assert 'dg' not in d or d['dg'].max()<=2,f'Genotype matrix contains values larger than 2'
	assert 'dg' not in d or d['dg'].min()>=-1,f'Genotype matrix contains values larger than 2'
	assert any(x not in d for x in ['dd','dimd']) or d['dd'].max()<nd,f'Donor ID contains values larger than {nd}'
	assert 'dd' not in d or d['dd'].min()>=0,f'Donor ID contains negative values'
	
	if 'dgmap' in d:
		assert 'dimd' not in d or d['dgmap'].shape==(nd,),f'Donor map shape mismatch: {d["dgmap"].shape} vs {(nd,)}'
		assert d['dgmap'].min()>=0,f'Donor map contains negative values'
		assert 'dg' not in d or d['dgmap'].max()<d['dg'].shape[1],f'Donor map contains values larger than {d["dg"].shape[1]}'
	if 'dimc' in d:
		for xi in set(['dccc','dccd'])&dset:
			assert len(d[xi])==nc,f'Number of cells in {xi}={d[xi].shape[0]} does not match {nc}'
	elif 'dccc' in d and 'dccd' in d:
		assert len(d['dccc'])==len(d['dccd']),f'Number of cells in dccc={d["dccc"].shape[0]} does not match dccd={d["dccd"].shape[0]}'
	if 'dimd' in d:
		for xi in set(['dcdc','dcdd'])&dset:
			assert len(d[xi])==nd,f'Number of donors in {xi}={d[xi].shape[0]} does not match {nd}'
	elif 'dcdc' in d and 'dcdd' in d:
		assert len(d['dcdc'])==len(d['dcdd']),f'Number of donors in dcdc={d["dcdc"].shape[0]} does not match dcdd={d["dcdd"].shape[0]}'
	assert any(x not in d for x in ['dmeta','dimc']) or len(set(d['dimc'])-set(d['dmeta'].index))==0

def load_dataset_slim(d,check=True,**ka):
	import numpy as np
	if 'dd' not in d:
		return d
	t1=(d['dd'].max()+1)
	if set(d['dd'])!=set(range(t1)):
		t1=-np.ones(t1,dtype=int)
		t2=sorted(np.unique(d['dd']))
		t1[t2]=np.arange(len(t2))
		d['dd']=t1[d['dd']]
		if 'dgmap' in d:
			d['dgmap']=d['dgmap'][t2]
		if 'dcdc' in d:
			d['dcdc']=d['dcdc'].iloc[t2]
		if 'dcdd' in d:
			d['dcdd']=d['dcdd'].iloc[t2]
		if 'dimd' in d:
			d['dimd']=d['dimd'][t2]
	if check:
		check_dataset(d,**ka)
	return d

def load_dataset(folder,meta=None,select=None,check=True,noslim=False,**ka):
	"""
	Load dataset from folder
	folder:	Folder to load dataset from
	meta:	Folder to load metadata from
	select:	Two possible types allowed:
			* List of variable names to load.
			* Dict of {folder:[variable names]} to load. Requires folder to be None
			* Defaults to all using list.
	check:	Whether to check dataset with check_dataset

	Return:	Dict of variables as in datasetfiles_data
			If meta is not None, dict also includes variables in datasetfiles_meta
	"""
	import gzip
	import itertools
	import json
	import logging
	from os.path import isfile
	from os.path import join as pjoin

	import numpy as np
	import pandas as pd
	from ..utils.numpy import smallest_dtype_int

	if isinstance(select,dict):
		assert folder is None,'folder must be None when select is dict'
		t1=list(itertools.chain.from_iterable(select.values()))
		assert len(t1)==len(set(t1))
		ka2=dict(ka)
		ka2['check_full']='none'
		ans={}
		for xi in select:
			ans.update(load_dataset(xi,meta=xi,select=select[xi],noslim=True,**ka2))
		ans=load_dataset_slim(ans,check=check,**ka)
		# if check:
		# 	check_dataset(ans,**ka)
		return ans
	if select is None:
		select=datasetfiles_data
		if meta is not None:
			select+=datasetfiles_meta
		select=[x.split('.')[0] for x in select]
	if not isinstance(select,list):
		raise ValueError(f'Unknown select type: {type(select)}')
	select=set(select)
	for f in set(datasetfiles_data)&select:
		assert isfile(pjoin(folder,f)),f'{f} not found'
	assert meta is None or len(set(datasetfiles_meta)&select)==0
	for f in set(datasetfiles_meta)&select:
		assert isfile(pjoin(meta,f)),f'{f} not found'
	tset=set(x.split('.')[0] for x in datasetfiles_data+datasetfiles_meta)
	assert len(select-tset)==0,f'Unknown variables: {select-tset}'

	logging.info(f'Loading dataset from folder {folder}')
	ans={}
	#Load dimensions
	for xi in set(['dimd','dimc','dime','dimg'])&select:
		fi=pjoin(folder,xi+'.txt.gz')
		logging.info(f'Reading file {fi}.')
		with gzip.open(fi,'rt') as f:
			ans1=np.array([x.strip() for x in f.readlines()])
		if ans1[-1]=="":
			ans1=ans1[:-1]
		if len(ans1)==0:
			raise EmptyDatasetError(f'Empty dataset with no {xi}')
		ans[xi]=ans1

	#Load expression, genotype matrices, donor ID and donor map
	for xi,dtype in [('de','u4'),('dg','i1'),('dd','u4'),('dgmap','u4')]:
		if xi not in select:
			continue
		fi=pjoin(folder,xi+'.tsv.gz')
		logging.info(f'Reading file {fi}.')
		with gzip.open(fi,'rt') as f:
			ans1=np.loadtxt(f,dtype=dtype,delimiter='\t')
		ans1=smallest_dtype_int(ans1)
		ans[xi]=ans1
	if 'dg' in ans and ans['dg'].shape[1]==0:
		raise EmptyDatasetError(f'Empty dataset with no raw donor')

	#Load covariates
	for xi in set(['dccc','dccd','dcdc','dcdd'])&select:
		fi=pjoin(folder,xi+'.tsv.gz')
		logging.info(f'Reading file {fi}.')
		try:
			ans[xi]=pd.read_csv(fi,sep='\t',index_col=None,header=0)
		except pd.errors.EmptyDataError:
			ans[xi]=pd.DataFrame(np.array([]).reshape(len(ans['dimc']) if xi[2]=='c' else len(ans['dimd']),0))

	#Load metadata
	if 'dmeta' in select:
		fi=pjoin(meta,'dmeta.tsv.gz')
		logging.info(f'Reading file {fi}.')
		ans['dmeta']=pd.read_csv(fi,sep='\t',index_col=0,header=0)
	if 'meta' in select:
		fi=pjoin(meta,'meta.json')
		logging.info(f'Reading file {fi}.')
		with open(fi,'r') as f:
			ans['meta']=json.load(f)
	if 'dmeta_g' in select:
		fi=pjoin(meta,'dmeta_g.tsv.gz')
		logging.info(f'Reading file {fi}.')
		ans['dmeta_g']=pd.read_csv(fi,sep='\t',index_col=0,header=0)
	if 'dmeta_e' in select:
		fi=pjoin(meta,'dmeta_e.tsv.gz')
		logging.info(f'Reading file {fi}.')
		ans['dmeta_e']=pd.read_csv(fi,sep='\t',index_col=0,header=0)
	if check:
		check_dataset(ans,**ka)
	
	#Remove donors without cells
	if not noslim:
		ans=load_dataset_slim(ans,check=check,**ka)

	return ans

def save_dataset(d,folder,foldermeta=None,check=True,**ka):
	"""
	Save dataset to folder
	d:			Dataset
	folder:		Folder to save dataset to
	foldermeta:	Folder to save metadata to
	check:		Whether to check dataset with check_dataset
	ka:			Keyword arguments to pass to check_dataset
	"""
	import gzip
	import json
	import logging
	from os import linesep
	from os.path import join as pjoin

	import numpy as np

	if any(x.split('.')[0] in d for x in datasetfiles_meta):
		assert foldermeta is not None, 'foldermeta must be specified if any meta data is not None'

	#Validations
	if check:
		check_dataset(d,**ka)

	#Save
	logging.info(f'Saving dataset to folder {folder}')
	for xi in set(['dimd','dimc','dime','dimg'])&set(d):
		fo=pjoin(folder,f'{xi}.txt.gz')
		logging.info(f'Writing file {fo}.')
		with gzip.open(fo,'wt') as f:
			f.write(linesep.join(d[xi])+linesep)
	for xi,fmt in list(filter(lambda x:x[0] in d,[('de','%u'),('dg','%i'),('dd','%u'),('dgmap','%i')])):
		fo=pjoin(folder,f'{xi}.tsv.gz')
		logging.info(f'Writing file {fo}.')
		with gzip.open(fo,'wt') as f:
			np.savetxt(f,d[xi],fmt=fmt,delimiter='\t')
	for xi in set(['dccc','dccd','dcdc','dcdd'])&set(d):
		fo=pjoin(folder,f'{xi}.tsv.gz')
		logging.info(f'Writing file {fo}.')
		d[xi].to_csv(fo,sep='\t',index=False)
	if 'dmeta' in d:
		fo=pjoin(foldermeta,'dmeta.tsv.gz')
		logging.info(f'Writing file {fo}.')
		d['dmeta'].to_csv(fo,sep='\t',index=True)
	if 'meta' in d:
		fo=pjoin(foldermeta,'meta.json')
		logging.info(f'Writing file {fo}.')
		with open(fo,'w') as f:
			json.dump(d['meta'],f)
	if 'dmeta_g' in d:
		fo=pjoin(foldermeta,'dmeta_g.tsv.gz')
		logging.info(f'Writing file {fo}.')
		d['dmeta_g'].to_csv(fo,sep='\t',index=True,header=True)
	if 'dmeta_e' in d:
		fo=pjoin(foldermeta,'dmeta_e.tsv.gz')
		logging.info(f'Writing file {fo}.')
		d['dmeta_e'].to_csv(fo,sep='\t',index=True,header=True)

def load_covs(folder):
	"""
	Load covariates
	folder:	Directory of dataset
	Return: [covariate names for dccc,dccd,dcdc,dcdc each as a list]
	"""
	import gzip
	from os.path import join as pjoin
	ans=[]
	for xi in ['dccc','dccd','dcdc','dcdd']:
		fi=pjoin(folder,f'{xi}.tsv.gz')
		logging.info(f'Reading file {fi}.')
		with gzip.open(fi,'rt') as f:
			ans.append(f.readline().strip().split('\t'))
	return ans

def check_truth(dt,d):
	import numpy as np
	varnames={'te','ta','tb','tdb','tnet','tnettot'}
	assert len(varnames-set(dt))==0,f'Unknown variables: {varnames-set(dt)}'
	assert len(set(dt)-varnames)==0,f'Missing variables: {set(dt)-varnames}'
	assert dt['te'].shape==d['de'].shape,f'Expression matrix shape mismatch: {dt["te"].shape} vs {d["de"].shape}'
	ncov=sum(d['dc'+x].shape[1] for x in 'cc,cd,dc,dd'.split(','))
	assert dt['ta'].shape==(ncov,d['de'].shape[0]),f'Covariate effect size matrix shape mismatch: {dt["ta"].shape} vs {(ncov,d["de"].shape[0])}'
	assert dt['tb'].shape==(d['dg'].shape[0],d['de'].shape[0]),f'SNP effect size matrix shape mismatch: {dt["tb"].shape} vs {(d["dg"].shape[0],d["de"].shape[0])}'
	assert dt['tdb'].shape==(ncov*d['dg'].shape[0],d['de'].shape[0]),f'Covariate*SNP effect size matrix shape mismatch: {dt["tdb"].shape} vs {(ncov*d["dg"].shape[0],d["de"].shape[0])}'
	assert dt['tnet'].shape==(d['de'].shape[0],d['de'].shape[0]),f'Network matrix shape mismatch: {dt["tnet"].shape} vs {(d["de"].shape[0],d["de"].shape[0])}'
	assert dt['tnettot'].shape==(d['de'].shape[0],d['de'].shape[0]),f'Total effect network matrix shape mismatch: {dt["tnettot"].shape} vs {(d["de"].shape[0],d["de"].shape[0])}'
	assert np.all(dt['te']>=0),f'Negative expression values found'

def load_truth(folder,data=None):
	import gzip
	from os.path import join as pjoin

	import numpy as np
	from scipy.io import mmread
	ans={}
	for xi in 'te/ta/tnet/tnettot'.split('/'):
		fi=pjoin(folder,f'{xi}.tsv.gz')
		logging.info(f'Reading file {fi}.')
		with gzip.open(fi,'r') as f:
			ans[xi]=np.loadtxt(f,delimiter='\t')
	if ans['ta'].ndim==1:
		ans['ta']=ans['ta'].reshape(1,-1)
	for xi in 'tb/tdb'.split('/'):
		fi=pjoin(folder,f'{xi}.mtx.gz')
		logging.info(f'Reading file {fi}.')
		with gzip.open(fi,'r') as f:
			ans[xi]=mmread(fi)
	if data is not None:
		check_truth(ans,data)
	return ans

def save_truth(dt,folder,data=None):
	import gzip
	from os.path import join as pjoin

	import numpy as np
	from scipy.io import mmwrite
	if data is not None:
		check_truth(dt,data)
	for xi in 'te/ta/tnet/tnettot'.split('/'):
		fo=pjoin(folder,f'{xi}.tsv.gz')
		logging.info(f'Writing file {fo}.')
		with gzip.open(fo,'w') as f:
			np.savetxt(f,dt[xi],delimiter='\t',fmt='%g')
	for xi in 'tb/tdb'.split('/'):
		fo=pjoin(folder,f'{xi}.mtx.gz')
		logging.info(f'Writing file {fo}.')
		with gzip.open(fo,'w') as f:
			mmwrite(f,dt[xi],symmetry='general')

def filter_any(d,select,arr1,arr2,df1,df2,check={}):
	d=dict(d)
	for xi in arr1:
		if xi in d:
			d[xi]=d[xi][select]
	for xi in arr2:
		if xi in d:
			d[xi]=d[xi][:,select]
	for xi in df1:
		if xi in d:
			d[xi]=d[xi].loc[select]
	for xi in df2:
		if xi in d:
			d[xi]=d[xi][d[xi].columns[select]]
	if check is not None:
		check_dataset(d,**check)
	return d

def filter_cells(d0,select,**ka):
	d=filter_any(d0,select,['dimc','dd'],['de'],['dccc','dccd'],[],check=None)
	if 'dmeta' in d and 'dimc' in d:
		d['dmeta']=d['dmeta'].loc[d['dimc']]
	if 'check' not in ka or ka['check'] is not None:
		check_dataset(d,**ka)
	return d

def filter_genes(d0,select,**ka):
	return filter_any(d0,select,['dime','de'],[],[],[],**ka)

def filter_donors(d0,select,**ka):
	import numpy as np
	d=filter_any(d0,select,['dimd','dgmap'],[],['dcdc','dcdd'],[],check=None)
	if 'dd' in d:
		selectid=select if np.issubdtype(select.dtype,np.integer) else np.nonzero(select)[0]
		selectdict=-np.ones(d['dd'].max()+1,dtype=int)
		selectdict[selectid]=np.arange(len(selectid))
		d['dd']=selectdict[d['dd']]
		if d['dd'].min()<0:
			d=filter_cells(d,d['dd']>=0)
	if 'check' not in ka or ka['check'] is not None:
		check_dataset(d,**ka)
	return d

def filter_rawdonors(d0,select,**ka):
	import numpy as np
	d=filter_any(d0,select,[],['dg'],[],[],check=None)
	if 'dgmap' in d and d['dgmap'].size>0:
		selectid=select if np.issubdtype(select.dtype,np.integer) else np.nonzero(select)[0]
		selectdict=-np.ones(d['dgmap'].max()+1,dtype=int)
		selectdict[selectid]=np.arange(len(selectid))
		d['dgmap']=selectdict[d['dgmap']]
		if d['dgmap'].min()<0:
			d=filter_genes(d,d['dgmap']>=0,check=None)
	if 'check' not in ka or ka['check'] is not None:
		check_dataset(d,**ka)
	return d

def filter_genotypes(d0,select,**ka):
	return filter_any(d0,select,['dimg','dg'],[],[],[],**ka)
