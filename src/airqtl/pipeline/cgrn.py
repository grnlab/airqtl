#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

"""
Causal GRN inference.
"""

from typing import Optional


def _mr_info(dc0,dt0):
	t1=set(dc0['Gene'].tolist())|set(dt0['Gene'].tolist())
	return '{} SNPs, {} genes, {} cis genes, {} trans genes, {} cis SNP-genes, {} trans SNP-genes'.format(len(dc0['SNP'].unique()),len(t1),len(dc0['Gene'].unique()),len(dt0['Gene'].unique()),len(dc0),len(dt0))

def mr(diri:str,fo:str,full:bool=False,qcis:float=0.1,qtrans:float=0.5,pcis:float=0.1)->None:
	"""
	Perform Mendelian randomization to infer causal GRNs for all cell states.

	Parameters
	----------
	diri:
		Path of input directory of cell state-specific sceQTL mapping results
	fo:
		Path of output file of causal GRN triplets as SNP->cis-gene->trans-gene
	full:
		Whether the trans-eQTL results are full without P-value cutoff
	qcis:
		Cutoff for Benjamini-Hochberg q value for cis-eQTLs to be regarded as significant
	qtrans:
		Cutoff for Benjamini-Hochberg q value for trans-eQTLs to be regarded as significant
	pcis:
		Cutoff for raw p value for cis-eQTLs to be regarded as highly insignificant
	"""
	import logging
	from glob import glob
	from os.path import join as pjoin

	import pandas as pd

	from ..utils.qv import bh

	if pcis<qcis:
		logging.warning('pcis should be larger than qcis. Otherwise some eQTL candidates may be both significant and highly insignificant.')

	#Find folders to load
	dirs=[glob(pjoin(diri,'*','cis.tsv.gz')),glob(pjoin(diri,'*','trans.tsv.gz'))]
	dirs=[set(x.split('/')[-2] for x in y) for y in dirs]
	dirs=sorted(list(dirs[0]&dirs[1]))

	ans=[]
	for dir1 in dirs:
		#Load data
		d=[]
		for xi in ['cis','trans']:
			fi=pjoin(diri,dir1,f'{xi}.tsv.gz')
			logging.info(f'Loading file {fi}')
			try:
				d.append(pd.read_csv(fi,sep='\t',header=0,index_col=None))
			except pd.errors.EmptyDataError:
				break
			d[-1]['type']=xi
		if len(d)<2:
			logging.info('Skipping {} due to missing cis- or trans-eQTL data'.format(dir1))
			continue
		d=pd.concat(d,axis=0)
		dc0=d[d['type']=='cis']
		dt0=d[d['type']=='trans']
		if len(dc0)==0 or len(dt0)==0:
			logging.info('Skipping {} due to no significant cis- or trans-eQTL detected.'.format(dir1))
			continue

		#Remove unknown and non-genes and recompute q
		qname='q_filtered'
		dc0=dc0.loc[dc0['Gene'].apply(lambda x:all(y not in x for y in '-._'))].copy()
		dt0=dt0.loc[dt0['Gene'].apply(lambda x:all(y not in x for y in '-._'))].copy()
		t1=[set(dc0['SNP'].tolist()),set(dt0['Gene'].tolist())]
		sizes=[len(x) for x in t1]
		dt0=dt0.loc[dt0['SNP'].isin(t1[0])].copy()
		dc0[qname]=bh(dc0['p'].values)
		ka={} if full else {'size':sizes[0]*sizes[1]-dc0.shape[0]}
		dt0[qname]=bh(dt0['p'].values,**ka)
		logging.info(str(_mr_info(dc0,dt0))+' after filtering of unknown genes')

		#Remove insignificant SNPs
		t1=dc0['SNP'][dc0[qname]<=qcis].unique()
		dc=dc0.loc[dc0['SNP'].isin(t1)].copy()
		logging.info(str(_mr_info(dc,dt0))+' after filtering of insignificant SNPs')
		
		#Remove pleiotropic SNPs
		t0=[]
		t1=dc.groupby('SNP').groups
		for xi in t1:
			t2=[(dc.loc[t1[xi],qname]<=qcis).sum(),(dc.loc[t1[xi],qname]<=pcis).sum(),len(t1[xi])]
			assert t2[1]>=t2[0]
			if t2[1]==1:
				t0.append(xi)
		t0=set(t0)
		dc=dc.loc[dc['SNP'].isin(t0)].copy()
		dt=dt0.loc[dt0['SNP'].isin(t0)].copy()
		logging.info(str(_mr_info(dc,dt))+' after filtering of pleiotropic SNPs')
		#Remove insignificant cis SNP-genes
		dc=dc.loc[dc[qname]<=qcis].copy()
		logging.info(str(_mr_info(dc,dt))+' after filtering of insignificant cis SNP-genes')
		
		#Recompute trans q
		qname2='q_trans'
		sizes[0]=len(dc['SNP'].unique())
		ka={} if full else {'size':sizes[0]*sizes[1]-dc0['SNP'].isin(dc['SNP']).sum()}
		dt[qname2]=bh(dt['p'],**ka)
		#Remove insignificant trans SNP-genes
		t1=dt[qname2]<qtrans
		if t1.all():
			logging.warning('All input trans-eQTL candidates have q values smaller than {} in cell state {}. Consider a larger cutoff for raw P values in sceQTL association step.'.format(qtrans,dir1))
		else:
			dt=dt.loc[dt[qname2]<qtrans]
		logging.info(str(_mr_info(dc,dt))+' after filtering of insignificant trans SNP-genes')
		
		#Full data
		d1=pd.merge(dc.rename({x:x+'_c' for x in dc.columns if x!='SNP'},axis=1),dt.rename({x:x+'_t' for x in dc.columns if x!='SNP'},axis=1),how='inner',on=['SNP'])
		d1['state']=dir1
		ans.append(d1)
	#Output to file
	ans=pd.concat(ans,axis=0)
	logging.info('Writing to file {}'.format(fo))
	ans.to_csv(fo,sep='\t',header=True,index=False)

mr._da=True

def _merge_relative_variability(d0,name):
	import numpy as np
	return np.abs((d0[f'{name}_max']-d0[f'{name}_min'])/(2*np.max(np.abs(d0[[f'{name}_max',f'{name}_min']]))))

def merge(fi:str,diri:str,fo:str,nmin:int=2,cvmin:float=0,svmax:float=1)->None:
	"""
	Merge SNP->cis-gene->trans-gene triplet relations to form cGRNs with optional filters.

	Parameters
	----------
	fi:
		Path of input file of SNP->cis-gene->trans-gene triplet relations
	diri:
		Path of input directory of cell state-specific sceQTL mapping data
	fo:
		Path of output file of causal GRNs
	nmin:
		Minimum number of unique significant SNPs required to support triplet relation. Set to 1 to disable.
	cvmin:
		Minimum relative variability of SNP->cis-gene effect size required among the significant SNPs supporting the triplet relation. Set to 0 to disable.
	svmax:
		Maximum relative variability of estimated cis-gene->trans-gene effect size required among the significant SNPs supporting the triplet relation. Set to 1 to disable.
	"""
	import logging
	from os.path import join as pjoin

	import numpy as np
	import pandas as pd

	from . import dataset as plib
	assert nmin>=1, 'nmin should be at least 1'
	assert cvmin>=0 and cvmin<=1, 'cvmin should be between 0 and 1'
	assert svmax>=0 and svmax<=1, 'svmax should be between 0 and 1'

	#Load data
	logging.info('Loading file {}'.format(fi))
	d00=pd.read_csv(fi,sep='\t',index_col=None,header=0)
	d00['effect']=d00['b_t']/(d00['b_c']+1E-300)

	#Choose columns
	cs=d00.columns
	cs2={'SNP','Gene_c','Gene_t','type_c','type_t','state'}
	cs2=list(filter(lambda x: x not in cs2,cs))

	ans=[]
	#Separate states
	states=list(d00['state'].unique())
	for state in states:
		dd=plib.load_dataset(pjoin(diri,state),select=['dg','dimg','dgmap'],check=False)
		dg,dimg,dgmap=[dd[x] for x in ['dg','dimg','dgmap']]
		del dd
		assert dg.shape[0]==len(dimg)
		assert dgmap.max()<dg.shape[1]
		dg=dg[:,dgmap]
		dimgdict=dict(zip(dimg,range(len(dimg))))

		d0=d00[d00['state']==state]
		t1=d0.groupby(['Gene_c','Gene_t']).groups
		for xi in t1:
			#Summarize data for each regulation
			d1=d0.loc[t1[xi]].copy()
			#Correct directionality for SNPs with opposite effect directionality
			t2=d1['b_c']<0
			d1.loc[t2,['b_c','r_c','b_t','r_t']]*=-1
			ans1=[
				d1[cs2].min().rename({x:x+'_min' for x in cs2}),
				d1[cs2].max().rename({x:x+'_max' for x in cs2}),
				d1[cs2].mean().rename({x:x+'_mean' for x in cs2}),
				d1[cs2].std().rename({x:x+'_std' for x in cs2}),
				d1[cs2].median().rename({x:x+'_median' for x in cs2}),
			]
			ans1=pd.concat(ans1,axis=0)
			ans1=ans1[sorted(ans1.index)]
			nuniq=len(np.unique(dg[[dimgdict[x] for x in d1['SNP'].tolist()]],axis=0))
			assert nuniq>0
			ans1=pd.concat([d1.iloc[0][['Gene_c','Gene_t','state']],pd.Series([len(d1)],index=['n']),pd.Series([nuniq],index=['nuniq']),ans1],axis=0)
			ans.append(ans1)
		del dg
	ans=pd.DataFrame(ans)

	if nmin>1:
		ans=ans[ans['nuniq']>=nmin]
	if cvmin>0:
		ans=ans[_merge_relative_variability(d0,'b_c')>=cvmin]
	if svmax<1:
		ans=ans[_merge_relative_variability(d0,'effect')<=svmax]
	
	logging.info('Writing to file {}'.format(fo))
	ans.to_csv(fo,sep='\t',header=True,index=False)

merge._da=True

assert __name__ != "__main__", "This module is not meant to be run directly."
