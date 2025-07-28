#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

from typing import Tuple, Union

from numpy.typing import NDArray


def find_cis(locs:list[NDArray],bound:Union[int,Tuple[int,int]],sizeonly:bool=False):
	"""
	Findss cis-relation as scipy.sparse.coo_array from a list of SNP and gene locations.
	locs:	[location of SNPs, location of genes]
			Each location has shape (n_x,3), where n_x is the number of SNPs or genes.
			Each column:
				0:	Chromosome
				1:	Start position (Transcription Start Site)
				2:	Stop position
	bound:	Maximum distance between SNP and gene to be considered cis. Either a scalar or (-distance before start position, distance after start position).
	sizeonly:	Only return the number of cis SNP-gene pairs.
	Return:	np.array((3,*),dtype=int) a sparse array between SNPs and genes for cis relation with each row representing values below:
			[0]:	SNPs as row index of the sparse array
			[1]:	genes as column index of the sparse array
			[2]:	distances as value of the sparse array (can be zero)
		or
		int: the number of cis SNP-gene pairs
	"""
	import numpy as np

	from ..utils.numpy import groupby
	
	if isinstance(bound,int):
		bound=(-bound,bound)
	ans=[0]
	g=[groupby(locs[x][:,0]) for x in range(2)]
	for xi in filter(lambda x:x!=-1,set(g[0])&set(g[1])):
		#Get distance
		ids=[g[x][xi] for x in range(2)]
		d=np.repeat(locs[0][ids[0]][:,[1]].astype(int),len(ids[1]),axis=1)-locs[1][ids[1],1].astype(int)
		d=d*np.sign(locs[1][ids[1],2]-locs[1][ids[1],1]).astype(int)
		#Find within distance bound
		t1=(d>=bound[0])&(d<=bound[1])
		if sizeonly:
			d=t1.sum()
		else:
			t1=np.nonzero(t1)
			d=[g[x][xi][t1[x]] for x in range(2)]+[d[t1[0],t1[1]]]
		ans.append(d)
	if sizeonly:
		return np.sum(ans)
	if len(ans)==1:
		return np.zeros((3,0),dtype=int)
	ans=np.array([np.concatenate(x) for x in zip(*ans[1:])])
	return ans

def compute_locs(dmeta_g,dmeta_e):
	"""
	Computes SNP and gene locations from dmeta_e and dmeta_g.
	dmeta_g:	Metadata for genotypes as pandas.DataFrame
	dmeta_e:	Metadata for genes as pandas.DataFrame
	Return:
	locs:		[location of SNPs, location of genes] as accepted by find_cis
				Each location has shape (n_x,3), where n_x is the number of SNPs or genes.
				Each column:
					0:	Chromosome
					1:	Start position (Transcription Start Site)
					2:	Stop position
	"""
	#Obtain SNP and gene locations
	dmeta_e=dmeta_e.copy()
	t1=(dmeta_e['strand']=='+')*dmeta_e['start']+(dmeta_e['strand']=='-')*dmeta_e['stop']
	t2=(dmeta_e['strand']=='+')*dmeta_e['stop']+(dmeta_e['strand']=='-')*dmeta_e['start']
	dmeta_e['start']=t1
	dmeta_e['stop']=t2
	t2=dmeta_g[['chr','start','start']].copy()
	t2['chr']=t2['chr'].fillna('MISSING')
	t2=t2.values.copy()
	t2[:,0]=[(x[3:].upper() if x.upper().startswith('CHR') else x.upper()) if hasattr(x,'upper') else x for x in t2[:,0]]
	t2[:,0]=[int(x) if hasattr(x,'isdigit') and x.isdigit() else x for x in t2[:,0]]
	locs=[t2]
	dmeta_e=dmeta_e[['chr','start','stop']].copy()
	dmeta_e['chr']=dmeta_e['chr'].fillna('MISSING')
	dmeta_e=dmeta_e.values.copy()
	dmeta_e[:,0]=[(x[3:].upper() if x.upper().startswith('CHR') else x.upper()) if hasattr(x,'upper') else x for x in dmeta_e[:,0]]
	dmeta_e[:,0]=[int(x) if hasattr(x,'isdigit') and x.isdigit() else x for x in dmeta_e[:,0]]
	dmeta_e[:,1:]=dmeta_e[:,1:].astype(int)
	locs.append(dmeta_e)
	return locs


assert __name__ != "__main__"
