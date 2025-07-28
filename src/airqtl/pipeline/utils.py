#!/usr/bin/python3
# Copyright 2025, Matthew Funk and Lingfei Wang
#
# This file is part of airqtl.

"""
Utility functions for airqtl.
"""

def _convert_anndata_covar_rename(s):
	return s.replace(" ", "_").replace("/", "_").replace("\\", "_")

def _convert_anndata_categories_codes(covar,cells_selected):
	"""
	Convert categories and codes to a list of categories
	"""
	import numpy as np
	categories = np.array(covar['categories'][cells_selected].astype(str))
	codes = np.array(covar['codes'][cells_selected].astype(int))
	codemap = [categories[code] for code in codes]
	return codemap

def _convert_anndata_cell_to_donor(donor_array,covariate, cov_array):
	"""
	Map cell level covariates to donor level covariates
	"""
	donor_map = []
	for i in range(donor_array.max() + 1):
		locations = cov_array[donor_array == i]
		if not (locations[1:]==locations[0]).all():
			raise ValueError(f"Donor-level covariate {covariate} is not the same for all cells in donor {i}")
		donor_map.append(locations[0])
	return donor_map

def _convert_anndata_covar(h5ad,covariates:str,donor_array,cells_selected,nc,covar_type):
	"""
	Create covariate files
	covar_type: dccc, dccd, dcdc, or dcdd
	"""
	import h5py
	import numpy as np
	import pandas as pd

	if not isinstance(covariates,str):
		raise TypeError(f"Covariates must be a string, got {type(covariates)}")
	if covar_type not in ['dccc', 'dccd', 'dcdc', 'dcdd']:
		raise ValueError(f"Invalid covariate type: {covar_type}")

	covariates=list(filter(lambda x:len(x)>0,covariates.split(",")))
	n=len(covariates)
	if n==0:
		return pd.DataFrame(np.zeros((donor_array.max()+1 if covar_type in ['dccd', 'dcdd'] else nc,0)))

	cov_list = []
	for covariate in covariates:
		if covariate not in h5ad['obs'].keys():
			raise ValueError(f"Error: Covariate {covariate} not found in h5ad file")
		if isinstance(h5ad['obs'][covariate],h5py.Group):
			cov_array = _convert_anndata_categories_codes(h5ad['obs'][covariate],cells_selected)
		else:
			cov_array = h5ad['obs'][covariate][cells_selected].astype(str)
		cov_list.append(cov_array)
	#map through dd to get donor level specificity
	if covar_type in ['dccd', 'dcdd']:
		cov_list = [[_convert_anndata_covar_rename(c) for c in i] for i in cov_list]
	if covar_type in ['dcdc', 'dcdd']:
		cov_list = [_convert_anndata_cell_to_donor(donor_array,covariates[i], np.array(cov_list[i])) for i in range(n)]
	covariates=[_convert_anndata_covar_rename(c) for c in covariates]
	if len(covariates)!=len(set(covariates)):
		raise ValueError(f"Covariate names are not unique.")
	return pd.DataFrame(np.array(cov_list).T,columns=covariates)

def convert_anndata(fi:str,diro:str,donor:str,ncc:str='',ncd:str='',ndc:str='',ndd:str='',mtx:bool=False,subset_donors:bool=True)->None:
	"""
	Convert AnnData h5ad file to input format for airqtl.
	
	The user still needs to generate airqtl input files regarding genotypes and gene location metadata

	Parameters
	------------
	fi:
		Path of h5ad file
	diro:
		Path of output directory
	donor:
		Name of donor id column in obs of h5ad file
	ncc:
		Comma separated column names in obs of h5ad file to be used as cell-level continuous covariates for dccc.tsv.gz
	ncd:
		Comma separated column names in obs of h5ad file to be used as cell-level discrete covariates for dccd.tsv.gz
	ndc:
		Comma separated column names in obs of h5ad file to be used as donor-level continuous covariates for dcdc.tsv.gz
	ndd:
		Comma separated column names in obs of h5ad file to be used as donor-level discrete covariates for dcdd.tsv.gz
	mtx:
		Whether to convert the expression matrix to mtx format. If True, the expression matrix will be converted to mtx format. If False, the expression matrix will be converted to tsv format.
	subset_donors:
		Whether to subset the cells and donors to the donors in dimd.txt.gz if present. If True, the donors will be subset to the donors in the donor file. If False, the dimd.txt.gz file will be overwritten but this may lead to inconsistence with dg.tsv.gz.
	"""
	import gzip
	import logging
	from functools import reduce
	from operator import add
	from os import linesep
	from os.path import join as pjoin
	from os.path import exists

	import h5py
	import numpy as np
	import pandas as pd
	from scipy.io import mmwrite
	from scipy.sparse import csr_array,coo_array

	with h5py.File(fi, "r") as h5ad:
		if donor not in h5ad['obs']:
			raise ValueError(f"Donor id column {donor} not found in h5ad file")

		fo=pjoin(diro,"dimd.txt.gz")
		if isinstance(h5ad['obs'][donor],h5py.Group):
			donors_full = h5ad['obs'][donor]['categories'][:].astype(str)
		else:
			donors_full = sorted(list(set(h5ad['obs'][donor][:].astype(str))))
		if len(donors_full)!=len(set(donors_full)):
			raise ValueError("Donor ids are not unique")
		if exists(fo) and subset_donors:
			logging.info(f"Reading existing donor file {fo} to filter cells")
			with gzip.open(fo,"rt") as f:
				donors = [x.strip() for x in f.readlines()]
				donors=list(filter(lambda x:len(x)>0,donors))
			if len(donors)!=len(set(donors)):
				raise ValueError("Donor ids are not unique")
			donorsdict=dict(zip(donors,np.arange(len(donors))))
			donors_selected=np.array([x in donorsdict for x in donors_full])
			logging.info(f"Selected {donors_selected.sum()} donors out of {len(donors_selected)} total donors")			
		else:
			donors=donors_full
			donorsdict=dict(zip(donors,np.arange(len(donors))))
			donors_selected=np.ones(len(donors_full),dtype=bool)
			logging.info(f"Writing {fo}")
			with gzip.open(fo,"wt") as f:
				f.write(linesep.join(donors))

		fo=pjoin(diro,"dd.tsv.gz")
		if isinstance(h5ad['obs'][donor],h5py.Group):
			donor_array = h5ad['obs'][donor]['codes'].astype(int)
			cells_selected=np.isin(donor_array,np.nonzero(donors_selected)[0])
			donor_array=donor_array[cells_selected]
		else:
			# raise NotImplementedError("Donor id column is not a group")
			donor_array = h5ad['obs'][donor][:].astype(str)
			cells_selected=np.array([x in donorsdict for x in donor_array])
			donor_array=[donorsdict[x] for x in donor_array[cells_selected]]
		
		cellsdict=-np.ones(len(cells_selected),dtype=int)
		logging.info(f"Selected {cells_selected.sum()} cells out of {len(cells_selected)} total cells")
		cellsdict[cells_selected]=np.arange(cells_selected.sum())
		logging.info(f"Writing {fo}")
		with gzip.open(fo,"wt") as f:
			f.write(linesep.join([str(donorsdict[donors_full[x]]) for x in donor_array]))

		fo=pjoin(diro,"dimc.txt.gz")
		if '_index' not in h5ad['obs']:
			raise ValueError(f"Cell id column _index not found in h5ad file")
		cells = h5ad['obs']['_index'][cells_selected].astype(str)
		if len(cells)!=len(set(cells)):
			raise ValueError("Cell ids are not unique")
		logging.info(f"Writing {fo}")
		with gzip.open(fo,"wt") as f:
			f.write(linesep.join(cells))
			
		fo=pjoin(diro,"dime.txt.gz")
		if '_index' not in h5ad['var']:
			raise ValueError(f"Gene id column _index not found in h5ad file")
		genes = h5ad['var']['_index'][:].astype(str)
		if len(genes)!=len(set(genes)):
			raise ValueError('Gene names are not unique')
		logging.info(f"Writing {fo}")
		with gzip.open(fo,"wt") as f:
			f.write(linesep.join(genes))

		#de.tsv.gz or de.mtx.gz
		if any(x not in h5ad['X'] for x in ['data','indices','indptr']):
			raise ValueError(f"X matrix in h5ad file is not in CSR format")
		data = h5ad['X']['data'][:]
		indices = h5ad['X']['indices'][:]
		indptr = h5ad['X']['indptr'][:]
		e_array = csr_array((data, indices, indptr), shape=(len(cells_selected), len(genes))).astype(int).tocoo()
		t1=cellsdict[e_array.coords[0]]>=0
		e_array=coo_array((e_array.data[t1],(cellsdict[e_array.coords[0][t1]],e_array.coords[1][t1])),shape=(len(cells_selected),len(genes)))
		if mtx:
			fo=pjoin(diro,"de.mtx.gz")
			logging.info(f"Writing {fo}")
			with gzip.open(fo,"wb") as f:
				mmwrite(f,e_array,symmetry='general')
		else:
			fo=pjoin(diro,"de.tsv.gz")
			logging.info(f"Writing {fo}")
			pd.DataFrame(e_array.toarray().T).to_csv(fo,index=False,sep="\t",header=False)

		if any(x in ncc+ncd+ndc+ndd for x in [' ','/','\\']):
			logging.warning(f"Space, slash, or backslash found in covariate names. These will be replaced with underscore.")
		
		fo=pjoin(diro,"dccc.tsv.gz")
		dccc=_convert_anndata_covar(h5ad,ncc,donor_array,cells_selected,len(cells),"dccc")
		logging.info(f"Writing {fo}")
		dccc.to_csv(fo,index=False,sep="\t",header=True)

		fo=pjoin(diro,"dccd.tsv.gz")
		dccd=_convert_anndata_covar(h5ad,ncd,donor_array,cells_selected,len(cells),"dccd")
		logging.info(f"Writing {fo}")
		dccd.to_csv(fo,index=False,sep="\t",header=True)

		fo=pjoin(diro,"dcdc.tsv.gz")
		dcdc=_convert_anndata_covar(h5ad,ndc,donor_array,cells_selected,len(cells),"dcdc")
		logging.info(f"Writing {fo}")
		dcdc.to_csv(fo,index=False,sep="\t",header=True)

		fo=pjoin(diro,"dcdd.tsv.gz")
		dcdd=_convert_anndata_covar(h5ad,ndd,donor_array,cells_selected,len(cells),"dcdd")
		logging.info(f"Writing {fo}")
		dcdd.to_csv(fo,index=False,sep="\t",header=True)

		covs=reduce(add,[list(x.columns) for x in [dccc,dccd,dcdc,dcdd]])
		if len(covs)!=len(set(covs)):
			raise ValueError(f"Duplicate covariate names found in dccc.tsv.gz, dccd.tsv.gz, dcdc.tsv.gz, and dcdd.tsv.gz, possibly due to space, slash, or backslash in covariate names.")

		fo=pjoin(diro,"dgmap.tsv.gz")
		logging.info(f"Writing {fo}")
		pd.Series(np.arange(len(donors))).to_csv(fo,index=False,sep="\t",header=False)

def _convert_vcf_handle_line(line, fw, fw2, fw3,missing="remove",nheader=9,ndonors=0):
	from os import linesep
	if line.startswith("#"):
		return
	line = line.strip().split("\t")
	if len(line)<=nheader:
		raise ValueError(f"Line {line} has less than {nheader} columns")
	if ndonors>0 and len(line)!=nheader+ndonors:
		raise ValueError(f"Line {line} has {len(line)} columns, but {nheader+ndonors} columns are expected")
	genotype_list = []
	for i in range(nheader, len(line)):
		genotype = line[i].split(":")[0]
		if "." in genotype:
			if missing == "remove":
				return
			elif missing == "keep":
				genotype_list.append('-1')
			else:
				raise ValueError(f"Invalid missing genotype option: {missing}")
		elif genotype == ("0/0") or (genotype == "0|0"):
			genotype_list.append('0')
		elif (genotype == "0/1") or (genotype == "1/0") or (genotype == "0|1") or (genotype == "1|0"):
			genotype_list.append('1')
		elif genotype == "1/1" or genotype == "1|1":
			genotype_list.append('2')
	fw.write("\t".join(genotype_list) + linesep)
	meta_code = line[2]
	fw2.write(meta_code + linesep)
	fw3.write(meta_code + "\t" + line[0] + "\t" + line[1] + linesep)

def convert_vcf(fi: str, diro: str, missing: str = "remove") -> None:
	"""
	Convert vcf file to input format for airqtl, and create genotype metadata file.

	Parameters
	------------
	fi:
		Path to vcf file
	diro:
		Path to output directory
	missing:
		How to handle missing genotypes. Options are "remove" or "keep". If "remove", genotypes missing in any sample will be removed from the output. If "keep", missing genotypes will be kept as -1.
	"""
	import gzip
	import logging
	from os import linesep
	from os.path import join as pjoin
	assert missing in ["remove", "keep"], f"Invalid missing genotype option: {missing}"

	fopen=gzip.open if fi.endswith(".gz") else open

	fo1,fo2,fo3,fo4=[pjoin(diro,f"{x}.gz") for x in ["dg.tsv","dimg.txt","dmeta_g.tsv","dimd.txt"]]
	logging.info(f"Writing {fo1}")
	logging.info(f"Writing {fo2}")
	logging.info(f"Writing {fo3}")
	starting_lines=True
	with fopen(fi, "rt") as f,gzip.open(fo1, "wt") as fw,gzip.open(fo2, "wt") as fw2,gzip.open(fo3, "wt") as fw3:
		fw3.write('name\tchr\tstart'+linesep)
		for line in f:
			if line.startswith("#") and not starting_lines:
				raise ValueError(f"More header lines found after '#CHROM' in vcf file: {fi}")
			if line.startswith("##"):
				continue
			if line.startswith("#"):
				if not line.startswith("#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT"):
					raise ValueError(f"Unrecognized header line: {line}")
				starting_lines=False
				donors=line.strip().split("\t")[9:]
				ndonors=len(donors)
				assert ndonors>0, f"No donors found in vcf file: {fi}"
				assert ndonors==len(set(donors)), f"Donor ids are not unique in vcf file: {fi}"
				logging.info(f"Writing {fo4}")
				with gzip.open(fo4, "wt") as fw4:
					fw4.write(linesep.join(donors))
				continue
			_convert_vcf_handle_line(line, fw, fw2, fw3, missing=missing,nheader=9,ndonors=ndonors)


convert_anndata._da=True
convert_vcf._da=True

assert __name__ != "__main__", "This module is not meant to be run directly."
