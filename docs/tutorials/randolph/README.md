# Cell state-specific single-cell eQTL mapping and cGRN inference

This is an [airqtl](https://github.com/grnlab/airqtl) tutorial to map single-cell expression quantitative trait loci (sceQTLs) and infer causal gene regulatory networks (cGRNs) at the cell state level of specificity. The Randolph et al [dataset](https://zenodo.org/records/4273999) from their [original study](https://www.science.org/doi/full/10.1126/science.abg0928) is used.

**This tutorial is being actively updated. Please check back often.**

## Running the tutorial
1. [Install airqtl](https://github.com/grnlab/airqtl#installation) and download this folder
2. (Optional) Customize pipeline configuration in `Snakefile.config`, especially the `device` parameter if you prefer to use a CPU or a different GPU. See [Understanding and customizing the tutorial](#Understanding-and-customizing-the-tutorial).
3. Run the pipeline with `snakemake -j 1` **twice** in shell. The first run will download the raw dataset from Zenodo. The second run will read in the cell states to map sceQTLs infer cGRNs for each cell state.
4. Check the sceQTL and cGRN [output files](#Output-files).

The whole run takes <1 day on a top-end Dell Alienware Aurora R16®, in which single-cell eQTL mapping takes ~10mins for each cell state. The download step can take longer if your internet is slow.

After a successful run of this tutorial, you can [repurpose it for your own dataset](#Repurposing-the-tutorial-pipeline-for-your-own-dataset).

## Understanding and customizing the tutorial
### Input files
Airqtl's sceQTL mapping pipeline accepts the following input files:
* dimd.txt.gz: Donor(/sample) names, one line each
* dimc.txt.gz: Cell names, one line each
* dime.txt.gz: Gene names, one line each
* dimg.txt.gz: Genotype names, one line each
* dg.tsv.gz: Genotype matrix, genotype x raw donor without row/column names
* dgmap.tsv.gz: Donor (or sample) to raw donor map, one line for each donor. If you have one sample per (raw) donor, it is just numbers 0, 1, 2, ... one per line. If you have multiple samples per (raw) donor like this tutorial, each line is the (raw) donor ID for each sample, allowing to store genotypes more efficiently by (raw) donor instead of sample. In airqtl, we call samples as donors and donors as raw donors.
* de.tsv.gz or de.mtx.gz: Expression read count matrix without row/column names, either gene x cell in tsv format or cell x gene in mtx format (CellRanger output). This tutorial uses tsv format.
* dd.tsv.gz: Donor ID for each cell, one line each
* dccc.tsv.gz: Continuous cell covariates, cell x covariate, with column names but not row names. Examples are low dimensional coordinates, mitochondrial RNA proportion.
* dccd.tsv.gz: Discrete cell covariates, cell x covariate, with column names but not row names. Examples are cell type, perturbation. Entries should contain discrete string values and one-hot encoding should not be used.
* dcdc.tsv.gz: Continuous donor covariates, donor x covariate, with column names but not row names. Examples are age, body weight.
* dcdd.tsv.gz: Discrete donor covariates, donor x covariate, with column names but not row names. Examples are sex, disease and treatment status, sample location and batch.
Covariates can be used either for removal during association tests or for subsetting cells into different contexts (only for discrete covariates). Entries should contain discrete string values and one-hot encoding should not be used.
* dmeta_g.tsv.gz: Metadata for genotypes, genotype x column, with column names but not row names. It should contain these columns in order: name (for genotype name matching dimg.txt.gz), chr (chromsome this genotype is on, such as 1, chr12, or X), start (genotype location). It should be numerically sorted by chr and then start.
* dmeta_e.tsv.gz: Metadata for genes, gene x column, with column names but not row names. It should contain these columns in order: name (for gene name matching dime.txt.gz), chr (chromsome this gene is on, such as 1, chr12, or X), start (transcription start site), stop (transcription termination site), and strand (+ or -). It should be numerically sorted by chr and then start.

All files are compressed with gzip. All files other than the bottom two metadata files should have their dimensions and ordering matching dimd.txt.gz, dimc.txt.gz, dime.txt.gz, and dimg.txt.gz. Input files of the pipeline are also defined in the `datasetfiles_data` and `datasetfiles_meta` variables in [airqtl.pipeline.dataset](../../../src/airqtl/pipeline/dataset.py).

For this tutorial, these input files are automatically downloaded into folder `data/raw`. When running airqtl on your own dataset, you should deposit these input files in the same place before running the pipeline.

### The pipeline
* Each step of the pipeline is defined as a rule sequentially in `Snakefile`. Take the sceQTL association as an example, it corresponds to i) the shell command `airqtl eqtl association` and ii) the python function `airqtl.pipline.eqtl.association`. Therefore, you can learn more from either the command `airqtl eqtl association -h` or the docstring of `airqtl.pipline.eqtl.association`. The output files and logs of each step are located in `data/x` and `log/x.log` respectively, where x is the name of the step/rule and can be either a folder or a file with name suffix. Some of the steps are run once for the whole dataset while some are run separately for each cell state.
* To change pipeline parameters, modify `Snakefile.config`. You can use custom command-line parameters of each step according to their accepted parameters such as those obtained from `airqtl eqtl association -h`.
* For standard use, you should not modify `Snakefile` which is based on [Snakemake](https://snakemake.readthedocs.io/en/stable/).
* For advanced use, such as to run the pipeline in parallel or on a cluster, modify `Snakefile` accordingly only if you have the expertise.

### Intermediate files
Intermediate files are store in specific subfolders inside folder `data`:
* In folders `subset/{subset}` (from command `airqtl eqtl subset`): Same as input files, but only for the cell and donor subsets selected based on the covariates in `Snakefile.config`.
* In folders `qc/{subset}` (from command `airqtl eqtl qc`): Same as `subset/{subset}`, but only after quality control that removes certain genotypes, genes, cells, and donors from the subset data above. If too few genotypes, genes, cells, or donors remain, expect empty files.

### Output files
**IMPORTANT**: Output file formats should be viewed alongside command documentation (e.g. `airqtl eqtl association --help`) and the paper to understand definitions and filters.

Output files are store in specific subfolders inside folder `data`:
* Files `association/{subset}/result.tsv.lz4` (from command `airqtl eqtl association`) contain eQTL mapping summary statitistics between genes and SNPs for each cell subset. It is a lz4-compressed file. Each row is one association test. Columns are:
  - Gene: Gene in the association test. Its gene expression, defined as normalized log mRNA proportion, is used for association testing.
  - SNP: SNP in the association test. Alternative allele count is used for association testing.
  - s0: Maximum likelihood estimator of sigma (standard deviation multiplier of gene expression) under the null hypothesis
  - l0: Maximum likelihood estimator of lambda under the null hypothesis. lambda/(1+lambda) is estimated heritability of gene expression from unexplained variance. Available only in linear mixed model.
  - b: Restricted maximum likelihood estimator for the SNP's effect size on gene expression under the alternative hypothesis
  - s: Restricted maximum likelihood estimator for sigma under the alternative hypothesis
  - r: Effective Pearson correlation between the SNP and gene expression. It is computed after transformation with genetic relationship matrix factors (only for linear mixed model) and removal of covariates, based on b and the variances of gene expression and SNP.
  - p: Two-sided P-value for b or r (equivalent)

* In folders `qvalue/{subset}` (from command `airqtl eqtl qvalue`): `cis.tsv.gz` and `trans.tsv.gz` contain Q values computed with Benjamini-Hochberg procedure separately for cis- and trans-eQTL candidates.

* File `mr.tsv.gz` (from command `airqtl cgrn mr`) contains the Mendelian randomization results for gene regulations using cis-eQTLs as instrumental variables for all cell subsets. Each row is a triplet SNP->cis-gene->trans-gene. Columns are mostly identical with those in folders `qvalue/{subset}`, except with suffices `_c` or `_t` indicating association summary statistics for cis- or trans-interaction, respectively. Additional columns are:
  - state: cell subset in which the association testing was performed
  - q_filtered_...: Recomputed Q values after removing unknown genes. q_filtered_c is the final cis-eQTL Q values used in MR step.
  - q_trans: Recomputed trans-eQTL Q values after removing SNPs not having a significant cis-gene. This column is the final trans-eQTL Q values used in MR step.

* File `merge.tsv.gz` (from command `airqtl cgrn merge`) contains the inferred cGRNs for each cell subset. It merges all available significant SNPs that support the same gene interaction cis-gene->trans-gene in the cell subset. Each row is a cis-gene->trans-gene interaction. Columns are mostly identical with those in file `mr.tsv.gz`, except with suffices `_(max/mean/median/min/std)` indicating the statistics used to merge the SNPs. Additional columns are:
  - effect_...: Estimated effect size in cis-gene->trans-gene as `b_t/b_c`
  - n: Number of SNPs merged
  - nuniq: Number of unique SNPs merged that differ in at least one donor

## Repurposing the tutorial pipeline for your own dataset
1. [Run this tutorial pipeline](#Running-the-tutorial) successfully
2. [Understand the format of input files](#Understanding-and-customizing-the-tutorial) in `data/raw` folder
3. Perform initial quality control of your own dataset
4. Download this tutorial folder to a new location on your computer
5. [Reformat your own dataset](#Preparing-input-files) into the accepted format and place the files in newly created `data/raw` folder
6. [Customize the pipeline](#Understanding-and-customizing-the-tutorial) as needed
7. [Run the pipeline](#Running-the-tutorial) for your own dataset
8. Check the [output files](#Output-files)

### Preparing input files
There are two ways to prepare your input files:
1. Check the downloaded files in `data/raw/` to understand their format. Then convert your own dataset into this format. The tutorial may contain files not [mentioned above](#Input-files). These are obsolete files no longer needed in the pipeline.
2. We built simple scripts `airqtl utils convert_anndata` and `airqtl utils convert_vcf` to generate these input files from h5ad and vcf files respectively. You can add `-h` option to read their usage. To generate dmeta_e.tsv.gz, you can use https://github.com/pinellolab/dictys/blob/master/src/dictys/scripts/helper/gene_gtf.sh to first generate a bed file (e.g. gene.bed) from the gtf file of your reference genome. To produce dmeta_e.tsv.gz from the bed file, you need to insert column names, remove prefix "chr" from values in the chr column, and rearrange column order. An example script for this step is: `( echo 'name	chr	start	stop	strand'; sed 's/^chr//' gene.bed | awk '{print $4"\t"$1"\t"$2"\t"$3"\t"$6}' ) | gzip > dmeta_e.tsv.gz`. In more complex designs such as having multiple samples for each donor, you may need to update some of the generated files manually based on the description above and this tutorial.

## Issues
If you encounter any error, you are suggested to first troubleshoot on your own. The main log is displayed in the terminal as stdout. The log for each step of the pipeline is saved separately in the `log/` folder.

If you cannot resolve the error or have any other question, please [check the FAQ](../../../#faq) or [raise an issue](../../../#issues).

If you applied any fix to the code or pipeline or updated airqtl, you are strongly suggested to start over from step 1 of [Running the tutorial](#Running-the-tutorial), unless you are experienced and know what you are doing.
