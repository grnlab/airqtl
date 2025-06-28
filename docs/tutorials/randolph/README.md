# Cell state-specific single-cell eQTL mapping and cGRN inference

This is an [airqtl](https://github.com/grnlab/airqtl) tutorial to map single-cell expression quantitative trait loci (sceQTLs) and infer causal gene regulatory networks (cGRNs) at the cell state level of specificity. The Randolph et al [dataset](https://zenodo.org/records/4273999) from their [original study](https://www.science.org/doi/full/10.1126/science.abg0928) is used.

**This tutorial is being actively updated. Please check back often.**

## Running the tutorial
1. [Install airqtl](https://github.com/grnlab/airqtl#installation) and download this folder
2. (Optional) Customize pipeline configuration in `Snakefile.config`, especially the `device` parameter if you prefer to use a CPU or a different GPU. See [Understanding and customizing the tutorial](#Understanding-and-customizing-the-tutorial).
3. Run the pipeline with `snakemake -j 1` **twice** in shell. The first run will download the raw dataset from Zenodo. The second run will read in the cell states to map sceQTLs infer cGRNs for each cell state.
4. Check the sceQTL output files at `data/association` and cGRN output file at `data/merge.tsv.gz`.

The whole run takes ~1 day on a top-end Dell Alienware Aurora R16®, in which single-cell eQTL mapping takes ~10mins for each cell state. The download step can take longer if your internet is slow.

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
* dmeta_g.tsv.gz: Metadata for genotypes, genotype x column, with column names but not row names. It should contain these columns in order: name (for gene name matching dime.txt.gz), chr (chromsome this gene is on, such as 1 or Y), start (genotype location). It should be numerically sorted by chr and then start.
* dmeta_e.tsv.gz: Metadata for genes, gene x column, with column names but not row names. It should contain these columns in order: name (for gene name consistent with dime.txt.gz), chr (chromsome this gene is on, such as 1 or Y), start (transcription start site), stop (transcription termination site), and strand (+ or -). It should be numerically sorted by chr and then start.

All files are compressed with gzip. All files other than the bottom two metadata files should have their dimensions and ordering matching dimd.txt.gz, dimc.txt.gz, dime.txt.gz, and dimg.txt.gz. Input files of the pipeline are also defined in the `datasetfiles_data` and `datasetfiles_meta` variables in [airqtl.pipeline.dataset](../../../src/airqtl/pipeline/dataset.py).

### The pipeline
* Each step of the pipeline is defined as a rule sequentially in `Snakefile`. Take the sceQTL association as an example, it corresponds to i) the shell command `airqtl eqtl association` and ii) the python function `airqtl.pipline.eqtl.association`. Therefore, you can learn more from either the command `airqtl eqtl association -h` or the docstring of `airqtl.pipline.eqtl.association`. The output files and logs of each step are located in `data/x` and `log/x.log` respectively, where x is the name of the step/rule and can be either a folder or a file with name suffix. Some of the steps are run once for the whole dataset while some are run separately for each cell state.
* To change pipeline parameters, modify `Snakefile.config`. You can use custom command-line parameters of each step according to their accepted parameters such as those obtained from `airqtl eqtl association -h`.
* To run the tutorial pipeline in parallel or on a cluster, modify `Snakefile` which is based on [Snakemake](https://snakemake.readthedocs.io/en/stable/).

## Repurposing the tutorial pipeline for your own dataset
1. [Run this tutorial pipeline](#Running-the-tutorial) successfully
2. [Understand the format of input files](#Understanding-and-customizing-the-tutorial) in `data/raw` folder
3. Perform initial quality control of your own dataset
4. Download this tutorial folder to a new location on your computer
5. [Reformat your own dataset](#Preparing-input-files) into the accepted format and place the files in newly created `data/raw` folder
6. [Customize the pipeline](#Understanding-and-customizing-the-tutorial) as needed
7. [Run the pipeline](#Running-the-tutorial) for your own dataset
8. Check the output files

### Preparing input files
There are two ways to prepare your input files:
1. Check the downloaded files in `data/raw/` to understand their format. Then convert your own dataset into this format. The tutorial may contain files not [mentioned above](#Input-files). These are obsolete files no longer needed in the pipeline.
2. We built simple scripts `airqtl utils convert_anndata` and `airqtl utils convert_vcf` to generate these input files from h5ad and vcf files respectively. You can add `-h` option to read their usage. You still need to generate dmeta_e.tsv.gz by yourself, such as from the gtf file of your reference genome. In more complex designs such as having multiple samples for each donor, you may need to update some of the generated files manually based on the description above and this tutorial.

## Issues
If you encounter any error, you are suggested to first troubleshoot on your own. The main log is displayed in the terminal as stdout. The log for each step of the pipeline is saved separately in the `log/` folder.

If you cannot resolve the error or have any other question, please [check the FAQ](../../../#faq) or [raise an issue](../../../#issues).

If you applied any fix to the code or pipeline, you are strongly suggested to start over from step 1 of [Running the tutorial](#Running-the-tutorial), unless you are experienced and know what you are doing.
