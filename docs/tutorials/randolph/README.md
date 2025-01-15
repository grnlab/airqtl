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
* Input files of the pipeline are described in the `datasetfiles_data` and `datasetfiles_meta` variables in [airqtl.pipeline.dataset](../../../src/airqtl/pipeline/dataset.py). Check the downloaded files in `data/raw/` to understand their format.
* Each step of the pipeline is defined as a rule sequentially in `Snakefile`. Take the sceQTL association as an example, it corresponds to i) the shell command `airqtl eqtl association` and ii) the python function `airqtl.pipline.eqtl.association`. Therefore, you can learn more from either the command `airqtl eqtl association -h` or the docstring of `airqtl.pipline.eqtl.association`. The output files and logs of each step are located in `data/x` and `log/x.log` respectively, where x is the name of the step/rule and can be either a folder or a file with name suffix. Some of the steps are run once for the whole dataset while some are run separately for each cell state.
* To change pipeline parameters, modify `Snakefile.config`. You can use custom command-line parameters of each step according to their accepted parameters such as those obtained from `airqtl eqtl association -h`.
* To run the tutorial pipeline in parallel or on a cluster, modify `Snakefile` which is based on [Snakemake](https://snakemake.readthedocs.io/en/stable/).

## Repurposing the tutorial pipeline for your own dataset
1. [Run this tutorial pipeline](#Running-the-tutorial) successfully
2. [Understand the format of input files](#Understanding-and-customizing-the-tutorial) in `data/raw` folder
3. Perform initial quality control of your own dataset
4. Download this tutorial folder to a new location on your computer
5. Reformat your own dataset into the accepted format and place the files in newly created `data/raw` folder
6. [Customize the pipeline](#Understanding-and-customizing-the-tutorial) as needed
7. [Run the pipeline](#Running-the-tutorial) for your own dataset
8. Check the output files

## Issues
If you encounter any error, you are suggested to first troubleshoot on your own. The error logs are located inside console output and the folder `log`.

If you cannot resolve the error or have any other question, please [check the FAQ](../../../#faq) or [raise an issue](../../../#issues).

If you applied any fix to the code or pipeline, you are strongly suggested to start over from step 1 of [Running the tutorial](#Running-the-tutorial), unless you are experienced and know what you are doing.
