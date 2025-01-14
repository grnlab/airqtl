# Cell state-specific single-cell eQTL mapping and cGRN inference

This is an [airqtl](https://github.com/grnlab/airqtl) tutorial to map expression quantitative trait loci (eQTLs) and infer causal gene regulatory networks (cGRNs) at the cell state level of specificity. The Randolph et al [dataset](https://zenodo.org/records/4273999) from their [original study](https://www.science.org/doi/full/10.1126/science.abg0928) is used.

To use this tutorial, first [install airqtl](https://github.com/grnlab/airqtl#installation) and download this folder. You may need to update `Snakefile.config`, especially the `device` parameter if you prefer to use a CPU or a different GPU. Then the pipeline can be run with `snakemake -j 1` in shell environment. It takes ~1 day on a top-end Dell Alienware Aurora R16®, in which single-cell eQTL mapping takes ~10mins for each cell state.

More documentation is underway to help you understand and customize this pipeline and repurpose it for your own data.

If you face any issues or need any assistance, see [FAQ](https://github.com/grnlab/airqtl#faq) and [Issues](https://github.com/grnlab/airqtl#issues).