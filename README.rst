=========
Airqtl
=========
Airqtl is an efficient method to map expression quantitative trait loci (eQTLs) and infer causal gene regulatory networks (cGRNs) from population-scale single-cell studies. The core of airqtl is Array of Interleaved Repeats (AIR), an efficient data structure to store and process donor-level data in the cell-donor hierarchical setting. Airqtl offers over 8 orders of magnitude of acceleration of eQTL mapping with linear mixed models, arising from its superior time complexity and Graphic Processing Unit (GPU) utilization. 

Installation
=============
Airqtl is on `PyPI <https://pypi.org/project/airqtl>`_. To install airqtl, you should first install `Pytorch 2 <https://pytorch.org/get-started/locally/>`_. Then you can install airqtl with pip: ``pip install airqtl`` or from github: ``pip install git+https://github.com/grnlab/airqtl.git``. Make sure you have added airqtl's install path into PATH environment before using the command-line interface (See FAQ_). Airqtl's installation can take several minutes including installing dependencies.

Usage
=====
Airqtl provides both command-line and python interfaces. For starters, you can run airqtl by typing ``airqtl -h`` on command-line. Try our tutorial below and adapt it to your own dataset.

Tutorials
==========================
Currently we provide `one tutorial <docs/tutorials/randolph>`_ to map cell state-specific single-cell eQTLs and infer cGRNs from the Randolph et al dataset in `docs/tutorials`.

Issues
==========================
Pease raise an issue on `github <https://github.com/grnlab/airqtl/issues/new>`_.

References
==========================
* `"Airqtl dissects cell state-specific causal gene regulatory networks with efficient single-cell eQTL mapping" <https://www.biorxiv.org/content/10.1101/2025.01.15.633041>`_ (2025) by Lingfei Wang. bioRxiv 2025.01.15.633041.

FAQ
==========================
* **What does airqtl stand for**?
	Array of Interleaved Repeats for Quantitative Trait Loci

* **Why do I see this error:** ``AssertionError: Torch not compiled with CUDA enabled``?
  
  This is because you installed a CPU-only pytorch but tried to run it on GPU. You have several options:
  
  1. To run pytorch on **CPU**, set `device='cpu'` in `Snakefile.config` of the tutorial pipeline you use.
  2. To run pytorch on **GPU**, reinstall pytorch with GPU support at `Installation`_.

* **I installed airqtl but typing ``airqtl`` says 'command not found'**.
	See below.
	
* **How do I use a specific python version for airqtl's command-line interface**?
	You can always use the python command to run airqtl, such as ``python3 -m airqtl`` to replace command ``airqtl``. You can also use a specific path or version for python, such as ``python3.12 -m airqtl`` or ``/usr/bin/python3.12 -m airqtl``. Make sure you have installed airqtl for this python version.
	
* **Why is airqtl killed mid-run**?
	One possible reason is you don't have enough memory. You can try it on a machine with more memory such as on the cloud.

* **How should I deal with CUDA out of memory error**?
	Airqtl runs sceQTL mapping on batches of SNPs and genes. For example, you can set the batch size to 16 SNPs and 10000 genes for the ``airqtl eqtl association`` step by adding ``--bsx 16 --bsy 10000`` in ``params_association`` in the ``Snakefile.config`` file. The default batch size is 256 SNPs and all genes for sceQTL mapping. If your dataset allows, use a smaller batch size for SNPs but all genes because it is the most efficient solution that minimizes recomputing.