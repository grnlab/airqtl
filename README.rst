=========
Airqtl
=========
Airqtl is an efficient method to map expression quantitative trait loci (eQTLs) and infer causal gene regulatory networks (cGRNs) from population-scale single-cell studies. The core of airqtl is Array of Interleaved Repeats (AIR), an efficient data structure to store and process donor-level data in the cell-donor hierarchical setting. Airqtl offers over 8 orders of magnitude of acceleration of eQTL mapping with linear mixed models, arising from its superior time complexity and Graphic Processing Unit (GPU) utilization. 

**This respository is being actively updated. Please check back later.**

Installation
=============
Airqtl is on `PyPI <https://pypi.org/project/airqtl>`_. To install airqtl, you should first install `Pytorch 2 <https://pytorch.org/get-started/locally/>`_. Then you can install airqtl with pip: ``pip install airqtl`` or from github: ``pip install git+https://github.com/grnlab/airqtl.git``. Make sure you have added airqtl's install path into PATH environment before using the command-line interface (See FAQ_). Airqtl's installation can take several minutes including installing dependencies.

Usage
=====
Airqtl provides command-line and python interfaces. For starters, you can run airqtl by typing ``airqtl -h`` on command-line. See our tutorials below.

Tutorials
==========================
Currently we provide one tutorial to map cell state-specific single-cell eQTLs and infer cGRNs from the Randolph et al dataset in `docs/tutorials`. We are working on better documentation so you can easily understand the tutorial and repurpose it for your own dataset.

Issues
==========================
Pease raise an issue on `github <https://github.com/grnlab/airqtl/issues/new>`_.

References
==========================
TBA

FAQ
==========================
* What does airqtl stand for?
	Array of Interleaved Repeats for Quantitative Trait Loci

* I installed airqtl but typing ``airqtl`` says 'command not found'.
	See below.
	
* How do I use a specific python version for airqtl's command-line interface?
	You can always use the python command to run airqtl, such as ``python3 -m airqtl`` to replace command ``airqtl``. You can also use a specific path or version for python, such as ``python3.12 -m airqtl`` or ``/usr/bin/python3.12 -m airqtl``. Make sure you have installed airqtl for this python version.
