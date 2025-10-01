GTensor API
===========

The GTensor module for genomic tensor analysis. This module provides functionality for creating, manipulating, and analyzing genomic tensors, including loading datasets, applying transformations, and generating explanations for model components.

GTensors are hierarchical, multi-dimensional arrays designed to represent complex genomic data structures. They are sliceable along multiple dimensions, and support lazy loading for memory efficiency.

Use the Gtensor CLI tool to interact with and build GTensor datasets from the command line - the python API is mostly intended for analysis and visualization.

Below is an interactive example of using GTensors in a Jupyter notebook:

.. raw:: html
   :file: ../_static/gtensor_example.html

GTensor API Reference
---------------------

.. automodule:: mutopia.gtensor.gtensor
   :members:
   :undoc-members:
   :exclude-members: ComponentWrapper
   :show-inheritance:
