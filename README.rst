Installation
===========

Start by making a new conda environment:

.. code-block:: bash

    $ conda create --name mutopia -c conda-forge -y python=3.11
    $ conda activate mutopia

Next, install the package from github:

.. code-block:: bash

    $ git clone https://github.com/AllenWLynch/Mutopia.git
    $ cd Mutopia
    $ pip install ".[docs]"

Finally, to view the docs, use:

.. code-block:: bash

    $ cd docs
    $ make html
    $ open build/html/index.html
