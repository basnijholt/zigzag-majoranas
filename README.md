# Project on Enhanced proximity effect in zigzag-shaped Majorana Josephson junctions
By Tom Laeven, Bas Nijholt, Michael Wimmer, Anton R. Akhmerov

* [arXiv:1903.06168](https://arxiv.org/abs/1903.06168)
* [10.1103/PhysRevLett.125.086802](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1103%2FPhysRevLett.125.086802&v=ae1cde07)
* [Zenodo.org](https://zenodo.org/record/4075036)

# Files
This folder contains one Jupyter notebook and one Python files:
* [`paper-figures.ipynb`](paper-figures.ipynb)
* [`zigzag.py`](zigzag.py)

All of the paper's plots are generated with [`paper-figures.ipynb`](paper-figures.ipynb) using functions defined in [`zigzag.py`](zigzag.py).

The notebook contains instructions of how it can be used.

# Data
Download the data used in [`paper-figures.ipynb`](paper-figures.ipynb) at http://dx.doi.org/10.5281/zenodo.2578027 and place it in a directory called [`data/`](data). If the data are missing, use the code in the notebook to generate it.

# Installation
Install [miniconda](http://conda.pydata.org/miniconda.html) and then the Python 
environment that contains all dependencies with:

```
conda env create -f environment.yml
```

then activate it with
```
source activate zigzag
```

Finally, open a `jupyter-notebook` to open [`paper-figures.ipynb`](paper-figures.ipynb).
