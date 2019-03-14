# Project on Enhanced proximity effect in zigzag-shaped Majorana Josephson junctions
By Tom Laeven, Bas Nijholt, Michael Wimmer, Anton R. Akhmerov

# Files
This folder contains one Jupyter notebook and one Python files:
* `paper-figures.ipynb`
* `zigzag.py`

All of the paper's plots are generated with `paper-figures.ipynb` using functions defined in `zigzag.py`.

The notebook contains instructions of how it can be used.

# Data
Download the data used in `paper-figures.ipynb` at http://dx.doi.org/10.5281/zenodo.2578027 and place it in a directory called `data/`. If the data are missing, use the code in the notebook to generate it.

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

Finally, open a `jupyter-notebook` to open `paper-figures.ipynb`.
