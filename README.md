# dsbasics
Data Science Basics

## Python Packaging

* [Getting Started with Python Packaging](https://github.com/ChadFulton/tsa-notebooks/blob/master/building_python_modules.ipynb)
* [Python Packaging User Guide](https://packaging.python.org/)
* [StackOverflow](https://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2/14753678#14753678)
* Example project [pgmpy](https://github.com/pgmpy/pgmpy)


* [setuptools](https://setuptools.readthedocs.io/en/latest/)
  * [setuptools resources](https://packaging.python.org/key_projects/#setuptools)
  * [installing-requirements](https://packaging.python.org/tutorials/installing-packages/#installing-requirements)
  * [distributing-packages](https://packaging.python.org/guides/distributing-packages-using-setuptools/#distributing-packages)



    python -m pip install --upgrade pip setuptools wheel
    git clone https://github.com/cs224/dsbasics.git
    cd dsbasics
    pip install -e .


    python setup.py sdist
    python setup.py bdist_wheel
