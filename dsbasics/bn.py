
import locale
locale.setlocale(locale.LC_ALL, 'C')

import numpy as np, pandas as pd, sklearn.base, matplotlib.pyplot as plt
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import warnings
import numpy as np, scipy, scipy.stats as stats, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import pgmpy, libpgm
import datetime, time
import matplotlib.dates
import pytz
from dateutil import relativedelta
import timeit

import rpy2, rpy2.rinterface, rpy2.robjects, rpy2.robjects.packages, rpy2.robjects.lib, rpy2.robjects.lib.grid, \
    rpy2.robjects.lib.ggplot2, rpy2.robjects.pandas2ri, rpy2.interactive.process_revents, \
    rpy2.interactive, rpy2.robjects.lib.grdevices
# rpy2.interactive.process_revents.start()
rpy2.robjects.pandas2ri.activate()

# import R's "base" package
base = rpy2.robjects.packages.importr('base')
# import R's utility package
utils = rpy2.robjects.packages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('bnlearn', 'gRain')

# R vector of strings

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpy2.robjects.packages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(rpy2.robjects.StrVector(names_to_install))

grdevices   = rpy2.robjects.packages.importr('grDevices')
bnlearn     = rpy2.robjects.packages.importr('bnlearn')
gRain       = rpy2.robjects.packages.importr('gRain')

# https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest
# https://github.com/pgmpy/pgmpy/pull/857/commits/4f97930d56384e0fca66011cff69b7e74f05364d
# https://github.com/pgmpy/pgmpy/issues/856
