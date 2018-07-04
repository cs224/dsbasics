

"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle et al 2012 [1]_

References
----------
.. [1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
"""
import numpy as np, pandas as pd, sklearn.base, matplotlib.pyplot as plt
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import sklearn.tree
import warnings, re


class FitnessFunc(object):
    """Base class for fitness functions

    Each fitness function class has the following:
    - fitness(...) : compute fitness function.
       Arguments accepted by fitness must be among [T_k, N_k, a_k, b_k, c_k]
    - prior(N, Ntot) : compute prior on N given a total number of points Ntot
    """
    def __init__(self, p0=0.05, neg_ln_gamma=None):
        self.p0 = p0
        self.neg_ln_gamma = neg_ln_gamma

    def validate_input(self, t, x, sigma):
        """Check that input is valid"""
        pass

    def fitness(**kwargs):
        raise NotImplementedError()

    def prior(self, Ntot):
        if self.neg_ln_gamma is None:
            return -(4 - np.log(73.53 * self.p0 * (Ntot ** -0.478)))
        else:
            return -self.neg_ln_gamma

    # the fitness_args property will return the list of arguments accepted by
    # the method fitness().  This allows more efficient computation below.
    @property
    def args(self):
        try:
            # Python 2
            return self.fitness.func_code.co_varnames[1:]
        except AttributeError:
            return self.fitness.__code__.co_varnames[1:]


class Events(FitnessFunc):
    """Fitness for binned or unbinned events

    Parameters
    ----------
    p0 : float
        False alarm probability, used to compute the prior on N
        (see eq. 21 of Scargle 2012).  Default prior is for p0 = 0.
    neg_ln_gamma : float or None
        If specified, then use this neg_ln_gamma to compute the general prior form,
        p ~ neg_ln_gamma^N.  If neg_ln_gamma is specified, p0 is ignored.
    """
    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(T_k))

class RegularEvents(FitnessFunc):
    """Fitness for regular events

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    neg_ln_gamma : float
        specifies the prior on the number of bins: p ~ neg_ln_gamma^N
    """
    def __init__(self, dt, p0=0.05, neg_ln_gamma=None):
        self.dt = dt
        self.p0 = p0
        self.neg_ln_gamma = neg_ln_gamma

    def validate_input(self, t, x, sigma):
        unique_x = np.unique(x)
        if list(unique_x) not in ([0], [1], [0, 1]):
            raise ValueError("Regular events must have only 0 and 1 in x")

    def fitness(self, T_k, N_k):
        # Eq. 75 of Scargle 2012
        M_k = T_k / self.dt
        N_over_M = N_k * 1. / M_k

        eps = 1E-8
        if np.any(N_over_M > 1 + eps):
            import warnings
            warnings.warn('regular events: N/M > 1.  '
                          'Is the time step correct?')

        one_m_NM = 1 - N_over_M
        N_over_M[N_over_M <= 0] = 1
        one_m_NM[one_m_NM <= 0] = 1

        return N_k * np.log(N_over_M) + (M_k - N_k) * np.log(one_m_NM)


class PointMeasures(FitnessFunc):
    """Fitness for point measures

    Parameters
    ----------
    neg_ln_gamma : float
        specifies the prior on the number of bins: p ~ neg_ln_gamma^N
        if neg_ln_gamma is not specified, then a prior based on simulations
        will be used (see sec 3.3 of Scargle 2012)
    """
    def __init__(self, p0=None, neg_ln_gamma=None):
        self.p0 = p0
        self.neg_ln_gamma = neg_ln_gamma

    def fitness(self, a_k, b_k):
        # eq. 41 from Scargle 2012
        return (b_k * b_k) / (4 * a_k)

    def prior(self, Ntot):
        if self.neg_ln_gamma is not None:
            return super(PointMeasures, self).prior(Ntot)
        elif self.p0 is not None:
            return super(PointMeasures, self).prior(Ntot)
        else:
            # eq. at end of sec 3.3 in Scargle 2012
            return 1.32 + 0.577 * np.log10(Ntot)


def bayesian_blocks(t, x=None, sigma=None, fitness='events', **kwargs):
    """Bayesian Blocks Implementation

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2012 [1]_

    Parameters
    ----------
    t : array_like
        data times (one dimensional, length N)
    x : array_like (optional)
        data values
    sigma : array_like or float (optional)
        data errors
    fitness : str or object
        the fitness function to use.
        If a string, the following options are supported:

        - 'events' : binned or unbinned event data
            extra arguments are `p0`, which gives the false alarm probability
            to compute the prior, or `neg_ln_gamma` which gives the slope of the
            prior on the number of bins.
        - 'regular_events' : non-overlapping events measured at multiples
            of a fundamental tick rate, `dt`, which must be specified as an
            additional argument.  The prior can be specified through `neg_ln_gamma`,
            which gives the slope of the prior on the number of bins.
        - 'measures' : fitness for a measured sequence with Gaussian errors
            The prior can be specified using `neg_ln_gamma`, which gives the slope
            of the prior on the number of bins.  If `neg_ln_gamma` is not specified,
            then a simulation-derived prior will be used.

        Alternatively, the fitness can be a user-specified object of
        type derived from the FitnessFunc class.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) bin edges

    Examples
    --------
    Event data:

    >>> t = np.random.normal(size=100)
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Regular event data:

    >>> dt = 0.01
    >>> t = dt * np.arange(1000)
    >>> x = np.zeros(len(t))
    >>> x[np.random.randint(0, len(t), len(t) / 10)] = 1
    >>> bins = bayesian_blocks(t, fitness='regular_events', dt=dt, neg_ln_gamma=0.9)

    Measured point data with errors:

    >>> t = 100 * np.random.random(100)
    >>> x = np.exp(-0.5 * (t - 50) ** 2)
    >>> sigma = 0.1
    >>> x_obs = np.random.normal(x, sigma)
    >>> bins = bayesian_blocks(t, fitness='measures')

    References
    ----------
    .. [1] Scargle, J `et al.` (2012)
           http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    See Also
    --------
    astroML.plotting.hist : histogram plotting function which can make use
                            of bayesian blocks.
    """
    # validate array input
    t = np.asarray(t, dtype=float)
    if x is not None:
        x = np.asarray(x)
    if sigma is not None:
        sigma = np.asarray(sigma)

    # verify the fitness function
    if fitness == 'events':
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        fitfunc = Events(**kwargs)
    elif fitness == 'regular_events':
        if x is not None and (np.any(x % 1 > 0) or np.any(x > 1)):
            raise ValueError("x must be 0 or 1 for fitness='regular_events'")
        fitfunc = RegularEvents(**kwargs)
    elif fitness == 'measures':
        if x is None:
            raise ValueError("x must be specified for fitness='measures'")
        fitfunc = PointMeasures(**kwargs)
    else:
        if not (hasattr(fitness, 'args') and
                hasattr(fitness, 'fitness') and
                hasattr(fitness, 'prior')):
            raise ValueError("fitness not understood")
        fitfunc = fitness

    # find unique values of t
    t = np.array(t, dtype=float)
    assert t.ndim == 1
    unq_t, unq_ind, unq_inv = np.unique(t, return_index=True,
                                        return_inverse=True)

    # if x is not specified, x will be counts at each time
    if x is None:
        if sigma is not None:
            raise ValueError("If sigma is specified, x must be specified")

        if len(unq_t) == len(t):
            x = np.ones_like(t)
        else:
            x = np.bincount(unq_inv)

        t = unq_t
        sigma = 1

    # if x is specified, then we need to sort t and x together
    else:
        x = np.asarray(x)

        if len(t) != len(x):
            raise ValueError("Size of t and x does not match")

        if len(unq_t) != len(t):
            raise ValueError("Repeated values in t not supported when "
                             "x is specified")
        t = unq_t
        x = x[unq_ind]

    # verify the given sigma value
    N = t.size
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape not in [(), (1,), (N,)]:
            raise ValueError('sigma does not match the shape of x')
    else:
        sigma = 1

    # validate the input
    fitfunc.validate_input(t, x, sigma)

    # compute values needed for computation, below
    if 'a_k' in fitfunc.args:
        ak_raw = np.ones_like(x) / sigma / sigma
    if 'b_k' in fitfunc.args:
        bk_raw = x / sigma / sigma
    if 'c_k' in fitfunc.args:
        ck_raw = x * x / sigma / sigma

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    ncp_prior = fitfunc.prior(N)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)
        kwds = {}

        # T_k: width/duration of each block
        if 'T_k' in fitfunc.args:
            kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]

        # N_k: number of elements in each block
        if 'N_k' in fitfunc.args:
            kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]

        # a_k: eq. 31
        if 'a_k' in fitfunc.args:
            kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]

        # b_k: eq. 32
        if 'b_k' in fitfunc.args:
            kwds['b_k'] = - np.cumsum(bk_raw[:R + 1][::-1])[::-1]

        # c_k: eq. 33
        if 'c_k' in fitfunc.args:
            kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = fitfunc.fitness(**kwds)

        A_R = fit_vec + ncp_prior
        A_R[1:] += best[:R]

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]

    #-----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]

# ./cross-validation-and-grid-search/2018-03-17-cv-a-gs.ipynb
# ./bayesian-networks-and-bayesialab/ames-housing.py
# [API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/pdf/1309.0238v1.pdf)
#
# * Estimators: fit() method. hyperparameters must be set as an instance variable (generally via a constructor parameter).
# * Transformers: some estimators can also transform a dataset; these are called transformers: transform() and fit_transform() method.
# * Predictors: some estimators can make predictions; these are called predictors: predict(), score() method.
#
# [Creating your own estimator in scikit-learn](http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/)



def bin_at_fixed_points_dataitem(bin_edges, data_item, bin_labels=None):
    if bin_labels is None:
        bin_labels = np.arange(len(bin_edges)+1)
    else:
        if (len(bin_edges) + 1) != len(bin_labels):
            raise RuntimeError("The number of labels needs to be one more than the bin boundaries")

    a1 = np.array(bin_edges, dtype=np.float)
    a2 = np.array([np.inf],dtype=np.float)
    bin_edges = np.concatenate((a1,a2))
    data_item = np.array(data_item,dtype=np.float)
    if len(data_item.shape) == 0:
        data_item = data_item.reshape(-1)
    outer_values = np.subtract.outer(data_item, bin_edges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = outer_values < 0.0

    idxs = np.argmax(values,axis=1)
    #idx = np.argmax(np.nonzero(values)[0][0],axis=1
    # idx_array = np.nonzero(values)[0]#[0]
    # if len(idx_array) == 0:
    #     idx = len(bin_labels) - 1
    # else:
    #     idx = idx_array[0]

    r = pd.Series(bin_labels)[idxs]#.reset_index(drop=True)#.values #
    r.loc[pd.isnull(data_item)] =  data_item[pd.isnull(data_item)] #np.nan

    return r.values

# bin_at_fixed_points_dataitem([75000, 150000, 225000, 300000], 215000, bin_labels=["75k","150k","225k","300k","high"])
# bin_at_fixed_points_dataitem([75000, 150000, 225000, 300000], [75000,215000,76000,-10,350000], bin_labels=["75k","150k","225k","300k","high"])
# bin_at_fixed_points_dataitem([75000, 150000, 225000, 300000], 75000, bin_labels=["75k","150k","225k","300k","high"])
# bin_at_fixed_points_dataitem([75000, 150000, 225000, 300000], -10, bin_labels=["75k","150k","225k","300k","high"])
# bin_at_fixed_points_dataitem([75000, 150000, 225000, 300000], 350000, bin_labels=["75k","150k","225k","300k","high"])


class BaseBinTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def plot(self, i=None):
        if i is not None:
            start = i
            end   = i + 1
        else:
            start = 0
            end   = self.X_.shape[1]

        y = end - start
        fig = plt.figure(figsize=(14, 7*y), dpi=180, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

        j = 1
        for i in range(start,end):
            ax = plt.subplot(y,2,j)

            # plot a standard histogram in the background, with alpha transparency
            x = self.X_[:,i]
            x = x[~pd.isnull(x)]

            ax.hist(x, bins=200, histtype='stepfilled', alpha=0.2, density=True, label='standard histogram')

            # bins = astroML.density_estimation.bayesian_blocks(x,p0=0.0001) # p0=p0
            bins = self.bins_[i]
            # plot an adaptive-width histogram on top
            ax.hist(x, bins=bins, color='black', histtype='step', density=True, label='blocks')

            ax.legend(prop=dict(size=12))
            ax.set_xlabel('t')
            ax.set_ylabel('P(t)')

            ax = plt.subplot(y,2,j + 1)
            ax.plot(np.linspace(0.0, 1.0, len(x)), np.sort(x));
            ax.bar(np.linspace(0.0, 1.0, len(x)), np.sort(x), 0.001);
            j += 2

    def transform(self, X):
        try:
            getattr(self, "bins_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        check_is_fitted(self, ['bins_'], all_or_any=all)

        # X = check_array(X)

        r = np.zeros(X.shape)
        for i, bin in enumerate(self.bins_):
            bin_boundaries = bin[1:-1]
            bin_labels = range(len(bin_boundaries) + 1)
            r[:,i] = bin_at_fixed_points_dataitem(bin_boundaries, X[:,i], bin_labels=bin_labels)

        return r


class BayesianBlocksBinTransformer(BaseBinTransformer):

    def __init__(self, p0=None, neg_ln_gamma=320):
        if p0 is not None:
            self.p0 = p0
            self.neg_ln_gamma = None
        else:
            self.p0 = None
            self.neg_ln_gamma = neg_ln_gamma


    def fit(self, X, y=None):
        # X = check_array(X)
        if len(X.shape) != 2:
            raise RuntimeError("X has invalid shape!")

        self.X_ = X

        bins_ = []
        for i in range(X.shape[1]):
            x = X[:, i]
            x = x[~pd.isnull(x)]
            if self.p0 is not None:
                bins_ += [bayesian_blocks(x, p0=self.p0)]
            else:
                bins_ += [bayesian_blocks(x, neg_ln_gamma=self.neg_ln_gamma)]
        self.bins_ = bins_
        # if len(X.shape) == 1: # one dimensional data
        #     self.bins_ = [bayesian_blocks(X, neg_ln_gamma=self.neg_ln_gamma)]
        # elif len(X.shape) == 2: # two dimensional data
        # else:
        #     raise RuntimeError("X has invalid shape!")

        return self


# https://sites.google.com/site/kittipat/programming-with-python/binningdatausingdecisiontreeregression
# http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

class DecisionTreeBinTransformer(BaseBinTransformer):

    def __init__(self, max_leaf_nodes=None, max_depth=5, min_samples_leaf=5):
        if max_leaf_nodes is not None:
            self.max_leaf_nodes = max_leaf_nodes
            self.max_depth = None
            self.min_samples_leaf = None
        else:
            self.max_leaf_nodes = None
            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf


    def fit(self, X, y=None):
        # X = check_array(X)
        if len(X.shape) != 2:
            raise RuntimeError("X has invalid shape!")

        self.X_ = X

        bins_ = []
        for i in range(X.shape[1]):
            x = X[:, i]
            x = x[~pd.isnull(x)]

            dtr = None
            if self.max_leaf_nodes is not None:
                dtr = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, random_state=0) # criterion='entropy',
            else:
                dtr = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=0) # criterion='entropy',

            dtr.fit(x.reshape(-1,1),x)
            thrs_out = np.unique(dtr.tree_.threshold[dtr.tree_.feature > -2])
            thrs_out = np.sort(thrs_out)
            min = np.min(x).reshape(-1)
            max = np.max(x).reshape(-1)
            thrs_out = np.concatenate([min, thrs_out, max])
            bins_ += [thrs_out]

        self.bins_ = bins_

        return self


def category_code_intervals_to_category_enumeration_labels(catdtype, intervals):
    if not isinstance(catdtype, pd.api.types.CategoricalDtype):
        raise RuntimeError('Only accepting pandas.api.types.CategoricalDtype values as input for catdtype!')
    if not catdtype.ordered:
        raise RuntimeError('Only accepting ordered pandas.api.types.CategoricalDtype as input for catdtype!')
    cat_levels = catdtype.categories.values
    cat_codes = range(int(intervals[0]),len(cat_levels))

    bin_index = 0
    labels = []
    label = []
    for i in cat_codes:
        if intervals[bin_index] <= i and i < intervals[bin_index + 1]:
            label += [cat_levels[i]]
        else:
            labels += ['[' + ','.join([str(l) for l in label]) + ']']
            label = [cat_levels[i]]
            bin_index += 1

    labels += ['[' + ','.join([str(l) for l in label]) + ']']

    return labels


class TargetVariableDecisionTreeBinTransformer0(BaseBinTransformer):

    def __init__(self, max_leaf_nodes=None, max_depth=5, min_samples_leaf=5):
        if max_leaf_nodes is not None:
            self.max_leaf_nodes = max_leaf_nodes
            self.max_depth = None
            self.min_samples_leaf = None
        else:
            self.max_leaf_nodes = None
            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf


    def fit(self, X, y):
        # X = check_array(X)
        if len(X.shape) != 2:
            raise RuntimeError("X has invalid shape!")
        # if len(X.shape[1]) != 1:
        #     raise RuntimeError("X has invalid shape; it has to be a column vector with only one column!")


        self.X_ = X
        self.y_ = y

        bins_   = []
        labels_ = []
        for i in range(X.shape[1]):
            x = X.iloc[:,i]
            # print('iteration: {}'.format(i))
            # if isinstance(x, pd.Series):
            #     print(x.name)
            not_null_index = ~pd.isnull(x) & ~pd.isnull(self.y_)

            y = self.y_[not_null_index]
            if isinstance(y, pd.Series) and y.dtype.name == 'category':
                y = y.cat.codes.values

            x = x[not_null_index]

            iscategory = False
            if isinstance(x, pd.Series) and x.dtype.name == 'category':
                iscategory = True
                x = x.cat.codes.values

            dtr = None
            if self.max_leaf_nodes is not None:
                dtr = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, random_state=0) # criterion='entropy',
            else:
                dtr = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=0) # criterion='entropy',

            if not isinstance(x, pd.Series):
                x = pd.Series(x)

            dtr.fit(x.values.reshape(-1,1),y)
            thrs_out = np.unique(dtr.tree_.threshold[dtr.tree_.feature > -2])
            thrs_out = np.sort(thrs_out)
            min = np.min(x).reshape(-1)
            max = np.max(x).reshape(-1) + 1
            thrs_out = np.concatenate([min, thrs_out, max])
            bins_ += [thrs_out]
            # print(bins_)
            if iscategory:
                labels_ += [category_code_intervals_to_category_enumeration_labels(self.X_.iloc[:,i].dtype, thrs_out)]
            else:
                labels_ += [None]

        self.bins_   = bins_
        self.labels_ = labels_

        return self

    def transform(self, X):
        try:
            getattr(self, "bins_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        check_is_fitted(self, ['bins_'], all_or_any=all)

        # X = check_array(X)

        r = X.copy()
        for i, bin in enumerate(self.bins_):
            # print('iteration: {}'.format(i))

            bin_boundaries = bin
            bin_labels = self.labels_[i]

            if bin_labels is None:
                r.iloc[:,i] = pd.cut(X.iloc[:,i], bin_boundaries, right=False)
            else:
                r.iloc[:,i] = pd.cut(X.iloc[:,i].cat.codes.values, bin_boundaries, labels=bin_labels, right=False)

        return r

def validate_node_or_level_name(name):
    if re.match(r'[A-Za-z][A-Za-z0-9_]*', name) is None:
        raise RuntimeError('You should only use node and/or factor level names that match the regex [A-Za-z][A-Za-z0-9_]*. You used: {}'.format(name))

def validate_node_or_level_names(name_list):
    for name in name_list:
        validate_node_or_level_name(name)

def sklearn_fit_helper_transform_X(X, sanitize_column_names_p=False):
    if isinstance(X, pd.DataFrame):
        ldf = X.copy()
    elif isinstance(X, np.ndarray):
        column_names = ['X{}'.format(i) for i in range(X.shape[1])]
        ldf = pd.DataFrame(X, columns=column_names)
    else:
        raise ValueError('Only accepting pandas dataframe or np.ndarray as input')

    if sanitize_column_names_p:
        ldf.columns = sanitize_column_names(ldf.columns)

    validate_node_or_level_names(ldf.columns)

    # check_array(ldf)
    if len(ldf.shape) != 2:
        raise RuntimeError("X has invalid shape!")
    return ldf

def sklearn_fit_helper_transform_y(y, sanitize_column_names_p=False):
    lds = None
    # if isinstance(l,(list,pd.core.series.Series,np.ndarray)):
    if isinstance(y, pd.Series):
        lds = y.copy()
    elif isinstance(y, np.ndarray):
        lds = pd.Series(y, name='y')
    else:
        raise ValueError('Only accepting pandas series or np.ndarray as input')

    if sanitize_column_names_p:
        lds.name = sanitize_column_name(lds.name)

    validate_node_or_level_names([lds.name])

    # check_array(ldf)
    if len(lds.shape) != 1:
        raise RuntimeError("y has invalid shape!")
    return lds


def discrete_and_continuous_variables_with_and_without_nulls(ldf, cutoff=30):
    discrete_non_null = []
    discrete_with_null = []
    continuous_non_null = []
    continuous_with_null = []
    levels_map = dict()
    for col in ldf.columns:
        uq = ldf[col].unique()
        # if col == 'Overall_Qual':
        #     print('Overall Qual')
        #     print(uq)
        number_type = False
        if all([np.issubdtype(type(level), np.number) for level in uq]):
            number_type = True

        if len(uq) > cutoff:
            if pd.isnull(uq).any():
                continuous_with_null += [col]
            else:
                continuous_non_null += [col]
        else:
            if pd.isnull(uq).any():
                discrete_with_null += [col]
                if number_type:
                    levels_map[col] = sorted(list(set(uq) - set([np.nan])))
                else:
                    levels_map[col] = set(uq)  - set([np.nan])
            else:
                discrete_non_null += [col]
                if number_type:
                    levels_map[col] = sorted(list(uq))
                else:
                    levels_map[col] = list(uq)

    return discrete_non_null, discrete_with_null, continuous_non_null, continuous_with_null, levels_map

def sanitize_column_name(name):
    r = '{}'.format(name).replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    if re.match(r'^[A-Za-z].*', r) is None:
        r = 'X{}'.format(r)
    return r


def sanitize_column_names(seq):
    r = [sanitize_column_name(e) for e in seq]
    return r

def convert_categories_to_string_categories(ldf, inplace=True):
    is_series = False
    if isinstance(ldf, pd.Series):
        ldf = pd.DataFrame(ldf, column=['y'])
        is_series = True

    if not inplace:
        ldf = ldf.copy()

    for column in ldf.columns:
        if ldf[column].dtype.name != 'category':
            continue
        levels = ['' + str(cat) for cat in ldf[column].dtype.categories]
        cdt = pd.api.types.CategoricalDtype(levels, ordered=True)
        ldf.loc[:,column] = ldf[column].apply(lambda x: str(x)).astype(cdt)

    if is_series:
        return ldf.y
    else:
        return ldf

def index_compare(o1, o2):
    if (o1 is None) or o2 is None:
        return None
    return set(o1.index) ^ set(o2.index)


class MetaDataInitTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, metadata, sanitize_column_names_p=True):
        self.metadata                = metadata
        self.sanitize_column_names_p = sanitize_column_names_p
        self.fit_count       = 0
        self.transform_count = 0

    def fit(self, X=None, y=None):
        # print('{}: fit, fit_count: {}, transform_count: {}'.format(type(self).__name__, self.fit_count, self.transform_count))
        self.fit_count += 1
        self.metadata.clear()


        self.df = X.copy()
        self.df_ = sklearn_fit_helper_transform_X(X, sanitize_column_names_p=self.sanitize_column_names_p)

        if y is not None:
            self.has_y = True
            self.df = pd.concat([self.df, y], axis=1)
            self.y = sklearn_fit_helper_transform_y(y, sanitize_column_names_p=self.sanitize_column_names_p)
            self.df_[self.y.name] = self.y
            self.metadata['y_']   = self.y

        self.metadata['sanitized_column_name_mapping'] = dict(zip(self.df.columns, self.df_.columns))

        return self

    def transform(self, X):
        # print('{}: transform, fit_count: {}, transform_count: {}'.format(type(self).__name__, self.fit_count, self.transform_count))
        if self.transform_count  < self.fit_count:
            self.transform_count += 1
        return sklearn_fit_helper_transform_X(X, sanitize_column_names_p=self.sanitize_column_names_p)



class MetaDataTransformerBase(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, metadata={}):
        self.metadata = metadata
        self.fit_count       = 0
        self.transform_count = 0

    def extract_X_and_y(self, X=None, y=None):
        has_y = False

        df = X.copy()

        y_untransformed = y
        if 'y_' in self.metadata:
            y = self.metadata['y_']

        if y is not None:
            has_y = True
            df = pd.concat([df, y], axis=1)

        return has_y, df, y_untransformed, y


    def fit(self, X=None, y=None, sanitize_column_names_p=False):
        # print('{}: fit, fit_count: {}, transform_count: {}'.format(type(self).__name__, self.fit_count, self.transform_count))
        self.fit_count += 1
        # print(X['Overall_Qual'].unique())
        has_y, df, y_untransformed, y = self.extract_X_and_y(X, y)
        # print(df['Overall_Qual'].unique())
        self.has_y           = has_y
        self.df              = df
        self.y_untransformed = y_untransformed
        self.y               = y

        self.scn = lambda x: x
        if 'sanitized_column_name_mapping' in self.metadata:
            self.scn = lambda x: self.metadata['sanitized_column_name_mapping'][x]

        self.fit_with_metadata()

        if self.has_y:
            self.metadata['y_'] = self.df.iloc[:,-1]


        return self

    def transform(self, X):
        # print('{}: transform, fit_count: {}, transform_count: {}'.format(type(self).__name__, self.fit_count, self.transform_count))
        if self.transform_count  < self.fit_count:
            self.transform_count += 1
            if self.has_y:
                return self.df.iloc[:,:-1]
            else:
                return self.df

        return self.transform_with_metadata(X)


class CategoricalTransformer(MetaDataTransformerBase):

    def __init__(self, categorical_columns, ordered_categorical_columns, discrete_columns, continuous_columns, levels_map, metadata={}):
        super().__init__(metadata=metadata)
        self.categorical_columns         = categorical_columns
        self.ordered_categorical_columns = ordered_categorical_columns
        self.discrete_columns            = discrete_columns
        self.continuous_columns          = continuous_columns
        self.levels_map                  = levels_map

    def fit_with_metadata(self):
        _, _, _, _, derived_levels_map = discrete_and_continuous_variables_with_and_without_nulls(self.df)
        self.derived_levels_map = derived_levels_map
        self.df = self.transform_with_metadata(self.df)

    def transform_with_metadata(self, ldf):
        for col in self.categorical_columns:
            scol = self.scn(col)
            if scol not in list(ldf.columns):
                if scol == self.y.name:
                    continue
                raise RuntimeError('You specified col: {} as one of the categorical_columns, but scol: {} does not exist in data frame columns: {}'.format(col, scol, list(ldf.columns)))

            if col in self.levels_map:
                levels = self.levels_map[col]
            elif scol in self.derived_levels_map:
                levels = self.derived_levels_map[scol]
            else:
                levels = None

            if all([np.issubdtype(type(level), np.number) for level in levels]):
                levels = sorted(levels)
                ldf[scol] = ldf[scol].astype(pd.api.types.CategoricalDtype(levels, ordered=True))
            else:
                ldf[scol] = ldf[scol].astype(pd.api.types.CategoricalDtype(levels, ordered=False))

        for col in self.ordered_categorical_columns:
            scol = self.scn(col)
            if scol not in list(ldf.columns):
                if scol == self.y.name:
                    continue
                raise RuntimeError('You specified col: {} as one of the ordered_categorical_columns, but scol: {} does not exist in data frame columns: {}'.format(col, scol, list(ldf.columns)))

            if col in self.levels_map:
                levels = self.levels_map[col]
            elif scol in self.derived_levels_map:
                levels = self.derived_levels_map[scol]
            else:
                levels = None

            ldf[scol] = ldf[scol].astype(pd.api.types.CategoricalDtype(levels, ordered=True))

        for col in self.continuous_columns:
            scol = self.scn(col)
            if scol not in list(ldf.columns):
                if scol == self.y.name:
                    continue
                raise RuntimeError('You specified col: {} as one of the continuous_columns, but scol: {} does not exist in data frame columns: {}'.format(col, scol, list(ldf.columns)))

            ldf[scol] = ldf[scol].astype(float)

        for col in self.discrete_columns:
            scol = self.scn(col)
            if scol not in list(ldf.columns):
                if scol == self.y.name:
                    continue
                raise RuntimeError('You specified col: {} as one of the discrete_columns, but scol: {} does not exist in data frame columns: {}'.format(col, scol, list(ldf.columns)))
            if pd.isnull(self.df[scol]).any():
                ldf[scol] = ldf[scol].astype(float)
            else:
                ldf[scol] = ldf[scol].astype(int)

        return ldf


class TargetVariableDecisionTreeBinTransformer(MetaDataTransformerBase):
    def __init__(self, max_leaf_nodes=None, max_depth=5, min_samples_leaf=5, binning_variables=[], metadata={}):
        super().__init__(metadata=metadata)
        self.max_leaf_nodes    = max_leaf_nodes
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.binning_variables = binning_variables

    def fit_with_metadata(self):
        sbvs = [self.scn(bv) for bv in self.binning_variables]
        self.tvbt = TargetVariableDecisionTreeBinTransformer0(max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.tvbt.fit(self.df[sbvs], self.y)
        self.df.loc[:, sbvs] = self.transform_with_metadata(self.df)


    def transform_with_metadata(self, ldf):
        sbvs = [self.scn(bv) for bv in self.binning_variables]
        ldf.loc[:, sbvs] = self.tvbt.transform(ldf[sbvs])
        return ldf


class NullToNATransformer(MetaDataTransformerBase):
    def __init__(self, null_to_NA_columns=[], metadata={}):
        super().__init__(metadata=metadata)
        self.null_to_NA_columns          = null_to_NA_columns

    def fit_with_metadata(self):
        self.df = self.transform_with_metadata(self.df)

    def transform_with_metadata(self, ldf):
        for col, na_symbol in self.null_to_NA_columns:
            scol = self.scn(col)
            ldf.loc[pd.isnull(ldf[scol]), [scol]] = na_symbol
        return ldf


class CategoryLevelsAsStringsTransformer(MetaDataTransformerBase):
    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)

    def fit_with_metadata(self):
        self.df = self.transform_with_metadata(self.df)

    def transform_with_metadata(self, ldf):
        return convert_categories_to_string_categories(ldf)


class PandasCutBinTransformer(MetaDataTransformerBase):
    def __init__(self, boundaries_map, right=False, derive_start_and_end=True, metadata={}):
        super().__init__(metadata=metadata)
        self.boundaries_map = boundaries_map
        self.right = right
        self.derive_start_and_end = derive_start_and_end

    def fit_with_metadata(self):
        self.df = self.transform_with_metadata(self.df)

    def transform_with_metadata(self, ldf):
        dec_left = 0
        inc_right = 0
        if self.right:
            dec_left = 1
        else:
            inc_right = 1

        for col, boundaries in self.boundaries_map.items():
            scol = self.scn(col)
            if scol not in list(ldf.columns):
                if scol == self.y.name:
                    continue
                raise RuntimeError('You specified col: {} as one of the boundaries, but scol: {} does not exist in data frame columns: {}'.format(col, scol, list(ldf.columns)))

            lds = ldf[scol]
            if self.derive_start_and_end:
                boundaries = [lds.min() - dec_left] + boundaries + [lds.max() + inc_right]

            ldf[scol] = pd.cut(lds, boundaries, right=self.right)
        return ldf


class FilterNullTransformer(MetaDataTransformerBase):
    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)

    def fit_with_metadata(self):
        _, discrete_with_null, _, continuous_with_null, _ = discrete_and_continuous_variables_with_and_without_nulls(self.df)
        null_fields = discrete_with_null + continuous_with_null
        self.null_fields = null_fields
        null_index = np.full(len(self.df), False)
        for col in null_fields:
            null_index |= pd.isnull(self.df[col])
        self.null_index = null_index

        self.df = self.df[~null_index]

    def transform_with_metadata(self, ldf):
        _, discrete_with_null, _, continuous_with_null, _ = discrete_and_continuous_variables_with_and_without_nulls(ldf)
        null_fields = discrete_with_null + continuous_with_null
        null_index = np.full(len(ldf), False)
        for col in null_fields:
            null_index |= pd.isnull(ldf[col])

        return ldf[~null_index].copy()


class DropColumnTransformer(MetaDataTransformerBase):
    def __init__(self, columns, metadata={}):
        super().__init__(metadata=metadata)
        self.columns = columns

    def fit_with_metadata(self):
        self.df = self.transform_with_metadata(self.df)

    def transform_with_metadata(self, ldf):
        for col in self.columns:
            ldf.drop(col, axis=1, inplace=True)
        return ldf

class MetaDataTransformerClassifierOrRegressorWrapper(MetaDataTransformerBase, sklearn.base.RegressorMixin):
    def __init__(self, base_classifier, metadata={}):
        super().__init__(metadata=metadata)
        self.base_classifier = base_classifier

    def fit_with_metadata(self):
        self.base_classifier.fit(self.df.iloc[:,:-1], self.df.iloc[:,-1])

    def predict(self, X):
        y = self.base_classifier.predict(X)
        y.index = X.index
        return y


class ClassifierToRegressorHelper(MetaDataTransformerBase, sklearn.base.RegressorMixin):
    def __init__(self, base_classifier, metadata={}):
        super().__init__(metadata=metadata)
        self.base_classifier = base_classifier

    def fit_with_metadata(self):
        self.base_classifier.fit(self.df.iloc[:,:-1], self.df.iloc[:,-1])

        y_untransformed = self.y_untransformed.loc[self.y.index]
        self.category_to_median_mapping = y_untransformed.groupby(self.y).median()
        # self.category_to_mean_mapping_dict = dict(zip(self.category_to_mean_mapping.index, self.category_to_mean_mapping.values))

    def predict(self, X):
        labels = self.base_classifier.predict(X)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)

        y =  labels.map(self.category_to_median_mapping)
        y.index = X.index
        return y

    def score(self, X, y, sample_weight=None):
        y_ = y.loc[X.index]
        return super().score(X, y_, sample_weight=sample_weight)
        # from sklearn.metrics import r2_score
        # return r2_score(y_, self.predict(X), sample_weight=sample_weight, multioutput='variance_weighted')

