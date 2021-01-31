import numpy as np

def residuals(x, t, y, target_function=None, weight=None, kwargs=None):
    """
    Returns the residual between the target function and the observed data y
    for a given weight.

    :param x:
        parameters passed to the target function
    :param t:
        np array of length N
        time, or X-axis (independent variable)
          OR
        a (2,N) array
        where N is the total number of data points in y
    :param y:
        np array of length N
        observed data
    :param target_function:
        function to calculate the residual against
            residual = target_function(t, *x) - y
        target_function takes x as arguments, and t as independent variable,
        E.G. for a linear target function:
        def target_function(t, a, b):
            return a*t + b
    :param weight:
        np array of length N
        weights between 0 and 1
        or None
    :param kwargs:
        dictionary
        keywords passed to the target function
    :return:
        np array of length N of residuals
    """
    if kwargs is None:
        kwargs = {}
    if target_function is None:
        raise TypeError('function must be provided to return a residual')

    if len(t.shape) == 2 and t.shape[0] == 2:  # 2D case
        t1, t2 = t
    else:  # assume standard 1D case
        t1 = t; t2 = None

    if weight is None:
        weight = np.ones(t1.shape)
    elif np.nanmin(weight) < 0.:
        raise ValueError('Weights should be larger than 0.')
    elif np.nanmax(weight) > 1.:
        raise ValueError('Weights should not be larger than 1.')
    elif weight.shape != y.shape:
        raise ValueError('Weights must have same dimension as input data')

    if len(t.shape) == 2 and t.shape[0] == 2:  # 2D case
        return weight*(target_function(t1, t2, *x, **kwargs) - y)
    else:  # assume standard 1D case
        return weight * (target_function(t1, *x, **kwargs) - y)

