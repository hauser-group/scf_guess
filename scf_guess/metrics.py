import numpy as np


def f_score(S, Da_true, Da_pred, Db_true=None, Db_pred=None):
    if Db_true is None:
        return np.trace(Da_pred @ S @ Da_true @ S) / np.trace(Da_true @ S)
    return (
        np.trace(Da_pred @ S @ Da_true @ S) + np.trace(Db_pred @ S @ Db_true @ S)
    ) / (np.trace(Da_true @ S) + np.trace(Db_true @ S))
