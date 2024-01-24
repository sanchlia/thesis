
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.reductions import ExponentiatedGradient


class FairReduction():
    def __init__(self):
        pass
 
    def fit(self, X_train, y_train, s_train,  weights = None):
      
        learn = ExponentiatedGradient(
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=EqualizedOdds())
        # learn.fit(X_train, y_train,sensitive_features= s_train)

        # Add support for model with params.
        learn.fit(X_train, y_train,sensitive_features= s_train, fair_weights = weights)
        return learn
        