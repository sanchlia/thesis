import numpy as np
import os
import pandas as pd
import logging 

from pathlib import PurePath

from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)
class Reweight():
    """
    Args:
    Constraint -> DP/EO/EOP
    stage -> preprocess/post process
    """
    def __init__(self):
        pass

        
    def fit(self, X_train, y_train, protected_train):
        logger.info("Inside reweight")
        self.obj = EOddsReweight()
        models =  self.obj.learn(X_train, y_train, [protected_train])

        return models




class EOddsReweight:
    def __init__(self) -> None:
        pass

    def debias_weights(self, original_labels, protected_attributes, multipliers):
        exponents_pos = np.zeros(len(original_labels))
        exponents_neg = np.zeros(len(original_labels))

        for i, protected in enumerate(protected_attributes):
            exponents_pos -= multipliers[2 * i] * protected
            exponents_neg -= multipliers[2 * i + 1] * protected
        weights_pos = np.exp(exponents_pos)/ (np.exp(exponents_pos) + np.exp(-exponents_pos))
        weights_neg = np.exp(exponents_neg)/ (np.exp(exponents_neg) + np.exp(-exponents_neg))

        #weights = np.where(predicted > 0, weights, 1 - weights)
        weights = np.where(original_labels > 0, 1 - weights_pos, weights_neg)
        return weights
    

    def get_error_and_violations(self, y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
            protected_idxs = np.where(np.logical_and(p > 0, y < 1))
            negative_idxs = np.where(y < 1)
            violations.append(np.mean(y_pred[negative_idxs]) - np.mean(y_pred[protected_idxs]))
        pairwise_violations = []
        
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
                pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))

        return acc, violations, pairwise_violations
            

    def learn(self, X_train, y_train, protected_train):
        multipliers = np.zeros(len(protected_train) * 2)
        weights = np.array([1] * X_train.shape[0])
        learning_rate = 1.
        n_iters = 100
        for it in range(n_iters):
            model = LogisticRegression()

            model.fit(X_train, y_train, weights)
            y_pred_train = model.predict(X_train)

            weights = self.debias_weights(y_train, protected_train, multipliers)
            acc, violations, pairwise_violations = self.get_error_and_violations(y_pred_train, y_train, protected_train)
            multipliers += learning_rate * np.array(violations)


            if (it + 1) % n_iters == 0:
                return model
                # y_pred_test = model.predict(X_test)
                # acc, violations, pairwise_violations = self.get_error_and_violations(y_pred_train, y_train, protected_train)
                # print("Train Accuracy", acc)
                # print("Train Violation", max(np.abs(violations)), " \t\t All violations", violations)
        #     if len(pairwise_violations) > 0:
        #         print("Train Intersect Violations", max(np.abs(pairwise_violations)), " \t All violations", pairwise_violations)

        #     acc, violations, pairwise_violations = self.get_error_and_violations(y_pred_test, y_test, protected_test)
        #     print("Test Accuracy", acc)
        #     print("Test Violation", max(np.abs(violations)), " \t\t All violations", violations)
        #     if len(pairwise_violations) > 0:
        #         print("Test Intersect Violations", max(np.abs(pairwise_violations)), " \t All violations", pairwise_violations)

        # return model