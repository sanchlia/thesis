import numpy as np
import pandas as pd



from pathlib import PurePath
from sklearn.linear_model import LogisticRegression



class Lr():
    def __init__(self) -> None:
        pass

    def fit(self, x_train, y_train, s_train = None):
        model = LogisticRegression()
        model.fit(x_train,y_train)
        return model
    
    def summary_path(self, name):
        folder_path = None
        if self.exp_id is None:
            folder_path = PurePath(DIR, "exp", self.dataset, "baseline", self.constraint, self.stage, str(self.threshold), name)

        if self.exp_id is not None:
            folder_path = PurePath(DIR, "exp", self.dataset, str(self.exp_id), self.constraint, self.stage, str(self.threshold), name)
        self.folder_path = folder_path
        return folder_path
        

