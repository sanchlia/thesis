import numpy as np
import pandas as pd
import random

class Noise:
    def __init__(self, noise_type, noise_level, majority_group, positive_class, label, sensitive_attribute, data):
        # self.train_label = df['train']
        # self.test_label = df['test']
        # self.noise_type = ['random', 'class', 'group']
        # self.noise_level = [0.2, 0.4, 0.6, 0.8]
        # self.df = df
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.majority_group = int(majority_group)
        self.positive_class = int(positive_class) 
        self.label = label
        self.sensitive_attribute = sensitive_attribute
        self._data = data.copy()

        
    def validate(self):
        if self.noise_type == 'bias':
            pass 
        elif self.noise_type == 'flip':
            pass 

    def inject_noise(self, label:pd.Series, group:pd.Series, seed = 1):        
        # for n_type in noise_type:
        #     for n_level in noise_level:
        if self.noise_type == 'flip':
            self.label_noisy = self.flip_noise(label, group, seed)
        elif self.noise_type == 'bias':
            self.label_noisy = self.bias_noise(label, group)
        elif self.noise_type == 'balanced':
            self.label_noisy = self.balanced_bias_noise(label, group)
        else:
            raise ValueError("Noise type not found.")   
        # validate noisy labels 
        # should have same size as label, count of group by unimpacted sv should be same
        if len(label) != len(self.label_noisy):
            raise "Noisy label not the same shape as given labels"
        return self.label_noisy

    def flip_noise(self, y:pd.Series, group:pd.Series, s):
        """
          

        Parameters
        ----------
        y : pd.Series
            Clean labels
        group : pd.Series
            Sensitive attribute values
        noise_rate : float
            Amount of noise to inject

        Returns
        -------
        y_noisy : pd.Series
            Noisy labels
        """
        random.seed(s)
        y_noisy = y.copy()

        for i in group.loc[group != self.majority_group].index:
            if random.random() < self.noise_level:
                y_noisy.loc[i] = self.positive_class - y.loc[i] 

        return y_noisy

    def bias_noise(self, y:pd.Series, group:pd.Series):
        """
        Generate group-dependant noise that is biased towards favoring the protected group.

        Parameters
        ----------
        y : pd.Series
            Clean labels
        group : pd.Series
            Sensitive attribute values
        noise_rate : float
            Amount of noise to inject

        Returns 
        -------
        y_noisy : pd.Series
            Noisy labels
        """
        random.seed(0)
        y_noisy = y.copy()

        for i in group.loc[group == self.majority_group].index:
            if random.random() < self.noise_level:
                y_noisy.loc[i] = self.positive_class

        return y_noisy

    def balanced_bias_noise(self, y:pd.Series, group:pd.Series):
        """
        Generate group-dependant noise that is biased towards simultaneously favoring the protected group and harming the unprotected group.

        Parameters
        ----------
        y : pd.Series
            Clean labels
        group : pd.Series
            Sensitive attribute values
        noise_rate : float
            Amount of noise to inject

        Returns
        -------
        y_noisy : pd.Series
            Noisy labels
        """
        random.seed(0)
        y_noisy = y.copy()

        for i in y_noisy.index:
            if random.random() < self.noise_level:
                if group.loc[i] == self.majority_group:
                    y_noisy.loc[i] = self.positive_class
                else:
                    y_noisy.loc[i] = self.positive_class ^ 1 

        return y_noisy


    def validate_result(self,result_df):
        cols = [c for c in self._data.columns if c not in [self.label]]
        for column in cols:
            if not self._data[column].equals(result_df[column]):
                print('Error')
                return False
        return True  



    def calculate_fair_probabilities(self, y, data, noise_type):

        if noise_type == 'bias':
            df = pd.DataFrame()
            df['xy'] = data[(data[self.sensitive_attribute] == self.majority_group) & (data[self.label] == 1)].apply(tuple, axis = 1)
            df['yf'] = y 

            df2 = df.groupby('xy').agg({'yf': ['count', 'sum']})
            prob = (df2['yf', 'sum'] + 1)/ (df2['yf', 'count'] + 1)
            prob_df= pd.DataFrame({'xy': prob.index.values, 'prob': prob.to_numpy()})
            data['xy'] = data.apply(tuple, axis = 1)
            data['ground_truth']  = y
            result_df = pd.merge(data, prob_df, on = 'xy', how = 'left')

            result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group ^ 1)) & (result_df[self.label] == 1), 'prob']= 1
            result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group  ^ 1)) & (result_df[self.label] == 0), 'prob'] = 0
            result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group )) & (result_df[self.label] == 0), 'prob'] = 0

            return result_df
        elif noise_type == 'flip':
            df = pd.DataFrame()
            df['xy'] = data[(data[self.sensitive_attribute] != self.majority_group)].apply(tuple, axis = 1)
            df['yf'] = y 

            df2 = df.groupby('xy').agg({'yf': ['count', 'sum']})
            prob = (df2['yf', 'sum'] + 1)/ (df2['yf', 'count'] + 1)
            prob_df= pd.DataFrame({'xy': prob.index.values, 'prob': prob.to_numpy()})

            _data = data.copy()
            _data['xy'] = data.apply(tuple, axis = 1)
            _data['ground_truth']  = y

            result_df = pd.merge(_data, prob_df, on = 'xy', how = 'left')
            result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group)) & (result_df[self.label] == 1), 'prob'] = 1
            result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group )) & (result_df[self.label] == 0), 'prob'] = 0
            
        elif noise_type == 'balanced':
            df = pd.DataFrame()
            df['xy'] = data.apply(tuple, axis = 1)
            df['yf'] = y 

            df2 = df.groupby('xy').agg({'yf': ['count', 'sum']})
            prob = (df2['yf', 'sum'] + 1)/ (df2['yf', 'count'] + 1)
            prob_df= pd.DataFrame({'xy': prob.index.values, 'prob': prob.to_numpy()})
            data['xy'] = data.apply(tuple, axis = 1)
            data['ground_truth']  = y
            result_df = pd.merge(data, prob_df, on = 'xy', how = 'left')
            # result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group ^ 1)) & (result_df[self.label] == 1), 'prob']= 1
            # result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group  ^ 1)) & (result_df[self.label] == 0), 'prob'] = 0
            # result_df.loc[(result_df[self.sensitive_attribute] == (self.majority_group )) & (result_df[self.label] == 0), 'prob'] = 0
    


        if not self.validate_result(result_df):
            raise "Error, result df is not same"
        return result_df