import pandas as pd
import numpy as np 
import os

from collections import defaultdict

class Dataset:
    """Dataset class for loading and processing data."""

    def __init__(self, meta, fair_flag = False, emp_flag = False):

        self.meta = meta

        # dataset details 
        self.label = self.meta["label"]
        self.name = self.meta["name"]
        self.exp_name = self.meta['exp_name']
        self.sensitive_attribute = self.meta["sensitive_attribute"]
        self.majority_group = self.meta['majority_group']
        self.positive_class = self.meta['positive_class']

        self.data = pd.read_csv(self.meta["data_path"])

        # path data 
        self.splits_path = self.meta["splits_path"]
        if 'ground_data' in self.meta:
            self.ground_data_path = self.meta['ground_data']
        
        # prob path 
        self.fair_prob_path = self.meta['fair_path']
        self.emp_prob_path = self.meta['emp_path']

        # noise details
        self.noise_level = self.meta['noise_level']
        self.noise_type = self.meta['noise_type']

        self.cols_to_drop = [self.label]
        
        if self.name in ['synthetic_20', 'synthetic_10', 'synthetic_30']:
            self.cols_to_drop.append('D')
        
        self.foldwise_split = self.load_foldwise_split()
        self.foldwise_data = self.load_data(self.data)
        
        self.set_fair_probabilities()

        if self.noise_level != None and self.noise_type != None:
            self.ground_data = self.load_data(pd.read_csv(self.ground_data_path))
            self.set_emp_prob()
            self.combine_train_valid(self.ground_data)
            train_label, test_label = self.get_label_df(self.ground_data)
            self.ground_labels = {'train': train_label, "test": test_label}
        self.combine_train_valid(self.foldwise_data)
        self.fair_clean_labels = self.generate_clean_labels('prob')
        if self.noise_level != None and self.noise_type != None:
            self.emp_clean_labels = self.generate_clean_labels('emp_prob')

    def generate_clean_labels(self, prob_name):
        clean_labels = defaultdict(dict)
        
        for fold, _split in self.foldwise_data.items():
            try:
                for split_type, split_data in _split.items():
                    clean_labels[fold][split_type] = self.process_preds(split_data[prob_name], 0.5)
            except KeyError as e:
                print(f"{self.name=} Key error in {fold=}, {split_type=} and {prob_name=}")
        return clean_labels
    
    def set_emp_prob(self):
        probs = pd.read_csv(self.emp_prob_path)
        for fold, split in self.foldwise_split.items():
            for split_type, ids in split.items():
                self.foldwise_data[fold][split_type].loc[:,'emp_prob'] = probs['prob'].iloc[ids].reset_index(drop=True)
        self.cols_to_drop.append('emp_prob')



    def set_fair_probabilities(self):
        # hardcoded path, need to change 
        base_path = 'FairProbabilities'
        for fold in self.foldwise_data.keys():
            try:

                prob_path = os.path.join(self.fair_prob_path,fold[1:], 'para', 'max-ll', 'predict-per-example-proxy-label.csv')
           
                df = pd.read_csv(prob_path)
            except FileNotFoundError as e:
                print(f"{prob_path} not found")
                continue
            for lvl in ['train', 'valid', 'test']:
                probs = pd.Series(df[f"P(Df|e) {lvl}_x"])
                # prob_y = pd.Series(df[f"P(Df) {lvl}_x"])
                probs =probs[~np.isnan(probs)].astype(float)
                # prob_y =prob_y[~np.isnan(prob_y)].astype(float)

                self.foldwise_data[fold][lvl].loc[:,'prob'] = probs 
                # self.foldwise_data[fold][lvl].loc[:,'prob_y'] = prob_y 

        self.cols_to_drop.append('prob')

    
    def get_label_df(self, df):
        train_dict = defaultdict(pd.Series)
        test_dict = defaultdict(pd.Series)
        for key in df.keys():
            for split, _df in df[key].items():
                if split == 'train':
                    train_dict[key] = _df[self.label].dropna()
                if split == 'test':
                    test_dict[key] = _df[self.label].dropna()
                # label_dict[key][split] = df[self.label].dropna()
        return train_dict, test_dict

    def combine_train_valid(self, df):
        for fold, data_dict in df.items():
            data_dict['train'] = pd.concat([data_dict['train'], data_dict['valid']]).reset_index(drop=True)
            

    def load_foldwise_split(self):
        """
        Loads all the folds in one go. 
        Saved in a nested dict with fold being the keys
        dict = {
            "fold_1":{
                "train":df,
                "test":df,
                "valid":df
            }
        }
        """
        foldwise_split = {}
        train_splits = pd.read_csv(os.path.join(self.splits_path, "train_ids.csv"))
        test_splits = pd.read_csv(os.path.join(self.splits_path, "test_ids.csv"))
        valid_splits = pd.read_csv(os.path.join(self.splits_path, "valid_ids.csv"))
        for fold in train_splits.columns:
            train_id = np.array(train_splits[fold]) - 1
            valid_id = np.array(valid_splits[fold]) - 1
            test_id = np.array(test_splits[fold]) - 1
            train_id = train_id[~np.isnan(train_id)].astype(int)
            valid_id = valid_id[~np.isnan(valid_id)].astype(int)
            test_id = test_id[~np.isnan(test_id)].astype(int)
            foldwise_split[fold] = {"train":  train_id, "test": test_id, 'valid': valid_id}

        return foldwise_split

    def load_data(self, df):
        """Load data from path for each split."""
        data = defaultdict(dict)
        for fold, fold_data in self.foldwise_split.items():
            data[fold]['train'] = df.iloc[fold_data['train']].reset_index(drop=True)
            # pd.concat([self.data.iloc[fold_data['train']] ,self.data.iloc[fold_data['valid']]])
            data[fold]['valid'] = df.iloc[fold_data['valid']].reset_index(drop=True)
            data[fold]['test'] = df.iloc[fold_data['test']].reset_index(drop=True)
        return data
    def eval_labels(self):
        label_map = {}
        # label_map['ground'] = {key: value['test'][self.label] for key, value in self.foldwise_data.items()}
        label_map['fair_clean'] = {key: self.process_preds(value['test']['prob'],0.5) for key, value in self.foldwise_data.items()}

        if self.noise_level != None and self.noise_type != None:
            label_map['emp_clean'] = {key: self.process_preds(value['test']['emp_prob'],0.5) for key, value in self.foldwise_data.items()}

            label_map['noisy'] = {key: value['test'][self.label] for key, value in self.foldwise_data.items()}
            label_map['ground'] = self.ground_labels['test']

        return label_map
    def process_preds(self,pred, threshold = 0.5):
        """
        Processes the predictions to be binary.
        threshold: float

        returns: binary predictions
        """
        #inclusive range is new update vis a vis TPM
        pred  = (pred[~np.isnan(pred)] >= threshold).astype(int)
        return pred
    def calculate_probabilities(self, prob_type):
        if prob_type == 'emp':
            _EVENT = "event"
            _GROUP_ID = "group_id"
            group_event_tuples = [('label=0', 0), ('label=0', 1) , ('label=1', 0), ('label=1', 1)]
            group_event_index = pd.MultiIndex.from_tuples(group_event_tuples, names = ['event', 'group_id'])
            self.emp_prob_map = defaultdict(dict)
            for fold, data in self.ground_data.items():
                # data is 
                # {"train:[]", 
                #  "test":[]
                # }
                # calculate probabilities 
                self.emp_prob_map[fold] = {}
                for split in ['train', 'test']:
                    temp_data = data[split].copy(deep = True)
                    temp_data = temp_data.rename(columns={self.sensitive_attribute:_GROUP_ID})
                    temp_data[_EVENT] = data[split][self.label]
                    # P(Y|S)
                    s_count = temp_data.groupby([_GROUP_ID]).size()
                    ys_count = temp_data.groupby([_EVENT, _GROUP_ID]).size()
                    p_y_s = ys_count/s_count
                    p_y_s.index = p_y_s.index.set_levels(['label=0', 'label=1'], level='event')
                    # P(Y)
                    p_y =  temp_data.groupby(_EVENT).size()/len(temp_data)
                    p_y = p_y.rename(index = {0:'label=0', 1: 'label=1'})
                    self.emp_prob_map[fold][split] = {
                        "p_y": p_y, 
                        "p_y_s":p_y_s
                    }

                
                
            # _EVENT = "event"
            # _GROUP_ID = "group_id"
            # self.data
            # ground_data = pd.read_csv(self.ground_data_path)
            # temp_data = self.data.copy(deep=True)
            # temp_data = temp_data.rename(columns={self.sensitive_attribute:_GROUP_ID})
            # temp_data[_EVENT] = ground_data[self.label]
            # # P(Y|S)
            # s_count = temp_data.groupby([_GROUP_ID]).size()
            # ys_count = temp_data.groupby([_EVENT, _GROUP_ID]).size()
            # self.p_y_s = ys_count/s_count
            # self.p_y_s.index = self.p_y_s.index.set_levels(['label=0', 'label=1'], level='event')
            # # P(Y)
            # self.p_y =  temp_data.groupby(_EVENT).size()/len(temp_data)
            # self.p_y = self.p_y.rename(index = {0:'label=0', 1: 'label=1'})
        elif prob_type == "fair":
            _EVENT = "event"
            _GROUP_ID = "group_id"
            group_event_tuples = [('label=0', 0), ('label=0', 1) , ('label=1', 0), ('label=1', 1)]
            group_event_index = pd.MultiIndex.from_tuples(group_event_tuples, names = ['event', 'group_id'])
            self.fair_prob_map = defaultdict(dict)
            for fold in self.foldwise_data.keys():
                try:

                    prob_path = os.path.join(self.fair_prob_path,fold[1:], 'para', 'max-ll', 'predict-per-example-proxy-label.csv')
            
                    df = pd.read_csv(prob_path)
                    prob_y = pd.Series(df[f"P(Df) train_x"])
                    prob_y =prob_y[~np.isnan(prob_y)].astype(float)
                    p_y_1 = prob_y.iloc[0]
                    p_y_0 = 1 - p_y_1
                    # print(f'{p_y_0=} and {p_y_1=}')
                    if p_y_1 + p_y_0 != 1.0:
                        print("fair probs are not correct")
                        print(p_y_0 + p_y_1)
                        raise Exception
                    
                    self.fair_prob_map[fold]["p_y"] = pd.Series([p_y_0, p_y_1], index = ['label=0', 'label=1'])
                    self.fair_prob_map[fold]["p_y"].index.names = ['event']
                    self.fair_prob_map[fold]["p_y_s"] = pd.Series([p_y_0, p_y_0, p_y_1, p_y_1], index =group_event_index)

                except FileNotFoundError as e:
                    print(f"{prob_path} not found")
                    continue