import numpy as np
import pandas as pd

from scipy.special import xlog1py
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, precision_score, accuracy_score, recall_score

from fairlearn.metrics import (
    MetricFrame, plot_model_comparison,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,true_positive_rate, 
    equalized_odds_difference)


class Metrics:
    def __init__(self):
        pass

    def process_preds(self,pred, threshold = 0.5):
        """
        Processes the predictions to be binary.
        threshold: float

        returns: binary predictions
        """
        #inclusive range is new update vis a vis TPM
        pred  = (pred[~np.isnan(pred)] >= threshold).astype(int)
        return pred
    
    def performance_metrics(self, y_true, y_pred, threshold = 0.5, meta = {}):
        
        res = {}
        y = (y_true[~np.isnan(y_true)]).astype(int)
        y_pred_threshold = self.process_preds(y_pred, threshold)

        res["accuracy"] = accuracy_score(y, y_pred_threshold)
        res["f1_score"] = f1_score(y, y_pred_threshold)
        res["precision"] = precision_score(y, y_pred_threshold)
        res["recall"] = recall_score(y, y_pred_threshold)
        res["FNR"] = false_negative_rate(y, y_pred_threshold)
        res["FPR"] = false_positive_rate(y, y_pred_threshold)
        res.update(meta)
        return res
    
    def fairness_metrics(self, y_true, y_pred, sv, threshold = 0.5, meta = {}):
        """
        Calculates the metrics for the given y_true and y_pred.
        Returns a dictionary
        Groups by fairness like equalized_odds, demographic_parity, demographic_parity_ratio, equal_opportunity.
        """
        y = (y_true[~np.isnan(y_true)]).astype(int)
        y_pred_threshold = (y_pred[~np.isnan(y_pred)] > threshold).astype(int)
        sv = (sv[~np.isnan(sv)]).astype(int)
        res = {}
        res["FNR_diff"] = false_negative_rate_difference(y, y_pred_threshold, sensitive_features = sv)
        res["FPR_diff"] = false_positive_rate_difference(y, y_pred_threshold, sensitive_features = sv)
        res["equalized_odds"] = equalized_odds_difference(y, y_pred_threshold, sensitive_features= sv )        
        res["demographic_parity"] = demographic_parity_difference(y, y_pred_threshold, sensitive_features=sv)
        res["demographic_parity_ratio"] = demographic_parity_ratio(y, y_pred_threshold, sensitive_features=sv)
        res["Equal_Opportunity"] = self.equal_opportunity(y,y_pred_threshold,sv)
        res.update(meta)
        return res

    def equal_opportunity(self,y, y_pred, sv):
        # Calculate true positive rate (TPR) for sensitive and non-sensitive groups
        cm_sensitive = confusion_matrix(y[sv==1], y_pred[sv==1])
        cm_nonsensitive = confusion_matrix(y[sv==0], y_pred[sv==0])
        tpr_sensitive = cm_sensitive[1,1]/(cm_sensitive[1,0]+cm_sensitive[1,1])
        tpr_nonsensitive = cm_nonsensitive[1,1]/(cm_nonsensitive[1,0]+cm_nonsensitive[1,1])
        
        # Calculate difference in TPR between sensitive and non-sensitive groups
        eo_diff = np.abs(tpr_sensitive - tpr_nonsensitive)
        
        return eo_diff
    
    def estimated_metrics(self, y_true, y_pred, sv, weights_dict, meta = {}):
        res = {}
        res['est_acc'] = self.expected_accuracy(y_true, y_pred, weights_dict['weights'])
        res['est_eop'] = self.weq(sv, y_true, y_pred, weights_dict['weights'], weights_dict['p_y_s'])
        res.update(meta)
        return res

    def process_preds(self,pred, threshold = 0.5):
        """
        Processes the predictions to be binary.
        threshold: float

        returns: binary predictions
        """
        #inclusive range is new update vis a vis TPM
        pred  = (pred[~np.isnan(pred)] >= threshold).astype(int)
        return pred

    # def expected_eq(self, df):
    #     # df will contain 
    #     # sv, label, 
    #     weighted_si = (df[df['y_hat'] == 1]).groupby(["sv"]).sum()['w']
    #     counts  = df.groupby(["sv"]).size()
    #     weighted_sum = weighted_si/counts
    #     p_y = calculate_py
    #     res = ((weighted_sum[0]/p_y) - (weighted_sum[1]/p_y))

    def weq(self,sv, y_f,y_hat ,fair_weights, p_y):
        y_hat = (y_hat[~np.isnan(y_hat)] > 0.5).astype(int)

        df = pd.DataFrame()
        df['group_id'] = sv
        df['event'] = pd.Series(y_f).apply(lambda x: f'label={x}')
        df['y_hat'] = y_hat
        df['w'] =  np.where(df['event'] == 'label=1' , fair_weights, 1 - fair_weights)
        # print(f'{df.shape=}')
        # print(f"{df[df['y_hat'] == 1].shape}")
        # /self.tags['weights'] = np.where(self.tags['label'] == 1 , fair_weights, 1 - fair_weights)
        # group_event_tuples = [('label=0', 0), ('label=0', 1) , ('label=1', 0), ('label=1', 1)]
        # group_event_index = pd.MultiIndex.from_tuples(group_event_tuples, names = ['event', 'group_id'])

        weighted_si = (df[df['y_hat'] == 1]).groupby(["event", 'group_id']).sum()['w']
        # weighted_si = pd.Series([weighted_si.loc[0][0], weighted_si.loc[0][1], weighted_si.loc[1][0],  weighted_si.loc[1][1]], index = group_event_index)
        counts  = df.groupby(["group_id"]).size()
        # print(f'{counts=}')
        weighted_sum = weighted_si/(counts * p_y)
        # print(f'{weighted_sum=}')
        # res = ((weighted_sum[0]/p_y) - (weighted_sum[1]/p_y))
        
        tpr_diff = abs(weighted_sum.iloc[3] - weighted_sum.iloc[2])
        fpr_diff = abs(weighted_sum.iloc[0] - weighted_sum.iloc[1])
        # print(f'{(weighted_sum[0]/p_y)=} and {weighted_sum[1]/p_y =}')
        print(f'{tpr_diff=} and {fpr_diff=}')
        return max(tpr_diff, fpr_diff)
        # return res
    
    def expected_accuracy(self, y, y_pred, weights):
        data = pd.DataFrame({'predictions': self.process_preds(y_pred), 'labels': y, 'weights': weights})

        # Filter the rows where predictions match labels
        # matching_rows = data[data['predictions'] == data['labels']]
        return (data[data['predictions'] == 1]['weights'].sum() + (1 - data[data["predictions"] == 0]['weights']).sum())/y_pred.shape[0]