import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class create_wrmsse_metric():
    """
    Calculates wrmsse for validation set and mse for training
    """
    
    def __init__(self,X_val_d,y_val):
        self.X_val_d = X_val_d
        self.days = 28
        sw_df = pd.read_pickle('sw_df.pkl')
        self.sw = sw_df.sw.values
        roll_mat_df = pd.read_pickle('roll_mat_df.pkl')
        self.roll_mat_csr = csr_matrix(roll_mat_df.values)
        del roll_mat_df
        
        self._initialize_days_masks()
        self.y_true = self._pivot_y(y_val)
        
    def _initialize_days_masks(self):
        min_d = self.X_val_d.min()
        self.days_masks = []
        for d in range(min_d,min_d+self.days):
            self.days_masks.append(self.X_val_d==d)
            
    def _pivot_y(self,y):
        y_pivot = np.zeros((30490,28))
        for i,mask_d in enumerate(self.days_masks):
            y_pivot[:,i] = y[mask_d]
        return y_pivot
        
        
    def _rollup(self,v):
        '''
        v - np.array of size (30490 rows, n day columns)
        v_rolledup - array of size (n, 42840)
        '''
        return self.roll_mat_csr*v #(v.T*roll_mat_csr.T).T
    
    def score(self,y_true,preds):
        preds_pivot = self._pivot_y(preds)
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(self._rollup(preds_pivot-self.y_true))
                            ,axis=1)) * self.sw)/12
    
    def eval_wrmsse(self, y_true, preds):
        '''
        preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
        y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
        sequence_length - np.array of size (42840,)
        sales_weight - sales weights based on last 28 days: np.array (42840,)
        '''
        if y_true.shape[0]==30490*28:
            preds_pivot = self._pivot_y(preds)
            score = np.sum(
                    np.sqrt(
                        np.mean(
                            np.square(self._rollup(preds_pivot-self.y_true))
                                ,axis=1)) * self.sw)/12
        else:
            score = ((preds-y_true)**2).sum()/preds.shape[0]
        return 'wrmsse', score, False