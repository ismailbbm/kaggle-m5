from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

import lightgbm as lgb

from model.time_series_kfold import TSKFold
from model.wrmsse import create_wrmsse_metric

class LGBWrapper():
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self,mode='binary'):
        if mode in ['binary','regression','poisson','tweedie']:
            self.mode=mode
        else:
            self.mode=mode
    
    def _get_eval_metric(self,params):
        if params is None:
            if self.mode=='binary':
                return 'logloss'
            else:
                return 'mae'
        elif 'eval_metric' in params.keys():
            return params['eval_metric']

        else:
            if self.mode=='binary':
                return 'logloss'
            else:
                return 'mae'
        
    def _evaluate(self,params,y_actual,y_pred):
        eval_metric = self._get_eval_metric(params)
        if eval_metric=='logloss':
            return metrics.log_loss(y_actual,y_pred)
        elif eval_metric=='mae':
            return metrics.mean_absolute_error(y_actual,y_pred)
        elif eval_metric=='mse':
            return metrics.mean_squared_error(y_actual,y_pred)
        elif eval_metric=='wrmsse':
            return self.wrmsse_metric.wrmsse_simple(y_actual,y_pred)
        
    def fit(self,X,y,X_holdout=None,y_holdout=None,folds=5,params=None,evaluate=True,
            timeseries=False,timeseries_params=None,
            custom_tweedie_params=None):
        #Create folds
        if self.mode=='binary':
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            if timeseries:
                cv = TSKFold(n_splits=folds,list_cutoff_dates=timeseries_params['timeseries_cutoffs'],
                             training_length=timeseries_params['timeseries_training'],validation_length=timeseries_params['timeseries_validation'])
            else:
                cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        
        
        #Custom tweedie
        self.custom_tweedie_params = custom_tweedie_params
        
        #Train models
        self.cv_models = []
        self._oof_idx = []
        self._X = X.copy()
        self.total_oof_loss = 0
        if timeseries:
            self._X_date = self._X['date'].copy()
            self._X.drop(['date','id','d'],axis=1,inplace=True)
            if not X_holdout is None:
                X_holdout = X_holdout.drop(['date','id','d'],axis=1)
        self._y = y
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print("Start training fold {} of {}".format(fold+1,folds))
            self._oof_idx.append((train_idx,val_idx))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if timeseries:
                X_train = X_train.drop(['date'],axis=1)
                X_val = X_val.drop(['date'],axis=1)
                
            if not custom_tweedie_params is None:
                if not custom_tweedie_params['rho'] is None:
                    global rho 
                    rho = custom_tweedie_params['rho'][train_idx]
                if not custom_tweedie_params['weights'] is None:
                    global weights 
                    weights = self.custom_tweedie_params['weights'][self._oof_idx[-1][0]]/self.custom_tweedie_params['weights'][self._oof_idx[-1][0]].mean()
                
            #for M5 only
            print('Columns deletion used for M5')
            if self._get_eval_metric(params)=='wrmsse':
                self.X_val_d = X_val['d'].values
            X_train = X_train.drop(['id','d'],axis=1)
            X_val = X_val.drop(['id','d'],axis=1)
            
            model = self._train(X_train,y_train,X_val,y_val,X_holdout,y_holdout,params)
            self.cv_models.append(model)
            
            if evaluate:
                #Evaluate fold model
                oof_preds = self._predict_proba_model(model,X_val)
                oof_loss = self._evaluate(params,y_val,oof_preds)
                self.total_oof_loss = self.total_oof_loss + oof_loss/folds
                if not X_holdout is None:
                    holdout_preds = self._predict_proba_model(model,X_holdout)
                    holdout_loss = self._evaluate(params,y_holdout,holdout_preds)

                if not X_holdout is None:
                    print('Fold {} out of fold score: {:.4f}; holdout score: {:.4f}'.format(fold+1,oof_loss,holdout_loss))
                else:
                    print('Fold {} out of fold score: {:.4f}'.format(fold+1,oof_loss))
             
        #Evaluate whole model
        if evaluate:
            if not X_holdout is None:
                y_pred_holdout = self.predict(X_holdout)
                holdout_loss = self._evaluate(params,y_holdout,y_pred_holdout)
            if not X_holdout is None:
                print('Total oof score: {:.4f}; holdout score: {:.4f}'.format(self.total_oof_loss,holdout_loss))
            else:
                print('Total oof score: {:.4f}'.format(self.total_oof_loss))
            print("\n")

    def _train(self, X_train, y_train, X_valid, y_valid, X_holdout=None, y_holdout=None, params=None):
        if self.mode=='binary':
            model = lgb.LGBMClassifier()
        else:
            model = lgb.LGBMRegressor(objective=self.mode)
        model = model.set_params(**params)
        
        eval_set = [(X_train, y_train)]
        eval_names = ['train']

        eval_set.append((X_valid, y_valid))
        eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')
        
        if params is None:
            categorical_columns = 'auto'
        elif 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'
            
        eval_metric = self._get_eval_metric(params)
        if eval_metric=='wrmsse':
            print('Initializing wrmsse')
            self.wrmsse_metric = create_wrmsse_metric(self.X_val_d,y_valid)
            eval_metric = self.wrmsse_metric.eval_wrmsse
        
        sample_weight = None
        if (not self.custom_tweedie_params is None):
            if not self.custom_tweedie_params['weights'] is None:
                #sample_weight = np.log1p(self.custom_tweedie_params['weights'][self._oof_idx[-1][0]]/self.custom_tweedie_params['weights'][self._oof_idx[-1][0]].mean())
                sample_weight = self.custom_tweedie_params['weights'][self._oof_idx[-1][0]]/self.custom_tweedie_params['weights'][self._oof_idx[-1][0]].mean()
        
        print('Start training fold')
        model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                       verbose=params['verbose'],
                       categorical_feature=categorical_columns,sample_weight=sample_weight)

        return model

    def _predict_proba_model(self, model, X):
        if model.objective_ == 'binary':
            return model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1]
        else:
            return model.predict(X, num_iteration=model.best_iteration_)
        
    def predict(self,X):
        y = np.zeros((X.shape[0],len(self.cv_models)))
        for i,model in enumerate(self.cv_models):
            y[:,i] = self._predict_proba_model(model,X)
        y = y.mean(axis=1)
        return y
    
    def predict_oof(self):
        y = self._y.copy()
        for (_, val_idx), model in zip(self._oof_idx, self.cv_models):
            y.iloc[val_idx] = self._predict_proba_model(model,self._X.iloc[val_idx])
        return y.values