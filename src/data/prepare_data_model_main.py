import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime

from utils import reduce_mem_usage

class prepareDataModel:

    def __init__(self, df, sample_df, price_weight_simulation=True, price_scale_correction=True):
        self.df = df.copy()
        self.sample_df = sample_df
        self.price_weight_simulation = price_weight_simulation
        self.price_scale_correction = price_scale_correction
        self._prepare_data(price_weight_simulation,price_scale_correction)
        
    def _label_encode(self,df,cat_cols):
        print('Encoding labels')
        self.le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.le_dict[col]=le
        return df
    
    def _add_rolling_values(self,lag_list,window_weekday,window_last_year,window_past_days):
        print('Calulating roll values')
        #Rolling values based on same weekday
        self.df.sort_values(by=['id','wday','wm_yr_wk'],inplace=True)
        
        df_tmp = self.df[['id','wday','wm_yr_wk','d','sales']].copy()
        now = datetime.datetime.now()
        for lag in lag_list:
            for window in window_weekday:
                self.df[f'lag_{str(lag)}_last_{window}_weeks_same_weekday'] = df_tmp.groupby(["id",'wday'])["sales"].transform(
                    lambda x: x.shift(lag).rolling(window).mean()
                )
    #             self.df[f'last_{window}_weeks_same_weekday_std'] = df_tmp.groupby(["id",'wday'])["sales"].transform(
    #                 lambda x: x.shift(n_preds_week).rolling(window).std()
    #             )
            print("Time for week days roll: {}s".format((datetime.datetime.now()-now).seconds))
        
        now = datetime.datetime.now()
        for window in window_last_year:
            self.df[f'last_year_{2*window+1}_weeks_same_weekday'] = df_tmp.groupby(["id",'wday'])["sales"].transform(
                lambda x: x.shift(52-2*(window-1)).rolling(2*window+1).mean().astype(np.float16)
            )
#             self.df[f'last_year_{2*window+1}_weeks_same_weekday_std'] = df_tmp.groupby(["id",'wday'])["sales"].transform(
#                 lambda x: x.shift(52-2*(window-1)).rolling(2*window+1).std()
#             )
        print("Time for week days last year roll: {}s".format((datetime.datetime.now()-now).seconds))

        #Rolling values based on past days
        now = datetime.datetime.now()
        df_tmp.sort_values(by=['id','d'],inplace=True)
        self.df.sort_values(by=['id','d'],inplace=True)
        for lag in lag_list:
            for window in window_past_days:
                self.df[f'lag_{str(lag)}_last_{window}_days'] = self.df.groupby(["id"])["sales"].transform(
                    lambda x: x.shift(lag*7).rolling(window).mean().astype(np.float16)
                )
#                 self.df[f'lag_{str(lag)}_last_{window}_days_oos'] = self.df.groupby(["id"])["oos"].transform(
#                     lambda x: x.shift(lag*7).rolling(window).mean().astype(np.float16)
#                 )
    #             self.df[f'last_{window}_days_std'] = df_tmp.groupby(["id"])["sales"].transform(
    #                 lambda x: x.shift(n_preds_week*7).rolling(window).std()
    #             )
    #             self.df[f'last_{window}_days_kurt'] = df_tmp.groupby(["id"])["sales"].transform(
    #                 lambda x: x.shift(n_preds_week*7).rolling(window).kurt()
    #             )
            print("Time for previous days roll: {}s".format((datetime.datetime.now()-now).seconds))
        
        del df_tmp
    
    def _prepare_data(self, price_weight_simulation=True, price_scale_correction=True):
        cat_cols_to_keep = ['item_id','dept_id','cat_id','store_id','state_id','wday','month']
        events_col = ['event_name_1','snap','1_day_prior_event_name_1','2_day_prior_event_name_1','3_day_prior_event_name_1','4_day_prior_event_name_1','5_day_prior_event_name_1','6_day_prior_event_name_1','7_day_prior_event_name_1',
                      '1_day_after_event_name_1','2_day_after_event_name_1','3_day_after_event_name_1','4_day_after_event_name_1','5_day_after_event_name_1','6_day_after_event_name_1','7_day_after_event_name_1',
                      '1_day_prior_snap','2_day_prior_snap','3_day_prior_snap','4_day_prior_snap','5_day_prior_snap','6_day_prior_snap','7_day_prior_snap',
                       '1_day_after_snap','2_day_after_snap','3_day_after_snap','4_day_after_snap','5_day_after_snap','6_day_after_snap','7_day_after_snap'
                     ]
        
        cols_to_keep = cat_cols_to_keep + ['id','d','sell_price','sell_price_norm_momentum','date','oos']+events_col
        
        #Add rolling values
        self.lag_list = [1,2,3,4]
        self.window_weekday = [1,4,8]
        #self.window_weekday = []
        self.window_last_year = []
        self.window_past_days = [7,14,28,56,112]
        #self.window_past_days = []
        self._add_rolling_values(self.lag_list,self.window_weekday,self.window_last_year,self.window_past_days)
        
        for lag in self.lag_list:
            for window in self.window_weekday:
                cols_to_keep.append(f'lag_{str(lag)}_last_{window}_weeks_same_weekday')
#                 cols_to_keep.append(f'last_{window}_weeks_same_weekday_std')
            for window in self.window_last_year:
                cols_to_keep.append(f'last_year_{2*window+1}_weeks_same_weekday')
#                 cols_to_keep.append(f'last_year_{2*window+1}_weeks_same_weekday_std')
            for window in self.window_past_days:
                cols_to_keep.append(f'lag_{str(lag)}_last_{window}_days')
#                 cols_to_keep.append(f'lag_{str(lag)}_last_{window}_days_oos')
#                 cols_to_keep.append(f'last_{window}_days_std')
#                 cols_to_keep.append(f'last_{window}_days_kurt')
        
        cols_to_keep.append('scale')
        cols_to_keep.append('rho')
        
        print("Replace values not to predict with zero for target and release_d=0")
        #self.df = self.df[(~self.df['sell_price'].isnull())].copy()
        mask_no_prices = self.df['sell_price'].isnull().values
        self.df.loc[mask_no_prices,'sales'] = 0
        self.df.loc[mask_no_prices,'oos'] = 0
        
        self.df = self._label_encode(self.df,cat_cols_to_keep)
        self.df[events_col] = self.df[events_col].fillna(-1)
        self.df[events_col] = self.df[events_col].astype('int8')
        
        print("Reducing memory")
        self.df = reduce_mem_usage(self.df)
        
        print("Create Train Val")
        self._create_train_val(cols_to_keep,price_weight_simulation,price_scale_correction)
    
    def _create_train_val(self,cols_to_keep,price_weight_simulation,price_scale_correction):
        train_df = self.df[(self.df['d']<1913+28+1)]
        val_df = self.df[(self.df['d']>=1913+1) & (self.df['d']<1913+28+1)]
        test_df = self.df[(self.df['d']>=1913+28+1)]

        self.X_train = train_df[cols_to_keep].copy()
        self.y_train = train_df['sales'].copy()

        self.X_val = val_df[cols_to_keep].copy()
        self.y_val = val_df[['id','d','sell_price','scale','sales']].copy()

        self.X_test = test_df[cols_to_keep].copy()
        self.y_test = test_df[['id','d','sell_price','scale','sales']].copy()

        self.X_train_scale = self.X_train['scale'].values
        self.X_val_scale = self.X_val['scale'].values
        self.X_test_scale = self.X_test['scale'].values
        
        self.X_train_rho = self.X_train['rho'].values
        self.X_val_rho = self.X_val['rho'].values
        self.X_test_rho = self.X_test['rho'].values
    
        
        self.X_train_weight = self.X_train['sell_price'].values/self.X_train_scale
        self.X_train_weight[np.isnan(self.X_train_weight)] = 0
        self.X_val_weight = self.X_val['sell_price'].values/self.X_val_scale
        self.X_val_weight[np.isnan(self.X_val_weight)] = 0
        self.X_test_weight = self.X_test['sell_price'].values/self.X_test_scale
        self.X_test_weight[np.isnan(self.X_test_weight)] = 0
        
        if price_weight_simulation:
            self.y_train = self.y_train * self.X_train['sell_price']
            self.y_val['sales'] = self.y_val['sales'] * self.y_val['sell_price']
            self.y_test['sales'] = self.y_test['sales'] * self.y_test['sell_price']
            
        if price_scale_correction:
            self.y_train = self.y_train / self.X_train['scale'].values
            self.y_val['sales'] = self.y_val['sales'] / self.X_val['scale'].values
            self.y_test['sales'] = self.y_test['sales'] / self.X_test['scale'].values
            
        self.X_train.drop(['scale','rho'],axis=1,inplace=True)
        self.X_val.drop(['scale','rho'],axis=1,inplace=True)
        self.X_test.drop(['scale','rho'],axis=1,inplace=True)
        
    def _sort_according_to_sample(self):
        self.df = pd.merge(self.df,self.sample_df[['id']],on=['id'])