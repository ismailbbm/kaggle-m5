import datetime
import pandas as pd
import numpy as np

from utils import running_mean, reduce_mem_usage
from data.tweedie import get_tweedie_power
from data.out_of_stock import out_of_stock_zeroes
from sklearn.preprocessing import LabelEncoder

class preprocessData:
    def __init__(self, path_data, test_size=28):
        now = datetime.datetime.now()
        self.path_data = path_data
        self.test_size = test_size

        self._import_data()

        print('calculate scaling factor')
        self._scaling_factor_and_rho()

        print('calculate corrected sales and trend normalization')
        self._create_corrected_sales_and_trend_normalization_df()

        print('unpivot sales, {}s so far'.format((datetime.datetime.now()-now).seconds))
        self._melt_sales_df()

        print('feature engineering calendar')
        self._add_features_event_calendar()

        print('unpivot calendar, {}s so far'.format((datetime.datetime.now()-now).seconds))
        self._melt_calendar_df()

        print('feature engineering snap')
        self._add_features_snap_calendar()

        print("reducing memory")
        self._reduce_memory()

        print("inferring probable missing data in prices, {}s so far".format((datetime.datetime.now()-now).seconds))
        self._infer_missing_prices_df()

        print("feature engineering prices")
        self._add_features_price()

        print('join data')
        self._join_datasets()

        print('add out of stock flag data')
        self._finalize()

        print('done in , {}s so far'.format((datetime.datetime.now()-now).seconds))

    def _import_data(self):
        self.calendar_df = pd.read_csv(self.path_data+'calendar_enriched.csv')
        self.calendar_raw_df = self.calendar_df.copy()
        self.sales_df = pd.read_csv(self.path_data+'sales_train_evaluation.csv')
        self.sales_raw_df = self.sales_df.copy()
        self.prices_df = pd.read_csv(self.path_data+'sell_prices.csv')
        self.sample_df = pd.read_csv(self.path_data+'sample_submission.csv')

    def _scaling_factor_and_rho(self):
        scale_list = []
        rho_list = []
        oos_list = []

        for i, r_raw in self.sales_df.drop(['id','item_id','dept_id','cat_id','store_id','state_id'],axis=1).iterrows():
            r_raw = r_raw.values
            r = r_raw[np.argmax(r_raw != 0):].copy()

            #Scale
            r1 = r[:-1]
            r2 = r[1:]
            r_diff = (r1-r2)**2
            r_sum = r_diff.mean()
            r_sum = np.sqrt(r_sum)
            scale_list.append(r_sum)

            #Rho
            rho = get_tweedie_power(r)
            rho_list.append(rho)

            #out of stock
            oos = out_of_stock_zeroes(r_raw)
            oos_list.append(oos.oos_array)

        self.oos_flag_df = pd.DataFrame(oos_list,columns=self.sales_df.columns[6:])
        self.oos_flag_df = self._add_nan_for_test(self.oos_flag_df)
        self.oos_flag_df['id'] = self.sales_df['id']
        self.sales_df['scale'] = scale_list
        self.sales_df['rho'] = rho_list

    def _add_nan_for_test(self,df,trail=0):
        max_d = int(df.columns[-1-trail][2:])
        a = np.empty(df.shape[0])
        a[:] = np.nan

        for i in range(self.test_size):
            df['d_'+str(max_d+1+i)] = a

        return df


    def _fill_zeros_with_last(self,arr):
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        return arr[prev]

    def _create_corrected_sales_and_trend_normalization_df(self):
        N = 28
        list_corrected_sales = []
        norm_factor = []

        for i in range(self.oos_flag_df.shape[0]):
            oos = self.oos_flag_df.iloc[i,:-1-self.test_size].values.astype(int)
            sales = self.sales_raw_df.iloc[i,6:].values.astype(int)

            sales_moving_avg = running_mean(sales,N)

            #Arrays without trailing zeros
            oos_without_beg = oos[np.argmax(sales!=0):]
            sales_mov_avg_without_beg = sales_moving_avg[np.argmax(sales!=0):]

            #Replace mov av sales with zero for timestamp where oos
            sales_mov_avg_without_beg = sales_mov_avg_without_beg*(1-oos_without_beg)
            #Fill zeroes with last know non zero value
            sales_mov_avg_without_beg = self._fill_zeros_with_last(sales_mov_avg_without_beg)
            #Create new sales by replacing wherever oos by the moving average
            sales_corrected = np.concatenate([sales[:np.argmax(sales!=0)],np.where(oos_without_beg==0,sales[np.argmax(sales!=0):],sales_mov_avg_without_beg)])
            list_corrected_sales.append(sales_corrected)
            
            #Normalization (has 28 more values than sales_df to be able to use it for inference)
            sales_mov_avg_corrected = np.concatenate([sales[:np.argmax(sales!=0)],sales_mov_avg_without_beg])
            sales_mov_avg_corrected = np.concatenate([sales_mov_avg_corrected[0]*np.ones(28),sales_mov_avg_corrected])
            normalization = np.where(sales_mov_avg_corrected==0,1,1/sales_mov_avg_corrected)
            norm_factor.append(normalization) 
                
            
        self.corrected_sales_df = pd.DataFrame(list_corrected_sales,columns=self.sales_raw_df.columns[6:])
        self.corrected_sales_df = self._add_nan_for_test(self.corrected_sales_df)
        self.corrected_sales_df['id'] = self.sales_df['id']
        
        self.normalization_factor_df = pd.DataFrame(norm_factor,columns=['d_'+str(i+1) for i in range(norm_factor[0].shape[0])])
        self.normalization_factor_df = self._add_nan_for_test(self.normalization_factor_df)
        self.normalization_factor_df['id'] = self.sales_df['id']

    def _melt_sales_df(self):

        self.sales_df = self._add_nan_for_test(self.sales_df,2)
        
        all_cols = self.sales_df.columns
        all_cols_mod = [int(x[2:]) if 'd_' in x else x for x in all_cols]
        self.sales_df.columns = all_cols_mod
        id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id','scale','rho']
        value_vars = list(set(all_cols_mod)-set(id_vars))
        self.sales_df = pd.melt(self.sales_df, id_vars=id_vars, value_vars=value_vars)
        self.sales_df.rename(columns={'variable':'d','value':'sales'},inplace=True)


    def _add_features_event_calendar(self):
        le = LabelEncoder()
        self.calendar_df['event_name_1'] = self.calendar_df['event_name_1'].fillna('No')
        self.calendar_df['event_name_1'] = le.fit_transform(self.calendar_df['event_name_1'])
        no_value = le.fit_transform(['No'])[0]
        self.calendar_df['event_name_1'] = self.calendar_df['event_name_1'].astype('int8')
        self.calendar_df['1_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-1).astype('float16').fillna(no_value)
        self.calendar_df['2_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-2).astype('float16').fillna(no_value)
        self.calendar_df['3_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-3).astype('float16').fillna(no_value)
        self.calendar_df['4_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-4).astype('float16').fillna(no_value)
        self.calendar_df['5_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-5).astype('float16').fillna(no_value)
        self.calendar_df['6_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-6).astype('float16').fillna(no_value)
        self.calendar_df['7_day_prior_event_name_1'] = self.calendar_df['event_name_1'].shift(-7).astype('float16').fillna(no_value)
        self.calendar_df['1_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(1).astype('float16').fillna(no_value)
        self.calendar_df['2_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(2).astype('float16').fillna(no_value)
        self.calendar_df['3_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(3).astype('float16').fillna(no_value)
        self.calendar_df['4_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(4).astype('float16').fillna(no_value)
        self.calendar_df['5_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(5).astype('float16').fillna(no_value)
        self.calendar_df['6_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(6).astype('float16').fillna(no_value)
        self.calendar_df['7_day_after_event_name_1'] = self.calendar_df['event_name_1'].shift(7).astype('float16').fillna(no_value)

    def _melt_calendar_df(self):
        self.calendar_df.rename(columns={'snap_CA':'CA','snap_TX':'TX','snap_WI':'WI'},inplace=True)
        self.calendar_df['d'] = self.calendar_df['d'].apply(lambda d: int(d[2:]))
        all_cols = self.calendar_df.columns
        id_vars = ['date','wm_yr_wk','weekday','wday','month','year','d','event_name_1','event_type_1','event_name_2','event_type_2',
                  '1_day_prior_event_name_1','2_day_prior_event_name_1','3_day_prior_event_name_1','4_day_prior_event_name_1','5_day_prior_event_name_1','6_day_prior_event_name_1','7_day_prior_event_name_1',
                   '1_day_after_event_name_1','2_day_after_event_name_1','3_day_after_event_name_1','4_day_after_event_name_1','5_day_after_event_name_1','6_day_after_event_name_1','7_day_after_event_name_1']
        value_vars = list(set(all_cols)-set(id_vars)) 
        self.calendar_df = pd.melt(self.calendar_df, id_vars=id_vars, value_vars=value_vars)
        self.calendar_df.rename(columns={'variable':'state_id','value':'snap'},inplace=True)


    def _add_features_snap_calendar(self):
        CA_df = self.calendar_df[self.calendar_df['state_id']=='CA'].copy()
        TX_df = self.calendar_df[self.calendar_df['state_id']=='TX'].copy()
        WI_df = self.calendar_df[self.calendar_df['state_id']=='WI'].copy()
        
        for df in [CA_df,TX_df,WI_df]:
            df.sort_values(by=['d'],ascending=True,inplace=True)
            df['1_day_prior_snap'] = df['snap'].shift(-1).fillna(0)
            df['2_day_prior_snap'] = df['snap'].shift(-2).fillna(0)
            df['3_day_prior_snap'] = df['snap'].shift(-3).fillna(0)
            df['4_day_prior_snap'] = df['snap'].shift(-4).fillna(0)
            df['5_day_prior_snap'] = df['snap'].shift(-5).fillna(0)
            df['6_day_prior_snap'] = df['snap'].shift(-6).fillna(0)
            df['7_day_prior_snap'] = df['snap'].shift(-7).fillna(0)
            df['1_day_after_snap'] = df['snap'].shift(1).fillna(0)
            df['2_day_after_snap'] = df['snap'].shift(2).fillna(0)
            df['3_day_after_snap'] = df['snap'].shift(3).fillna(0)
            df['4_day_after_snap'] = df['snap'].shift(4).fillna(0)
            df['5_day_after_snap'] = df['snap'].shift(5).fillna(0)
            df['6_day_after_snap'] = df['snap'].shift(6).fillna(0)
            df['7_day_after_snap'] = df['snap'].shift(7).fillna(0)
        
        self.calendar_df = pd.concat([CA_df,TX_df,WI_df],axis=0)

    def _reduce_memory(self):
        self.calendar_df = reduce_mem_usage(self.calendar_df)
        self.sales_df = reduce_mem_usage(self.sales_df)
        self.prices_df = reduce_mem_usage(self.prices_df)

    def _infer_missing_prices_df(self):
        max_date = self.prices_df['wm_yr_wk'].max()
        min_date = self.prices_df['wm_yr_wk'].min()

        store_item_min_df = self.prices_df.sort_values(by=['wm_yr_wk']).drop_duplicates(subset=['item_id','store_id'],keep='first')
        store_item_min_df = store_item_min_df[store_item_min_df['wm_yr_wk']!=min_date].copy()

        store_item_max_df = self.prices_df.sort_values(by=['wm_yr_wk']).drop_duplicates(subset=['item_id','store_id'],keep='last')
        store_item_max_df = store_item_max_df[store_item_max_df['wm_yr_wk']!=max_date].copy()

        list_date = list(set(self.prices_df['wm_yr_wk'].values.tolist()))
        list_date.sort()
        list_missing = []

        for index,r in store_item_min_df.iterrows():
            store_id = r['store_id']
            item_id = r['item_id']
            wm_yr_wk = r['wm_yr_wk']
            sell_price = r['sell_price']
            wm_yr_wk_previous = list_date[list_date.index(wm_yr_wk)-1]
            list_missing.append([store_id,item_id,wm_yr_wk_previous,sell_price])
            if wm_yr_wk_previous!=min_date:
                wm_yr_wk_previous = list_date[list_date.index(wm_yr_wk_previous)-1]
                list_missing.append([store_id,item_id,wm_yr_wk_previous,sell_price])
                
        for index,r in store_item_max_df.iterrows():
            store_id = r['store_id']
            item_id = r['item_id']
            wm_yr_wk = r['wm_yr_wk']
            sell_price = r['sell_price']
            wm_yr_wk_next = list_date[list_date.index(wm_yr_wk)-1]
            list_missing.append([store_id,item_id,wm_yr_wk_next,sell_price])
            if wm_yr_wk_next!=max_date:
                wm_yr_wk_next = list_date[list_date.index(wm_yr_wk_next)-1]
                list_missing.append([store_id,item_id,wm_yr_wk_next,sell_price])
                
        missing_df = pd.DataFrame(list_missing,columns=['store_id','item_id','wm_yr_wk','sell_price'])
        self.prices_df = pd.concat([self.prices_df,missing_df])


    def _add_features_price(self):
        price_agg = self.prices_df.groupby(['store_id','item_id'],as_index=False).agg(
            {
                'sell_price':'mean'
            }
        )
        price_agg.rename(columns={'sell_price':'avg_sell_price'},inplace=True)
        self.prices_df = pd.merge(self.prices_df,price_agg,on=['store_id','item_id'])
        self.prices_df['sell_price_norm'] = self.prices_df['sell_price']/self.prices_df['avg_sell_price']


        self.prices_df['sell_price_norm_lag'] = self.prices_df.groupby(['store_id','item_id'])["sell_price_norm"].transform(
                        lambda x: x.shift(1)
                    )
        self.prices_df['sell_price_norm_momentum'] = self.prices_df['sell_price_norm_lag']/self.prices_df['sell_price_norm']

        self.prices_df.drop(['avg_sell_price','sell_price_norm_lag'],inplace=True,axis=1)


    def _join_datasets(self):
        self.all_data_preprocessed = pd.merge(
            self.sales_df, self.calendar_df, on=['d','state_id'],how='left'
        )
        self.all_data_preprocessed = pd.merge(
            self.all_data_preprocessed, self.prices_df, on=['wm_yr_wk','item_id','store_id'], how='left'
        )
        
    def _finalize(self):
        self.all_data_preprocessed.sort_values(by=['id','d'],inplace=True)
        
        #Melt out of stock
        all_cols = self.oos_flag_df.columns
        all_cols_mod = [int(x[2:]) if 'd_' in x else x for x in all_cols]
        self.oos_flag_df.columns = all_cols_mod
        id_vars = ['id']
        value_vars = list(set(all_cols_mod)-set(id_vars))
        oos_flag_pivot_df = pd.melt(self.oos_flag_df, id_vars=id_vars, value_vars=value_vars)
        oos_flag_pivot_df.rename(columns={'variable':'d','value':'oos'},inplace=True)
        oos_flag_pivot_df.sort_values(by=['id','d'],inplace=True)
        self.all_data_preprocessed['oos'] = oos_flag_pivot_df['oos'].values
        
        #Melt sales corrected
        all_cols = self.corrected_sales_df.columns
        all_cols_mod = [int(x[2:]) if 'd_' in x else x for x in all_cols]
        self.corrected_sales_df.columns = all_cols_mod
        id_vars = ['id']
        value_vars = list(set(all_cols_mod)-set(id_vars))
        corrected_sales_pivot_df = pd.melt(self.corrected_sales_df, id_vars=id_vars, value_vars=value_vars)
        corrected_sales_pivot_df.rename(columns={'variable':'d','value':'sales_corrected'},inplace=True)
        corrected_sales_pivot_df.sort_values(by=['id','d'],inplace=True)
        self.all_data_preprocessed['sales_corrected'] = corrected_sales_pivot_df['sales_corrected'].values
        
        #Test vs train
        max_train_d = int(self.sales_raw_df.columns[-1][2:])
        d_list = self.all_data_preprocessed['d'].values
        is_test = np.where(d_list>max_train_d,1,0)
        self.all_data_preprocessed['is_test'] = is_test