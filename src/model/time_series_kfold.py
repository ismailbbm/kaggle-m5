import datetime

class TSKFold():
    def __init__(self,n_splits,list_cutoff_dates,training_length=3*365,validation_length=28):
        self.n_splits = n_splits
        self.list_cutoff_dates = list_cutoff_dates #last day of training
        self.training_length=training_length
        self.validation_length=validation_length
        if len(list_cutoff_dates)!=n_splits:
            raise ValueError('list_cutoff_dates must have the same length than n_splits')
        
    def split(self,X,y):
        to_return = []
        dates_col = X['date'].values
        max_date = datetime.datetime.strptime(dates_col.max(),"%Y-%m-%d").date()
        min_date = datetime.datetime.strptime(dates_col.min(),"%Y-%m-%d").date()
        for date in self.list_cutoff_dates:
            if date + datetime.timedelta(self.validation_length)>max_date:
                raise ValueError('Cutoff date + validation length after last date available in dataset')
            if date - datetime.timedelta(self.training_length)<min_date:
                raise ValueError('Cutoff date - training length before first date available in dataset')
        for date in self.list_cutoff_dates:
            training_start = datetime.datetime.strftime(date + datetime.timedelta(-self.training_length+1),"%Y-%m-%d")
            training_end = datetime.datetime.strftime(date,"%Y-%m-%d")
            validation_start = datetime.datetime.strftime(date + datetime.timedelta(1),"%Y-%m-%d")
            validation_end = datetime.datetime.strftime(date + datetime.timedelta(self.validation_length),"%Y-%m-%d")
            training_mask = (dates_col>=training_start) & (dates_col<=training_end)
            validation_mask = (dates_col>=validation_start) & (dates_col<=validation_end)
            to_return.append([training_mask,validation_mask])
        return to_return