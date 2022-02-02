from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA,ARIMAResults
import pmdarima as pm
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import datetime

def adfuller_test(df):
    result = adfuller(df.dropna())
    if(result[1] > 0.05):
        return False
    else:
        return True

def DataDiff(df,diff_number):
    if(adfuller_test(df) == True):
            return df
    else:
        if(diff_number !=0):
            df = df.diff()
            return DataDiff(df,diff_number-1)
        else:
            return False





class Forecast:
    def __init__(self,data,lag_number):
        self.df = data
        self.lag_number = lag_number

    def get_diff(self):
        return DataDiff(self.df,self.lag_number)

    def auto_arima(df,is_sessional):
        model = pm.auto_arima(df, 
                    start_p=1, start_q=1,
                    test='adf',       # use adftest to find optimal 'd'
                    max_p=3, max_q=3, # maximum p and q
                    m=1,              # frequency of series
                    d=None,           # let model determine 'd'
                    seasonal=is_sessional,   #  Seasonality status
                    trace=True,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True)
        
        pred_data_test = model.predict(n_periods=len(df))
        pred_df = pd.DataFrame({'Prediction':pred_data_test},index=df.index,columns=['Prediction'])
        
        return pred_df,model

    def arima_model(self,df,p,d,q,P=None,D=None,Q=None):
        if(P is None):
            model = ARIMA(df,order=(p,d,q))
            model_fit =model.fit(df)
        else:
            model = ARIMA(df,order=(p,d,q), seasonal_order=(P,D,Q,7))
            model_fit =model.fit(df)

        pred_data_test = model_fit.predict(n_periods=len(df))
        prediction = pd.DataFrame({'y-pred':pred_data_test}, index=df.index)
        prediction = prediction.join(df)

        return prediction,model_fit
        
    def predict_feature(model,end,start=datetime.datetime.today()):
        pred_data_test = model.predict(start,end)
        prediction = pd.DataFrame({'y-pred':pred_data_test}, index=pd.date_range(start=start,end=end,freq='D'))
        return prediction 


    def MAPE(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def RMSE(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def R2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    def MAE(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def MSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

