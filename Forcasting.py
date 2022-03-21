"""
This class include all forcasting methods

"""
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Special function on LSTM
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

# Difference hypothesis test function
def adfuller_test(df):
    result = adfuller(df.dropna())
    if(result[1] > 0.05):
        return False
    else:
        return True

# Data diffirance function
def DataDiff(df,diff_number):
    if(adfuller_test(df) == True):
            return df
    else:
        if(diff_number !=0):
            df = df.diff()
            return DataDiff(df,diff_number-1)
        else:
            return False

# LSTM dateset creater function
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


"""
    This class is used for forcasting the time series data
    params:
        None
"""
class Forecast:
    def __init__(self,data,lag_number):
        self.df = data
        self.lag_number = lag_number
    
    # Prediction function     
    def predict_feature(model,end,start):
        pred_data_test = model.predict(0,(end-start).days)  
        prediction = pd.DataFrame({'y-pred':pred_data_test}).reset_index()
        
        return prediction

    def get_diff(self):
        return DataDiff(self.df,self.lag_number)
    
    """
    --------------------------------------------------------------------------------------------------------------------
    """

    # Auto Arima model
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

    # Manual Arima model
    def arima_model(self,df,p,d,q,P=None,D=None,Q=None):
        if(P is None):
            model = ARIMA(df,order=(p,d,q))
            model_fit =model.fit()
        else:
            model = ARIMA(df,order=(p,d,q), seasonal_order=(P,D,Q,7))
            model_fit =model.fit()

        
        pred_data_test = model_fit.predict(n_periods=len(df))
        prediction = pd.DataFrame({'y-pred':pred_data_test}, index=df.index)
        prediction = prediction.join(df)

        return prediction,model_fit

    def LSTM(df,size,loss,optimizer,epochs,look_back = 1):

        scaler = StandardScaler()

        train,test = df['Adj Close'][:size],df['Adj Close'][size:]
        
        trainT = scaler.fit_transform(train.values.reshape(-1,1))
        testT = scaler.fit_transform(test.values.reshape(-1,1))

        
        trainX, trainY = create_dataset(trainT, look_back)
        testX, testY = create_dataset(testT, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss=loss, optimizer=optimizer)
        model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
        

        # predict dataset
        TrainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # invert predictions
        trainPredict = scaler.inverse_transform(TrainPredict)
        trainY = scaler.inverse_transform([trainY])

        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        trainDATA = pd.DataFrame({'train_pred':trainPredict.flatten(),'Actual':trainY.flatten()})
        testDATA = pd.DataFrame({'test_pred':testPredict.flatten(),'Actual':testY.flatten()})

        
        return trainDATA,testDATA,model

    def RNN(df,train_size,window_size,loss,optimizer,epochs):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        scaler = StandardScaler()

        time_train = df.index[:train_size]
        x_train = df['Adj Close'][:train_size]

        time_valid = df.index[train_size:]
        x_valid = df['Adj Close'][train_size:]
        
        x_train = scaler.fit_transform(x_train.values.reshape(-1,1))
        
        train_set = windowed_dataset(x_train, window_size, batch_size=10, shuffle_buffer=10)
        
        model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape=[None]),
        tf.keras.layers.SimpleRNN(100, return_sequences=True),
        tf.keras.layers.SimpleRNN(40,return_sequences=True),
        tf.keras.layers.Dropout(.6),
        tf.keras.layers.SimpleRNN(60),
        tf.keras.layers.Dense(30, activation = 'relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100.0)
        ])

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10**(epoch / 20))
        model.compile(loss=loss, optimizer=optimizer)

        model.fit(train_set, epochs=epochs, callbacks=[lr_schedule])


        y_pred = model.predict(x_valid)
        x_pred = model.predict(x_train)
        
        pred_df_train = pd.DataFrame({'train_pred':scaler.inverse_transform(x_pred).flatten()}, index=time_train)
        pred_df_test = pd.DataFrame({'test_pred':scaler.inverse_transform(y_pred).flatten()}, index=time_valid)

        return pred_df_test,pred_df_train

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    # evulation metrics staticmethot olarak da tanımlanabilir
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

