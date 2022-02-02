from glob import glob
import re
from turtle import color
from marshmallow import pre_dump
import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from  Collection import Data
from Forcasting import Forecast
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
     page_title="Stok Price Analyis App",
     page_icon="💰",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get Help': 'https://github.com/burakugurr',
         'Report a bug': "https://github.com/burakugurr",
         'About': "# Stok Price Analyis App created by Burak Uğur",
     }
 )
@st.cache
def load_data(s_w,start,end):
    data = Data(s_w,start,end)
    df = data.get_data()
    df = data.tranform_data(df)
    df.to_csv("data.csv")
    return df


# Graph Functions

def timeSeries(df):
    
    fig = px.line(df, x=df.index, y="Adj Close", width=1080,height=720,labels={'Adj Close':'Price'},
    template="seaborn",color_discrete_sequence=["#0066ff"])
    return fig

def plotpred(df,pred):
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.plot(df.index,df['Adj Close'].values,label='Actual',color='#0066ff')
    ax.plot(pred.index,pred['y-pred'].values,label='Predict',color='#ff0000')
    ax.legend(["Actual","Predict"])
    return fig


def plot_feature_pred(df,preddata):
    fig, ax = plt.subplots()
    ax.plot(df.index,df['Adj Close'].values,label='Actual',color='#0066ff')
    ax.plot(preddata.index,preddata['y-pred'].values,label='Predict',color='#ff0000')





def evulation(method_name,y_pred,y_true):

    if(method_name=="MAPE"):
        return Forecast.MAPE(y_true,y_pred).round(2)
        
    if(method_name=="RMSE"):
        return Forecast.RMSE(y_true,y_pred).round(2)
    if(method_name=="MAE"):
        return Forecast.MAE(y_true,y_pred).round(2)
    if(method_name=="MSE"):
        return Forecast.MSE(y_true,y_pred).round(2)
    if(method_name=="R2"):
        return Forecast.R2(y_true,y_pred).round(2)
    else:
        return st.error("Please select valid evulation metrics")










 
# Layout Functions IndexError

def homepage():
    st.title("Welcome to Stock App")
    st.subheader("Stock App is a web app that can help you to predict stock prices and anomaly detection")
    st.markdown(""" Please select a stock from the dropdown menu below to see the stock price result""")
    col1, col2 = st.columns(2)
    s_box = col1.selectbox("Select a stock", ("AAPL", "MSFT", "AMZN", "GOOG", "FB", "NFLX","Other"))
    if s_box == "Other":
        s_w = col1.text_input("Please type a stock name")
        
    start = col2.date_input("Start Date", value=datetime.date(2000, 1, 1),max_value=datetime.date.today())
    end = col2.date_input("End Date", value=datetime.datetime.now(),max_value=datetime.date.today())


    if(st.button("Get Stock Price")):
            global df,data
            if(s_box == "Other"):
                if(s_w != ""):
                    try:
                        df = load_data(s_w,start,end)
                        st.success("Data is loaded successfully")
                        st.subheader("Stock Price on "+str(s_w)+" between "+str(start)+" and "+str(end))
                        st.plotly_chart(timeSeries(df))

                    except:
                        st.error("Please type a valid stock name")
                else:
                    st.error("Please type a stock name")
            else:
                df = load_data(s_box.upper(),start,end)
                st.success("Data is loaded successfully")
                st.subheader("Stock Price on "+str(s_box)+" between "+str(start)+" and "+str(end))
                st.plotly_chart(timeSeries(df))

                
    

def forecasting():
    # try:
    st.title("Forcecasting")
    df = pd.read_csv("data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index("Date",inplace=True)
    st.header("Stock Price Forecasting")
    diffnum = st.sidebar.slider("Select a discrete difference of data",0,30,1)
    lagnum = st.sidebar.slider("Select lagg discrete difference of data",0,100,1)
    
    fc = Forecast(df['Adj Close'],diffnum)
    df_diff = fc.get_diff()
    if(df_diff is False):
        st.error("Please select valid discrete difference of data")
    else:
        col1,col2 = st.columns(2)
        col1.pyplot(plot_acf(df_diff.dropna(),lags=lagnum,title="Stock Price ACF"),color="green",vlines_kwargs={"color": "green"})
        col2.pyplot(plot_pacf(df_diff.dropna(),lags=lagnum,title="Stock Price PACF"))
    
    st.subheader("Forecast Method")
    method = st.sidebar.selectbox("Select a method",("Box-Jenkins","Neural Network"))
    if(method == "Box-Jenkins"):
        st.subheader("ARIMA Model Result")
        st.sidebar.caption("Plase select a model parameters")
        issesion = st.sidebar.checkbox("Is Time Series Sessional ?")
        n_size = st.sidebar.slider("Select a size of training data",1,len(df),5)

        p = st.sidebar.number_input("Select a p value",0,3,1)
        d = st.sidebar.number_input("Select a d value",0,3,1)
        q = st.sidebar.number_input("Select a q value",0,3,1)

        
        method_name = st.sidebar.multiselect("Select a evulation metrics",["MAPE","RMSE","MAE","MSE","R2"],default=["MAPE"])
                
        

        if(issesion):
            P = st.sidebar.number_input("Select a P value",0,3,1)
            D = st.sidebar.number_input("Select a D value",0,3,1)
            Q = st.sidebar.number_input("Select a Q value",0,3,1)
            if st.sidebar.button("Forecast"):
                pred, model = fc.arima_model(df['Adj Close'][:n_size],int(p),int(d),int(q),int(P),int(D),int(Q))
                col1,col2 = st.columns(2)
                col1.write(pred)
                """

                GELŞTİRME DEVAM EDECEK

                """

                col2.pyplot(plotpred(df[:n_size],pred))



                
        # Sessional olmadığı zaman
        else:
            
            if st.sidebar.button("Forecast"): # buton
                pred, predmodel = fc.arima_model(df['Adj Close'][:n_size],int(p),int(d),int(q),P=None,D=None,Q=None)
                col1,col2 = st.columns(2)
                col1.write(pred)

                col2.pyplot(plotpred(df[:n_size],pred))
                st.subheader("Forecast Evaluation")

                valueList = []
                metricsList = []
                for i in method_name:
                
                    values = evulation(i,pred['y-pred'],pred['Adj Close'])             
                    metricsList.append(i)
                    valueList.append(values)

                result = dict(zip(metricsList, valueList))
                col1, col2, col3,col4,col5 = st.columns(5)
                try:
                    for i in range(len(result)):
                        col1.metric(list(result.keys())[i],list(result.values())[i])
                        col2.metric(list(result.keys())[i+1],list(result.values())[i+1])
                        col3.metric(list(result.keys())[i+2],list(result.values())[i+2])
                        col4.metric(list(result.keys())[i+3],list(result.values())[i+3])
                        col5.metric(list(result.keys())[i+4],list(result.values())[i+4])
                        break
                except:
                    pass
                
                pred_feature = st.checkbox("Forecast Feature")
            if(pred_feature):
                date_pred_start = st.sidebar.date_input("Select a date to predict",min_value =df.index.min())
                date_pred_end = st.sidebar.date_input("Select a date to predict")
                pred = Forecast.predict_feature(model=predmodel,start=date_pred_start,end=date_pred_end)
                col1,col2 = st.columns(2)
                st.subheader("Prediction Feature")
                col1,col2 = st.columns(2)
                col1.write(pred)
                #col2.pyplot(plot_feature_pred(df[:n_size],pred))





    

"""
https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/#:~:text=ARIMA%2C%20short%20for%20'AutoRegressive%20Integrated,to%20predict%20the%20future%20values.
https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
"""














def force_layout():
    st.markdown("""
    #### About
    This is a web app that can help you to predict stock prices and anomaly detection.
    It can be used for any stock.
    """)
    st.markdown("""
    #### Features
    1. You can see the stock price graph
    2. You can see the stock price anomaly
    3. You can see the stock price forecast
    """)
    st.markdown("""
    #### How to use
    1. Select a stock from the dropdown menu below to see the stock price result
    2. You can type a stock name if you select "Other"
    3. You can select a start date and end date
    4. You can see the stock price anomaly
    5. You can see the stock price anomaly with the original graph
    """)
    st.markdown("""
    #### Developed by
    Burak Uğur
    """)
    st.markdown("""
    #### Contact
    >   Email: burak.uugur12@gmail.com

    >   Github: @burakugurr """)

    





# Main Function





app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home Page","Forecasting", "Anomaly Detection","About"])


if app_mode == "Home Page":
    homepage()
elif app_mode == "Forecasting":
    forecasting()
elif app_mode =='Anomaly Detection':
    anomaly()
elif app_mode == "About":
    force_layout()