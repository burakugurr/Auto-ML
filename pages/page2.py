# include the methods in the module

from time import time
import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sys
from Forcasting import Forecast
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from AnomalyDetection import Detector as ad
import DbConnection as dbc
from PIL import Image

import warnings
warnings.filterwarnings('ignore')



"""
    Returns the dataframe of the stock
"""
@st.cache
def load_data(s_w,start,end):
    data = Data(s_w,start,end)
    df = data.get_data()
    df = data.tranform_data(df)
    df.to_csv("data.csv")
    return df


"""
  This functions is used to plot the graph of the stock
"""

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

def plot_feature_pred(preddata):
    fig = px.line(preddata, x='Date', y="y-pred", width=1080,height=720,labels={'y-pred':'$'},
    template="seaborn",color_discrete_sequence=["#0066ff"])
    return fig

def plot_nn(train,test,df,title):
    fig = go.Figure([
        go.Scatter(
            name='Train Pred',
            x=df.index[:len(train)],
            y=train['train_pred'],
            mode='markers+lines',
            marker=dict(color='red', size=3),
            showlegend=True
        ),
        go.Scatter(
            name='Test Pred',
            x=df.index[len(train):],
            y=test['test_pred'],
            mode='markers+lines',
            marker=dict(color="darkblue", size=3),
            line=dict(width=2),
            showlegend=True
        ),
        go.Scatter(
            name='Actual',
            x=df.index,
            y=df['Adj Close'],
            marker=dict(color="orange"),
            line=dict(width=3),
            mode='markers+lines',
            showlegend=True
        )
    ])
    fig.update_layout(
        yaxis_title='Stock Price ($/Day)',
        xaxis_title='Date',
        title=title,
        hovermode="x",
        legend_title="Legend",
        plot_bgcolor="LightSteelBlue",
        width=1080,
        height=720,
    )
    return fig

def plot_nn_train(traindata):
    fig = make_subplots(rows=1, cols=2)
    fig.append_trace(
        go.Scatter(x=traindata['train_pred'].index,
                    y=traindata['train_pred'],
                    name='Train_Pred',
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True) , 1, 1)
    fig.append_trace(
        go.Scatter(x=traindata['train_pred'].index,
                    y=traindata['Actual'],
                    name='Train_Actual',
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True) , 1, 2)
    fig.update_layout(
        title="Train Data Result",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        legend=dict(x=1, y=1),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig

def plot_nn_test(testdata):
    fig = make_subplots(rows=1, cols=2)

    fig.append_trace(
        go.Scatter(x=testdata['test_pred'].index,
                    y=testdata['test_pred'],
                    name='Test_Pred',
                    mode='markers+lines',
                    marker=dict(color='purple', size=2),
                    showlegend=True) , 1, 1)

    fig.append_trace(
        go.Scatter(x=testdata['test_pred'].index,
                    y=testdata['Actual'],
                    name='Test_Actual',
                    mode='markers+lines',
                    marker=dict(color='cyan', size=2),
                    showlegend=True) , 1, 2)

    fig.update_layout(
        title="Test Data Result",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        legend=dict(x=1, y=1),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig

"""
    This function is used to evaluate the model
    params:
        method: the method used to evaluate the model
        y_pred: the predicted value
        y_true: the actual value
    return:
        the evaluation result
"""
def evulation(method_name,y_pred,y_true):

    if(method_name=="MAPE"):
        return Forecast.MAPE(y_true,y_pred).round(3)
        
    if(method_name=="RMSE"):
        return Forecast.RMSE(y_true,y_pred).round(3)
    if(method_name=="MAE"):
        return Forecast.MAE(y_true,y_pred).round(3)
    if(method_name=="MSE"):
        return Forecast.MSE(y_true,y_pred).round(3)
    if(method_name=="R2"):
        return Forecast.R2(y_true,y_pred).round(3)
    else:
        return st.error("Please select valid evulation metrics")

"""
    Home Page: the main page of the app
    welcome page
    Params:
        username: the username of the user
    
"""

def homepage(username):
    
    image = Image.open('images\Stockred.png')
    st.title("Stockred")
    
    
    col1,col2 = st.columns(2)
    col1.image(image)
    col2.markdown("## Welcome back *%s*"% (username))
    col2.markdown("**Stock App is a web app that can help you to predict stock prices and anomaly detection.**")
    col2.write("""
        To continue, please select the data insight page from the sidebar.
    """)
    
 
"""
    Data insgiht: the data collection page of the app
    You can select the data and visualize the data. If you want to predict the stock price, you can save the data.
    
"""

def datainsgiht():
    st.markdown("> ❗ **Save a data between two dates before starting the app.**")
    st.markdown(""" Please select a stock from the dropdown menu below to see the stock price result""")
    col1, col2 = st.columns(2)
    s_box = col1.selectbox("Select a stock", ("AAPL", "MSFT", "AMZN", "GOOG", "FB", "NFLX","Other"))
    if s_box == "Other":
        s_w = col1.text_input("Please type a stock name")
        
    start = col2.date_input("Start Date", value=datetime.date(2000, 1, 1),max_value=datetime.date.today())
    end = col2.date_input("End Date", value=datetime.datetime.now(),max_value=datetime.date.today())
    
    getstock = col1.button("Get Stock Price")
    savedata = col1.button("Save Data")

    if(getstock):
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

        st.info("To continue the application, you need to save the data")
        if (savedata):
            with st.spinner('Plase wait'):
                time.sleep(2)
                df.to_csv("data.csv")
                
            st.success("Data is saved successfully")
              

"""
    forecast page: the page that can predict the stock price. You can select the two method(Box-Jenkins or Neural Network) to predict the stock price.
    Params:
        none
    
"""

def forecasting():
    # try:
    st.title("Forecasting Page ")
    df = pd.read_csv("data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index("Date",inplace=True)
    

    st.write("Plase select forecasting method on sidebar")
    method = st.sidebar.selectbox("Select a Forecast method",("Box-Jenkins","Neural Network"))

    if(method == "Box-Jenkins"):
        
        diffnum = st.sidebar.slider("Select a discrete difference of data",0,30,1)
        lagnum = st.sidebar.slider("Select lagg discrete difference of data",0,100,1)

        fc = Forecast(df['Adj Close'],diffnum)
        df_diff = fc.get_diff()
        if(df_diff is False):
            st.error("Please select valid discrete difference of data")
        else:
            st.subheader("ACF and PACF Graph")
            col1,col2 = st.columns(2)
            col1.pyplot(plot_acf(df_diff.dropna(),lags=lagnum,title="Stock Price ACF"),color="green",vlines_kwargs={"color": "green"})
            col2.pyplot(plot_pacf(df_diff.dropna(),lags=lagnum,title="Stock Price PACF"))
        
        st.subheader("ARIMA Model Result")
        st.sidebar.write("Plase select a model parameters")
        issesion = st.sidebar.checkbox("Seasonal Data")
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
                col2.pyplot(plotpred(df[:n_size],pred))
                st.write(model.summary())
                


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

                



                
        # Sessional olmadığı zaman
        else:
            pred, predmodel = fc.arima_model(df['Adj Close'][:n_size],int(p),int(d),int(q),P=None,D=None,Q=None)

            if st.sidebar.button("Forecast"): # buton
                
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

                st.write(predmodel.summary())

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
                date_pred_start = st.date_input("Select a start date to predict",min_value = df.index.max())
                date_pred_end = st.date_input("Select a end date to predict",min_value = date_pred_start)
                if st.button("Predict"):
                    pred = Forecast.predict_feature(model=predmodel,start=date_pred_start,end=date_pred_end)
                    st.subheader("Prediction Result")
                    col1,col2 = st.columns(2)
                    col1,col2 = st.columns(2)
                    col1.write(pred)
                    col2.plotly_chart(plot_feature_pred(pred))

    elif(method == "Neural Network"):
        model_methot = st.sidebar.selectbox("Select a Neural Network Model",("LSTM","RNN"))
        train_size = st.sidebar.slider("Select a size of training data",1,len(df),5)
        if(model_methot == "LSTM"):
            
            look_back = st.sidebar.number_input("Select a size of forget ",1,train_size,step=1)
        else:
            window_size = st.sidebar.slider("Select a size of window size",1,train_size,2)

        loss_func = st.sidebar.radio("Select a loss function",("MSE","MAE","MSLE","MAPE"))
        optimizer_methot = st.sidebar.radio("Select a optimizer",("SGD","RMSprop","Adam","Adagrad","Adadelta","Adamax","Nadam"))
        epoch_size = st.sidebar.number_input("Select a epochs",1,500,5)
        method_name = st.sidebar.multiselect("Select a evulation metrics",["MAPE","RMSE","MAE","MSE","R2"],default=["MAPE"])
        get_model = st.sidebar.button("Train Model")


        if(get_model):
            if(model_methot == "LSTM"):
                with st.spinner('Wait for result...'):
                    trainDATA,testDATA,model = Forecast.LSTM(df,train_size,loss_func,optimizer_methot,epoch_size,look_back)
                    if trainDATA is not None:
                        st.success('Done!')
                
                st.subheader("Train Result")    
                st.plotly_chart(plot_nn_train(trainDATA))
                st.subheader("Test Result")
                st.plotly_chart(plot_nn_test(testDATA))

                st.plotly_chart(plot_nn(trainDATA,testDATA,df,"Stoke Price Prediction with LSTM" ))
                st.subheader("Result Data")
                col1,col2 = st.columns(2)
                col1.write(testDATA)
                col2.write(trainDATA)

                st.subheader("Model Evaluation")
                st.write("Train Accuracy")

                valueList = []
                metricsList = []
                for i in method_name:
                
                    values = evulation(i,trainDATA['train_pred'],trainDATA['Actual'])             
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
    # =============================================================================
                
                st.write("Test Accuracy")
                
                valueList = []
                metricsList = []
                for i in method_name:
                
                    values = evulation(i,testDATA['test_pred'],testDATA['Actual'])             
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

            elif(model_methot == "RNN"):
                with st.spinner('Wait for result...'):
                    pred_df_test,pred_df_train = Forecast.RNN(df,train_size,window_size,loss_func,optimizer_methot,epoch_size)
                    if(pred_df_test is not None):
                        st.success('Done!')
                st.caption("Prediction Result")
                st.plotly_chart(plot_nn(pred_df_train,pred_df_test,df,"Stock Price Prediction with RNN"))
                
                
                st.subheader("Result Data")
                col1,col2 = st.columns(2)
                col1.write(pred_df_test)
                col2.write(pred_df_train)

                

                st.subheader("Model Evaluation")
                st.write("Train Accuracy")

                valueList = []
                metricsList = []
                for i in method_name:
                
                    values = evulation(i,pred_df_train['train_pred'],df['Adj Close'][:len(pred_df_train)])          
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
    # =============================================================================
                
                st.write("Test Accuracy")
                
                valueList = []
                metricsList = []
                for i in method_name:
                
                    values = evulation(i,pred_df_test['test_pred'],df['Adj Close'][len(pred_df_train):])            
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


"""
    anomaly page: Detecting anomalies in stock price data
    params:
        None

"""
def anomaly():
    sys.stderr = open("error.log", "w")
    st.title("Anomaly Detection Page")
    modelname = st.sidebar.selectbox("Select the Model",('abod', 'cluster', 'cof', 'iforest',
    'histogram', 'knn', 'lof','svm','pca','mcd','sod','sos'),help="Select the Model",index=3)
    fraction = st.sidebar.slider("Select the number of fraction",0.0,max_value = 1.0,value=0.5,step=0.01)
    buttonstate = st.sidebar.button("Get Anomalies")
    if(buttonstate):
        df = pd.read_csv("data.csv")
        data = ad.Preprocess(data=df)
        result = ad.CaretModel(data,modelname,fraction)
        with st.spinner('Wait for plot...'):
            st.plotly_chart(ad.plotModel(result))
            st.success('Done!')
        
"""
    accout page: Personal information of the user
    params:
        username: username of the user
        db: database object of the postgresql database
        user: user object of the postgresql database
"""
def account(username,db,user):
    st.title("My Profile")
    st.write("Welcome to the account page " + username)
    with st.form("Update"):
        mail = st.text_input("Email",value="")
        new_pass = st.text_input("New Password",type="password",value="")
        confirm_pass = st.text_input("Confirm Password",type="password",value="")
        confirmButton = st.form_submit_button("Confirm")

        if(confirmButton):
            if(new_pass == confirm_pass):
                if dbc.password_update(mail,new_pass,db,user):
                    st.success("Update Successfully")
                else:
                    st.error("Update Failed. Please try again")

"""
    about page: Information about the project
"""
def aboutpage():
    col1,col2 = st.columns(2)
    col1.markdown("""
    #  🟢 About
    This is a web application that can help you estimate the stock prices and anomaly detection. 
    
    Available for any company.
    """)
    col2.markdown("""
    # 🟣 Features 
    - Stock Price Prediction V.1.0
    
    1. You can see the stock price graph
    2. You can see the stock price anomaly
    3. You can see the stock price forecast

    
    """)
    st.markdown("""
    *********************************
    # 🔴 How to use
    1. Select a stock from the dropdown menu below to see the stock price result
    2. You can type a stock name if you select "Other"
    3. You can select a start date and end date
    4. You can see the stock price anomaly
    5. You can see the stock price anomaly with the original graph
    6. You can see the stock price forecast

    **❗️ WARNING:** Can't switch to other pages without saving the data you selected.

    """)
    st.markdown("""
    *********************************
    # 👨🏽‍💻 Developed by

    - Burak Uğur

    """)
    st.markdown("""
    #### 📞 Contact
    >   Email: burak.uugur12@gmail.com

    >   Github: @burakugurr 
    
    >   LinkedIn: @burakugurr

    [![](https://img.shields.io/twitter/follow/burakugur?style=social)](https://www.twitter.com/bburakuugur)
    """)
    st.markdown("""

    # 🛡 License 
    This project is licensed under the MIT License.

    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
    
    """)
    st.markdown("""**If you like this project, please give a star on Github. Thank you!** """)


"""
    The main function of the web application
    params:
        db: database object of the postgresql database
        user: user object of the postgresql database
        username: username of the user
"""
def app(db,user,username="User"):

    # =============================================================================
    app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Home","Data Insight","Forecasting", "Anomaly Detection","My Account","About"],
    help="Select the page",index=0)

    
    # =============================================================================

    if app_mode == "Data Insight":
        try:
            datainsgiht()
        except Exception as e:
            st.error("Please restart the app"+"\n"+str(e))
    elif app_mode == "Home":
        homepage(username)
    elif app_mode == "Forecasting":
        try:
            forecasting()
        except FileNotFoundError:
            st.error("Please go main page, select company and save the data")
        except Exception as e:
            st.error("Please try again"+"\n"+str(e))

    elif app_mode =='Anomaly Detection':
        try:
            anomaly()

        except FileNotFoundError:
            st.error("Please go main page, select company and save the data")
        except Exception as e:
            st.error("Please try again"+"\n"+str(e))

    
    elif app_mode == 'About':
        aboutpage()

    elif app_mode == 'My Account':
        try:
            account(username,db,user)
        except Exception as e:
            st.error("Please try again"+"\n"+str(e))