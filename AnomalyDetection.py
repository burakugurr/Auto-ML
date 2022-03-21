from pycaret.anomaly import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycaret

"""
This Class is used for detecting anomalies in the data

"""
class Detector:
    # Initialize the class with the dataframe
    def Preprocess(data):
        data = data.drop(["High","Low","Open","Close","Volume"], axis=1)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.set_index("Date",inplace=True)
        data['day'] = [i.day for i in data.index]
        data['day_name'] = [i.day_name() for i in data.index]
        data['day_of_year'] = [i.dayofyear for i in data.index]
        data['week_of_year'] = [i.weekofyear for i in data.index]
        data['hour'] = [i.hour for i in data.index]
        data['is_weekday'] = [i.isoweekday() for i in data.index]
        return data
    """
        Create a model and return the model results.
        params:
            data: dataframe
            model: model name
            fraction: fraction of data to use for training
        return:
            model results
    """

    def CaretModel(data,model,fraction = 0.1):
        global iforest_results
        s = setup(data, session_id=123, silent=True,n_jobs=-1)

        for modelname in models().index.tolist():
            if modelname == str(model):
                modelend = create_model(modelname, fraction = fraction)
                iforest_results = assign_model(modelend)
                return iforest_results
        
    """
        Visualization anomalies and visualize the results
        params:
            iforest_results: model results
        return:
            visualization of the results    
    """    
    def plotModel(iforest_results):
        outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index

        y_values = [iforest_results.loc[i]['Adj Close'] for i in outlier_dates]

        fig = go.Figure([
            go.Scatter(
                name='Price $',
                x=iforest_results.index,
                y=iforest_results['Adj Close'],
                mode='lines',
                line=dict(color='goldenrod', width=5),
                showlegend=True
            ),
            go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                            name = 'Anomaly', 
                            marker=dict(color='darkred',size=10))
            ])

        fig.update_layout(
        yaxis_title='Stock Price ($/Day)',
        xaxis_title='Date',
        title='UNSUPERVISED ANOMALY DETECTION',
        hovermode="x",
        legend_title="Legend",
        plot_bgcolor="darkblue",
        width=1080 ,
        height=720,)   
        return fig