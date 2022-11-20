import  plotly 
import  plotly.graph_objs as go
import pandas as pd
import numpy as np

def get_relative_error(X, sensor_name):
    q95 = ((X['E']-X['E pred'])/X['E']).quantile(0.95)
    q05 = ((X['E']-X['E pred'])/X['E']).quantile(0.05)
    fig = go.Figure(go.Scatter(x=X['Date'],y=(X['E']-X['E pred'])/X['E']))
    fig.add_hline(y=((X['E']-X['E pred'])/X['E']).quantile(0.95), line_dash="dot", line_color="red",)
    fig.add_hline(y=((X['E']-X['E pred'])/X['E']).quantile(0.05), line_dash="dot", line_color="red")
    fig.update_layout(
        title=f"Relative Error - {sensor_name} - Quantile 95% = {100*round(q95,2)} % of error and Quantile 5% = {100*round(q05,2)} % of error",
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title="Relative Error",
    )
    return fig

def get_comparative_exposure(X,sensor_name):
    x = np.linspace(X['E pred'].min(),X['E pred'].max(),1000)
    fig = go.Figure(go.Scatter(x=X['E pred'],y=X['E'],mode='markers',name='Exposure Mesured by Predicted Exposure'))
    fig.add_trace(go.Scatter(x=x,y=x,mode='lines', name='Identity Line (y=x)'))
    fig.update_layout(
        title=f"Exposure Mesured by Predicted Exposure - {sensor_name}",
        title_x=0.5,
        xaxis_title="Predicted Exposure (V/m)",
        yaxis_title="Exposure Mesured (V/m)",
    )
    return fig

def get_absolute_std_error(X,sensor_name):
    errors = (X['E']-X['E pred'])
    errors = (errors-errors.mean())/errors.std()
    errors.index = X['Date'] 

    fig = go.Figure(go.Scatter(x=X['Date'],y=np.abs(errors)))
    fig.add_hline(y=4, line_dash="dot", line_color="red",)
    fig.update_layout(
        title=f"Absolute Error - {sensor_name}",
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title="Absolute Error",
    )

    outliers = []

    for i,error in enumerate(errors):
        if error>4:
            outliers.append(errors.index[i])

    return fig, outliers
