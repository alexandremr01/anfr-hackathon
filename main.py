import streamlit as st

import numpy as np
import  pandas as pd
import  seaborn as sns
import  scipy.stats as stats
import  matplotlib.pyplot as plt

import  plotly 
import  plotly.graph_objs as go
import  plotly.io as pio
from plotly.subplots import make_subplots
import  plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import  cufflinks as cf
import  plotly.figure_factory as ff 
from plotly.offline import iplot
from plotly import tools
from ipywidgets import widgets

from matplotlib.colors import to_rgba

plt.style.use('seaborn-darkgrid')

# You can go offline on demand by using
cf.go_offline() 
# initiate notebook for offline plot
init_notebook_mode(connected=False)         

# set some display options:
plt.rcParams['figure.dpi'] = 100
colors = px.colors.qualitative.Prism
pio.templates.default = "plotly_white"
import geojson

plotly.offline.init_notebook_mode(connected = True)
import plotly.io as pio
import plotly.express as px


pio.renderers.default = 'colab'

antenna=pd.read_csv('data/Etats reseaux telecoms/2021_06_30_Etat reseaux.csv',sep=';')
antenna["DATE MES EMETTEUR"] =  pd.to_datetime(antenna["DATE MES EMETTEUR"], format="%d/%m/%Y")
antenna['CITY']=antenna['NUMERO SONDE FIXE'].apply(lambda x: x[:-3])

sensors = pd.read_excel('data/Dates_mise en service_sondes_autonomes.xlsx')
sensors['city']=sensors['numero sonde'].apply(lambda x: x[:-3])

measurements = pd.read_csv('data/mesures_exposition_sondes_autonomes.csv', delimiter=';')
measurements['date']= pd.to_datetime(measurements['date'])
measurements['day'] = measurements['date'].map(lambda x: x.date())


def plot_graph(date,open_view=True):

    antenna_data = antenna[antenna['DATE MES EMETTEUR'].dt.date<date]
    sensors_data = sensors[sensors['date MES'].dt.date<date]

    scatter = go.Scattermapbox(
        mode = "markers+text",
        lon = sensors_data['longitude'],
        lat = sensors_data['latitude'],
        text = sensors_data['numero sonde'],
        textposition = "bottom right",
        name = 'Sensors',
        marker=dict(color='blue',size=12)
    )
    fig = go.Figure(scatter)

    fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = antenna_data['LONGITUDE DD'],
            lat = antenna_data['LATITUDE DD'],
            name = 'Antenna'
        )
    )

    if open_view:
        style = 'open-street-map'
    else:
        style = 'carto-positron'

    fig.update_layout(mapbox=dict(style=style,
            bearing=0,
            pitch=0,
            zoom=4,
            center = {"lat": sensors_data['latitude'].mean(), "lon": sensors_data['longitude'].mean()}
        ))


    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
    #   fig.show()
    st.plotly_chart(fig, use_container_width=True,)

def plot_sensor(sensor_name, start_date, end_date):
    df_city = measurements[measurements['numero'] == sensor_name]
    df_city = df_city.sort_values(by='date')
    df_city['day'] = pd.to_datetime(df_city['date']).dt.date
    df_city['day'] = df_city['day'].astype('datetime64[ns]')

    df_new = pd.DataFrame()
    df_new['day'] = pd.date_range(start=start_date, end=end_date)
    df_new = pd.merge(df_new, df_city, on='day', how='left')
    fig = px.line(df_new, x='date', y='E_volt_par_metre' )
    fig.update_layout(
        title="Exposure by day",
        xaxis_title="Date",
        yaxis_title="Exposure - V/m",
    )
    st.plotly_chart(fig, use_container_width=True,)

import datetime
st.set_page_config(layout="wide")
st.title("Antennas and sensors")
st.write("Use the following map to search for antennas and sensors")
reference_date = st.date_input("Reference date")

plot_graph(reference_date,open_view=False)

c1, c2, c3 = st.columns(3)
with c1:
    selected_sensor = st.selectbox('Sensor', sensors['numero sonde'])
with c2:
    start_date = st.date_input("Start date", value=datetime.date(2022, 1, 1))
with c3:
    end_date = st.date_input("End date")

plot_sensor(selected_sensor, start_date, end_date)

