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


pio.renderers.default = 'colab'

antenna=pd.read_csv('data/Etats reseaux telecoms/2021_06_30_Etat reseaux.csv',sep=';')
antenna["DATE MES EMETTEUR"] =  pd.to_datetime(antenna["DATE MES EMETTEUR"], format="%d/%m/%Y")
antenna['CITY']=antenna['NUMERO SONDE FIXE'].apply(lambda x: x[:-3])

sensors = pd.read_excel('data/Dates_mise en service_sondes_autonomes.xlsx')
sensors['city']=sensors['numero sonde'].apply(lambda x: x[:-3])





def plot_graph(date,open_view=True):

    antenna_data = antenna[antenna['DATE MES EMETTEUR']<date]
    sensors_data = sensors[sensors['date MES']<date]

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


    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #   fig.show()
    st.plotly_chart(fig, use_container_width=True)



st.set_page_config(layout="wide")
st.title("Antennas and sensors")
st.write("Use the following map to search for antennas and sensors")
plot_graph('2021-04-23',open_view=False)

