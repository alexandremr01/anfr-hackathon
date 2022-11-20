import streamlit as st
import  streamlit_toggle as tog

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

import datetime

from map import get_map
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

from statsmodels.tsa.seasonal import seasonal_decompose

pio.renderers.default = 'colab'

antenna=pd.read_csv('data/Etats reseaux telecoms/2021_06_30_Etat reseaux.csv',sep=';')
antenna["DATE MES EMETTEUR"] =  pd.to_datetime(antenna["DATE MES EMETTEUR"], format="%d/%m/%Y")
antenna['CITY']=antenna['NUMERO SONDE FIXE'].apply(lambda x: x[:-3])

sensors = pd.read_excel('data/Dates_mise en service_sondes_autonomes.xlsx')
sensors['city']=sensors['numero sonde'].apply(lambda x: x[:-3])

measurements = pd.read_csv('data/mesures_exposition_sondes_autonomes.csv', delimiter=';')
measurements['date']= pd.to_datetime(measurements['date'])
measurements['day'] = measurements['date'].map(lambda x: x.date())

mean = measurements[['day', 'E_volt_par_metre'] ].groupby('day').mean().reset_index()
mean['day'] = pd.to_datetime(mean['day']).map(lambda x: x.date()).astype('datetime64[ns]')

def get_highest_frequencies(ffty_half, n_max=5):
    max = np.argpartition(abs(ffty_half), -25)[-25:]
    max = np.flip(max)
    max = max[np.where((max > 3) ) ]
    periods = 1 / (max / 2 / (2*len(ffty_half)))
    _,idx = np.unique(np.round(periods), return_index=True)
    values = np.round(periods[np.sort(idx)]) / 24
    return max[:n_max], values[:n_max]

def plot_sensor(sensor_name, start_date, end_date):
    df_city = measurements[measurements['numero'] == sensor_name]
    df_city = df_city.sort_values(by='date')
    df_city['day'] = pd.to_datetime(df_city['date']).dt.date
    df_city['day'] = df_city['day'].astype('datetime64[ns]')

    df_new = pd.DataFrame()
    df_new['day'] = pd.date_range(start=start_date, end=end_date)
    df_new = pd.merge(df_new, df_city, on='day', how='left')

    df_new = pd.merge(df_new, mean, on='day', how='left')
    fig = go.Figure()

    fig.update_layout(
        title="Exposure by day",
        xaxis_title="Date",
        yaxis_title="Exposure - V/m",
    )
    fig.add_trace(go.Scatter(x=df_new['day'], y=df_new['E_volt_par_metre_x'],
                    mode='lines',
                    name=sensor_name))
    fig.add_trace(go.Scatter(x=df_new['day'], y=df_new['E_volt_par_metre_y'],
                    mode='lines',
                    name='Sensors average'))
                    
    series = df_new['E_volt_par_metre_x'].to_numpy()
    series = series[~np.isnan(series)]

    # result=seasonal_decompose(df['#Passengers'], model='multiplicable', period=12)
    fft = abs(np.fft.fft(series))
    fft_half = fft[:int(len(fft)/2)]

    max_frequencies, periods = get_highest_frequencies(fft_half)
    pairs = zip(periods, max_frequencies)
    
    fft_x = np.array(list(range(len(fft_half[1:])))) * (1/2) / len(series)
    fig2 = go.Figure(go.Scatter(x=fft_x, y=fft[1:]))
    fig2.update_layout(
        title="Fast Fourrier Transform (Omitting point at f=0)",
        xaxis_title="Frequency (1/h)",
        yaxis_title="Amplitude",
    )

    fft[50:] = 0
    lp_filtered = np.real(np.fft.ifft(fft))
    fig.add_trace(go.Scatter(x=df_new['day'], y=lp_filtered,
                    mode='lines',
                    name='Lowpass'))

    return fig, fig2, pairs


st.set_page_config(layout="wide")
st.title("Antennas and sensors")
st.write("Use the following map to search for antennas and sensors")
reference_date = st.date_input("Reference date")

val = tog.st_toggle_switch(label="Full map", 
                default_value=False, 
                label_after = False, 
                inactive_color = '#D3D3D3', 
                active_color="#11567f", 
                track_color="#29B5E8"
                )
fig = get_map(antenna, sensors, reference_date,open_view=val)
st.plotly_chart(fig, use_container_width=True,)

c1, c2, c3 = st.columns(3)
with c1:
    selected_sensor = st.selectbox('Sensor', sensors['numero sonde'])
with c2:
    start_date = st.date_input("Start date", value=datetime.date(2022, 1, 1))
with c3:
    end_date = st.date_input("End date")

fig_sensor, fig_fourrier, pairs = plot_sensor(selected_sensor, start_date, end_date)
st.plotly_chart(fig_sensor, use_container_width=True,)
st.plotly_chart(fig_fourrier, use_container_width=True,)

for pair in pairs:
    st.write(f'Period {pair[0]}h, amplitude {pair[1]}')