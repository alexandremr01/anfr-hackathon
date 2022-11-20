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

antenna=pd.read_csv('data/antennas.csv')
antenna["DATE MES EMETTEUR"] =  pd.to_datetime(antenna["DATE MES EMETTEUR"], format="%d/%m/%Y")
antenna['CITY']=antenna['NUMERO SONDE FIXE'].apply(lambda x: x[:-3])
antenna['FINAL TRIMESTER']= pd.to_datetime(antenna['FINAL TRIMESTER'])

sensors = pd.read_excel('data/Dates_mise en service_sondes_autonomes.xlsx')
sensors['city']=sensors['numero sonde'].apply(lambda x: x[:-3])

measurements = pd.read_csv('data/mesures_exposition_sondes_autonomes.csv', delimiter=';')
measurements['date']= pd.to_datetime(measurements['date'])
measurements['day'] = measurements['date'].map(lambda x: x.date())

means = measurements[['numero', 'E_volt_par_metre']].groupby('numero').mean().rename(columns={'E_volt_par_metre': 'mean'})
stds = measurements[['numero', 'E_volt_par_metre']].groupby('numero').std().rename(columns={'E_volt_par_metre': 'std'})
measurements = pd.merge(measurements, means, on='numero')
measurements = pd.merge(measurements, stds, on='numero')
measurements['E_volt_par_metre'] = (measurements['E_volt_par_metre'] - measurements['mean'] ) / measurements['std']


mean = measurements[['day', 'E_volt_par_metre'] ].groupby('day').mean().reset_index()
mean['day'] = pd.to_datetime(mean['day']).map(lambda x: x.date()).astype('datetime64[ns]')

def get_highest_frequencies(ffty_half, n_max=5):
    max = np.argpartition(abs(ffty_half), -25)[-25:]
    max = max[np.argsort(abs(ffty)[max])]
    max = np.flip(max)
    max = max[np.where((max > 3) ) ]
    periods = 1 / (max / 2 / (2*len(ffty_half)))
    _, idx = np.unique(np.round(periods/24), return_index=True)
    values = np.round(periods[np.sort(idx)]/24)
    return np.round(ffty_half[max[:n_max]]), values[:n_max]

def get_city_data(sensor_name, start_date, end_date):
    df_city = measurements[measurements['numero'] == sensor_name]
    df_city = df_city.sort_values(by='date')
    df_city['day'] = pd.to_datetime(df_city['date']).dt.date
    df_city['day'] = df_city['day'].astype('datetime64[ns]')

    df_new = pd.DataFrame()
    df_new['day'] = pd.date_range(start=start_date, end=end_date)
    df_new = pd.merge(df_new, df_city, on='day', how='left')

    df_new = pd.merge(df_new, mean, on='day', how='left')
    return df_new

def plot_sensor(df, sensor_name, lp_filtered):
    fig = go.Figure()
    fig.update_layout(
        title="Exposure by day",
        xaxis_title="Date",
        yaxis_title="Standardized Exposure - V/m",
    )
    fig.add_trace(go.Scatter(x=df['date'], y=df['E_volt_par_metre_x'],
                    mode='lines',
                    name=sensor_name))
    fig.add_trace(go.Scatter(x=df['date'], y=df['E_volt_par_metre_y'],
                    mode='lines',
                    name='Sensors average'))
    fig.add_trace(go.Scatter(x=df['date'], y=lp_filtered,
                    mode='lines',
                    line=dict(color='purple'),
                    name='Lowpass'))
    return fig

def plot_fourrier(fftx, ffty, hide_first=False):
    first_element = 1 if hide_first else 0 
    fig = go.Figure(go.Scatter(x=fftx[first_element:], y=ffty[first_element:]))
    fig.update_layout(
        title="Fast Fourrier Transform",
        xaxis_title="Frequency (1/h)",
        yaxis_title="Amplitude",
    )
    return fig

def plot_seasonal(result):  
    fig3 = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
    fig3.add_trace(
        go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines'),
            row=1, col=1,
        )

    fig3.add_trace(
        go.Scatter(x=result.trend.index, y=result.trend, mode='lines'),
            row=2, col=1,
        )

    fig3.add_trace(
        go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines'),
            row=3, col=1,
        )

    fig3.add_trace(
        go.Scatter(x=result.resid.index, y=result.resid, mode='lines'),
            row=4, col=1,
        )
    return fig3


st.set_page_config(layout="wide")
st.title("UnExpose")
st.write("Use the following map to search for antennas and sensors")


reference_date = st.date_input("Reference date")
val = st.checkbox("Full map")
fig = get_map(antenna, sensors, reference_date,open_view=val)
st.plotly_chart(fig, use_container_width=True,)

c1, c2, c3 = st.columns(3)
with c1:
    selected_sensor = st.selectbox('Sensor', sensors['numero sonde'])
with c2:
    start_date = st.date_input("Start date", value=datetime.date(2022, 1, 1))
with c3:
    end_date = st.date_input("End date")

df_city = get_city_data(selected_sensor, start_date, end_date)
series = df_city['E_volt_par_metre_x']
# series = pd.Series(np.where(np.isnan(series), series.interpolate(method='linear'), series))
series = series[~np.isnan(series)]

ffty = abs(np.fft.fft(series))
fft_half = ffty[:int(len(ffty)/2)]
max_frequencies, periods = get_highest_frequencies(fft_half)
pairs = zip(periods, max_frequencies)
fftx = np.array( list(range(len(fft_half[1:])))) * (1/2) / len(series)

def lowpass(ffty, a=30):
    ffty_aux = ffty.copy()
    ffty_aux[a:-a] = 0
    return np.real(np.fft.ifft(ffty_aux))

lp_filtered = lowpass(ffty)

fig_sensor = plot_sensor(df_city, selected_sensor, lp_filtered)
# result = seasonal_decompose(series, model='multiplicable', period=7*24*2)
# fig_seasonal = plot_seasonal(result)

st.plotly_chart(fig_sensor, use_container_width=True,)

# tab1, tab2 = st.tabs(["Fourrier", "Seasonal"])

# with tab1:
hide_f0 = st.checkbox("Hide f=0")
fig_fourrier = plot_fourrier(fftx, ffty, hide_f0)

st.plotly_chart(fig_fourrier, use_container_width=True,)
for pair in pairs:
    st.write(f'Period {int(pair[0])} days, amplitude {pair[1]}')

# with tab2:

#     st.plotly_chart(fig_seasonal, use_container_width=True,)
