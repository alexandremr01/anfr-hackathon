import  plotly.graph_objs as go
import numpy as np 

def get_angle(sensor, antena, azimuth):
  delta = sensor - antena
  orientation = np.array([np.sin(azimuth), np.cos(azimuth)])
  cosine = np.dot(delta, orientation) / (np.linalg.norm(delta) * np.linalg.norm(orientation))
  return np.arccos(cosine)

def antenna_to_angle(row):
  sensor = np.array([row['longitude'], row['latitude']])
  antena = np.array([row['LONGITUDE DD'], row['LATITUDE DD']])
  angle = row['AZIMUT ANTENNE']
  return np.rad2deg(get_angle(sensor, antena, np.deg2rad(angle)))

def get_map(antenna, sensors, date,open_view=True):

    antenna_data = antenna[(antenna['DATE MES EMETTEUR'].dt.date<date) & (antenna['FINAL TRIMESTER'].dt.date>date)]
    sensors_data = sensors[sensors['date MES'].dt.date<date]

    # Gets the angular distance
    antenna_aux = antenna_data.merge(sensors_data, left_on=['NUMERO SONDE FIXE'], right_on=['numero sonde'])[['LATITUDE DD', 'LONGITUDE DD', 'AZIMUT ANTENNE', 'latitude', 'longitude']]
    antenna_aux['angular_distance'] = antenna_aux.apply(antenna_to_angle, axis=1)
    antenna_aux = antenna_aux[['LATITUDE DD', 'LONGITUDE DD', 'angular_distance']].groupby(['LATITUDE DD', 'LONGITUDE DD']).min().reset_index()
    antenna_aux['formatted_angular_distance'] = antenna_aux.apply(lambda row : f'Angular diff: {np.round( row["angular_distance"] )}', axis=1)

    # Separate three kinds of antenna
    antenna_disaligned = antenna_aux[ antenna_aux['angular_distance'] > 60 ]
    antenna_semi_aligned = antenna_aux[ (antenna_aux['angular_distance'] < 60) &  (antenna_aux['angular_distance'] > 20)]
    antenna_aligned = antenna_aux[ antenna_aux['angular_distance'] < 20 ]

    scatter = go.Scattermapbox(
        mode = "markers+text",
        lon = sensors_data['longitude'],
        lat = sensors_data['latitude'],
        text = sensors_data['numero sonde'],
        textposition = "bottom right",
        name = 'Sensors',
        marker=dict(color='red',size=15)
    )
    fig = go.Figure(scatter)

    fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = antenna_disaligned['LONGITUDE DD'],
            lat = antenna_disaligned['LATITUDE DD'],
            text = antenna_disaligned['formatted_angular_distance'],
            name = 'Disaligned Antenna',
            marker=dict(color = 'lightskyblue', size=8)
        )
    )
    fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = antenna_semi_aligned['LONGITUDE DD'],
            lat = antenna_semi_aligned['LATITUDE DD'],
            text = antenna_semi_aligned['formatted_angular_distance'],
            name = 'Semi-Aligned Antenna',
            marker=dict(color = 'deepskyblue', size=8)
        )
    )
    fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = antenna_aligned['LONGITUDE DD'],
            lat = antenna_aligned['LATITUDE DD'],
            text = antenna_aligned['formatted_angular_distance'],
            name = 'Aligned Antenna',
            marker=dict(color = 'darkblue', size=8)
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
    return fig