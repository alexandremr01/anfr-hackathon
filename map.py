import  plotly.graph_objs as go

def get_map(antenna, sensors, date,open_view=True):

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
    return fig