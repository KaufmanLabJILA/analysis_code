import dash
import dash_core_components as dcc
import dash_html_components as html
# from pandas_datareader.data import DataReader
import time
from collections import deque
import plotly.graph_objs as go
# import random

# import datetime
import urllib
import re
from numpy import array as arr

app = dash.Dash('klab-dash')

max_length = 3600
starttime = time.time()
normtimes = deque(maxlen=max_length)
ch0 = deque(maxlen=max_length)
ch1 = deque(maxlen=max_length)
ch2 = deque(maxlen=max_length)
ch3 = deque(maxlen=max_length)
ch4 = deque(maxlen=max_length)
ch5 = deque(maxlen=max_length)
ch6 = deque(maxlen=max_length)
ch7 = deque(maxlen=max_length)
ch8 = deque(maxlen=max_length)
ch9 = deque(maxlen=max_length)
ch10 = deque(maxlen=max_length)
ch11 = deque(maxlen=max_length)

data_dict = {"Laser Power":ch0,
             "Channel 1":ch1,
             "Channel 2":ch2,
             "Channel 3":ch3,
             "Channel 4":ch4,
             "Channel 5":ch5,
             "Channel 6":ch6,
             "Channel 7":ch7,
             "Channel 8":ch8,
             "689nm Heat Pipe":ch9,
             "461nm Heat Pipe":ch10,
             "Z Coil":ch11
             }

ylabel_dict = {"Laser Power":"Arb. Normalized Scale",
             "Channel 1":"Arb. Normalized Scale",
             "Channel 2":"Arb. Normalized Scale",
             "Channel 3":"Arb. Normalized Scale",
             "Channel 4":"Arb. Normalized Scale",
             "Channel 5":"Arb. Normalized Scale",
             "Channel 6":"Arb. Normalized Scale",
             "Channel 7":"Arb. Normalized Scale",
             "Channel 8":"Arb. Normalized Scale",
             "689nm Heat Pipe":"Temperature (C)",
             "461nm Heat Pipe":"Temperature (C)",
             "Z Coil":"Temperature (C)"
             }

def ad8495_vc(v):
    # Returns temperature from K-type thermocouple fed through ad8495 thermocouple amplifier.
    return (v-1.25)/0.005

def update_obd_values(normtimes, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11):
    if len(normtimes)==max_length:
        normtimes.append(time.time()-starttime)
        normtimes=deque([i-normtimes[0] for i in normtimes],maxlen=max_length)
    else:
        normtimes.append(time.time()-starttime)

    response = urllib.request.urlopen('http://128.138.141.105/')
    html = response.read()
    rawdata = re.sub('[^0-9, ]', '', str(html,'ascii'))
    data = arr(rawdata.split()).astype(float).reshape((12,3))
    data[:9,2] = data[:9,2]/4096.0
    data[9,2] = ad8495_vc(data[9,2]*2/4096.*3.3) #1/2 Voltage divider on thermocouple signal.
    data[10:,2] = ad8495_vc(data[10:,2]/4096.*3.3)
    i=0
    for data_of_interest in [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11]:
        data_of_interest.append(data[i,2])
        i += 1
    # print(normtimes)

    return normtimes, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11

normtimes, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11 = update_obd_values(normtimes, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11)

app.layout = html.Div([
    html.Div([
        html.H2('Kaufman Lab Dashboard',
                style={'float': 'left',
                       }),
        ]),
    dcc.Dropdown(id='data-name',
                 options=[{'label': s, 'value': s}
                          for s in data_dict.keys()],
                 value=['Laser Power','Z Coil', '689nm Heat Pipe','461nm Heat Pipe'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
    dcc.Interval(
        id='graph-update',
        interval=1000),
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})


@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('data-name', 'value')],
    events=[dash.dependencies.Event('graph-update', 'interval')]
    )
def update_graph(data_names):
    graphs = []
    update_obd_values(normtimes, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11)
    if len(data_names)>2:
        class_choice = 'col s12 m6 l4'
    elif len(data_names) == 2:
        class_choice = 'col s12 m6 l6'
    else:
        class_choice = 'col s12'


    for data_name in data_names:

        data = go.Scatter(
            x=list(normtimes),
            y=list(data_dict[data_name]),
            name='Scatter',
            fill="tozeroy",
            fillcolor="#6897bb"
            )

        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data],'layout' : go.Layout(
                xaxis=dict(
                    title="Time (s)",
                    range=[min(normtimes),max(normtimes)]
                    ),
                yaxis=dict(
                    title=ylabel_dict[data_name],
                    range=[min(data_dict[data_name]),max(data_dict[data_name])]
                    ),
                # margin={'l':50,'r':1,'t':45,'b':1},
                title='{}'.format(data_name))
                }
            ), className=class_choice))

    return graphs



external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})


if __name__ == '__main__':
    app.run_server(debug='False',port=80,host='0.0.0.0')
