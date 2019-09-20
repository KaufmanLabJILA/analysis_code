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
import numpy as np
from numpy import array as arr

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

### Email system:
user = 'kaufmanlabJILA'
passwd = 'Sr88IsABoson'
smtp_server = 'smtp.gmail.com:587'
sender = 'kaufmanlabJILA@gmail.com'
receivers = 'aaron.young@jila.colorado.edu, kaufmanlabJILA@gmail.com'

def send_alert(channelname, value, maxval, label):
	txt = 'ALERT: Lab sensor \"' + \
	 str(channelname) + '\" has reached a value of ' + str(value)
	msg = MIMEText(txt + ', exceededing its maximum value of ' +
				str(maxval) + ' in units of ' + label)
	msg['Subject'] = txt
	msg['From'] = sender
	msg['To'] = receivers

	server = smtplib.SMTP(smtp_server)
	server.starttls()
	server.login(user, passwd)
	server.send_message(msg)
	server.quit()

### Web app:

app = dash.Dash('klab-dash')

max_length = 100000 # time in # delay times
starttime = time.time()

mail_time = starttime
mail_delay = 300 #in seconds

delay = 2000 #refresh delay in ms

normtimes = deque(maxlen=max_length)

channelnames = [
"Laser Power",
"Channel 1",
"Channel 2",
"Channel 3",
"Channel 4",
"Channel 5",
"Channel 6",
"Channel 7",
"Chilled Water",
"689nm Heat Pipe",
"461nm Heat Pipe",
"Z Coil",
"Science Table Temperature",
"Science Table Humidity"
]

temperature_channels = [
"689nm Heat Pipe",
"461nm Heat Pipe",
"Chilled Water",
"Z Coil",
"Science Table Temperature"
]

humidity_channels = [
"Science Table Humidity"
]

limit_dict = {
"Laser Power": 1,
"Channel 1": 1,
"Channel 2": 1,
"Channel 3": 1,
"Channel 4": 1,
"Channel 5": 1,
"Channel 6": 1,
"Channel 7": 1,
"Chilled Water": 25,
"689nm Heat Pipe": 500,
"461nm Heat Pipe": 410,
"Z Coil": 25,
"Science Table Temperature": 25,
"Science Table Humidity": 100
}

data_dict = {}
ylabel_dict = {}
for name in channelnames:
	data_dict[name] = deque(maxlen=max_length)
	ylabel_dict[name] = "Arb. Normalized Scale"

for name in temperature_channels:
	ylabel_dict[name] = "Temperature"

for name in humidity_channels:
	ylabel_dict[name] = "Humidity (%)"

def ad8495_vc(v):
	# Returns temperature from K-type thermocouple fed through ad8495 thermocouple amplifier.
	return (v - 1.25) / 0.005

def c_to_f(Tc):
	return (Tc * 9./5.) + 32.

def update_obd_values(normtimes, data_dict):
	global mail_time
	if len(normtimes) == max_length:
		normtimes.append(time.time() - starttime)
		normtimes = deque([i - normtimes[0] for i in normtimes], maxlen=max_length)
	else:
		normtimes.append(time.time() - starttime)

	response = urllib.request.urlopen('http://128.138.141.105/')
	# response = urllib.request.urlopen('http://128.138.141.111/')
	html = response.read()
	rawdata = re.sub('nan', '0', str(html, 'ascii'))
	rawdata = re.sub(r'[^\d.]', ' ', rawdata)
	data = arr(rawdata.split()).astype(float).reshape((14, 3))
	data[:8, 2] = data[:8, 2] / 4096.0
	data[[8, 10, 11], 2] = ad8495_vc(data[[8, 10, 11], 2] / 4096. * 3.3)
	# 1/2 Voltage divider on thermocouple signal:
	data[9, 2] = ad8495_vc(data[9, 2] * 2 / 4096. * 3.3)

	i = 0
	for key, value in data_dict.items():
		value.append(data[i, 2])
		if data[i,2] >= limit_dict[key] and time.time()> mail_time + mail_delay:
			send_alert(key, data[i,2], limit_dict[key], ylabel_dict[key])
			print("Sending Alert!")
			mail_time = time.time()
		i += 1
	# print(normtimes)
	# print(time.time(), mail_time + mail_delay)
	return normtimes, data_dict

normtimes, data_dict = update_obd_values(normtimes, data_dict)

logscale = np.log10(max_length)

app.layout = html.Div([
	html.H2(
		'Kaufman Lab Dashboard', style = {'text-align': 'center'}),

	html.Div([
		dcc.Dropdown(
			id='data-name',
			options=[{'label': s, 'value': s} for s in data_dict.keys()],
			value=['Science Table Temperature', 'Z Coil', '689nm Heat Pipe', '461nm Heat Pipe'],
			multi=True
		 ),], style = {'width': '80%', 'marginLeft': 'auto', 'marginRight': 'auto'}),

	html.Div([
		html.Div([
			html.Label('Time Range'),
			dcc.Slider(
				id='time-slider',
				min = 1,
				max = logscale,
				step = logscale/1000,
				value = np.log10(3600),
			    marks={
		        1: {'label': 'S'},
		        np.log10(60): {'label': 'M'},
		        np.log10(3600): {'label': 'H'},
				np.log10(24*3600): {'label': 'D'},
		        np.log10(7*24*3600): {'label': 'W'}
		    	}
			),
			], style={'width': '80%', 'marginLeft': 'auto', 'marginRight': 'auto', 'display': 'inline-block'}),
		html.Div([
			html.Label('Temperature Units'),
			dcc.Dropdown(
				id = 'unit-select',
				clearable = False,
				options=[
					{'label': 'Celsius', 'value': 'C'},
					{'label': 'Fahrenheit', 'value': 'F'}
				],
				value='F'
			),
		], style={'width': '15%', 'marginLeft': '5%', 'display': 'inline-block'}),
	], style={'width': '80%', 'marginTop': 10, 'marginBottom': 60, 'marginLeft': 'auto', 'marginRight': 'auto'}),

	html.Div(children=html.Div(id='graphs'), className='row'),

	dcc.Interval(
		id='graph-update',
		interval=delay),
	], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})

@app.callback(
	dash.dependencies.Output('graphs','children'),
	[dash.dependencies.Input('data-name', 'value'),
	dash.dependencies.Input('time-slider', 'value'),
	dash.dependencies.Input('unit-select', 'value')
	],
	events=[dash.dependencies.Event('graph-update', 'interval')]
	)
def update_graph(data_names, time_range, units):
	graphs = []
	update_obd_values(normtimes, data_dict)
	if len(data_names)>2:
		class_choice = 'col s12 m6 l4'
	elif len(data_names) == 2:
		class_choice = 'col s12 m6 l6'
	else:
		class_choice = 'col s12'

	for data_name in data_names:
		yname = ylabel_dict[data_name]
		if data_name in temperature_channels:
			if units == 'C':
				ydat = list(data_dict[data_name])
			elif units == 'F':
				ydat = list(map(c_to_f, data_dict[data_name]))
		else:
			ydat = list(data_dict[data_name])

		data = go.Scatter(
			x=list(normtimes),
			y=ydat,
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
					range=[max(normtimes)-10 ** time_range, max(normtimes)]
					),
				yaxis=dict(
					title=yname,
					range=[min(ydat[int((max(normtimes)-10 ** time_range)/(delay/1000)):]), max(ydat[int((max(normtimes)-10 ** time_range)/(delay/1000)):])]
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
	app.run_server(debug='True', port=80, host='0.0.0.0')
