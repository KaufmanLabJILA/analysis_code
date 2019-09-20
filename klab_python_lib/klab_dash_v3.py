# GENERAL:
import urllib
import re
import numpy as np
from numpy import array as arr
import pandas as pd
# from pandas_datareader.data import DataReader
import time
import os
import datetime

# DASH:
import dash
import dash_core_components as dcc
import dash_html_components as html
from collections import deque
import plotly.graph_objs as go
# import random

# EMAIL MODULE:
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

starttime = datetime.datetime.now()
mail_time = starttime
mail_delay = datetime.timedelta(seconds=600)

delay = 2000 #refresh delay in ms

channelnames = [
"Ti:Saph Power",
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

limit_dict = {
"Ti:Saph Power": 4096,
"Channel 1": 4096,
"Channel 2": 4096,
"Channel 3": 4096,
"Channel 4": 4096,
"Channel 5": 4096,
"Channel 6": 4096,
"Channel 7": 4096,
"Chilled Water": 25,
"689nm Heat Pipe": 550,
"461nm Heat Pipe": 500,
"Z Coil": 25,
"Science Table Temperature": 25,
"Science Table Humidity": 100
}

unit_dict = {
"Ti:Saph Power": 'NA',
"Channel 1": 'NA',
"Channel 2": 'NA',
"Channel 3": 'NA',
"Channel 4": 'NA',
"Channel 5": 'NA',
"Channel 6": 'NA',
"Channel 7": 'NA',
"Chilled Water": 'temp',
"689nm Heat Pipe": 'temp',
"461nm Heat Pipe": 'temp',
"Z Coil": 'temp',
"Science Table Temperature": 'temp',
"Science Table Humidity": 'hum'
}

ylabel_dict = {}
for name in channelnames:
	if unit_dict[name] == 'temp':
		ylabel_dict[name] = "Temperature"
	elif unit_dict[name] == 'hum':
		ylabel_dict[name] = "Humidity (%)"
	else:
		ylabel_dict[name] = "Arb. Normalized Scale"

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def time_ind(df, itime):
	i = 0
	for time in df.index.values:
		i += 1
		if datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S.%f')>itime:
			break
	return i

def update_df():
	global f
	global channelnames
	global df_main
	n = file_len(f)
	df_temp = pd.read_csv(f, skiprows=n-1, names = channelnames, index_col=None)
	df_main = pd.concat([df_main,df_temp], ignore_index=False)

def ad8495_vc(v):
	# Returns temperature from K-type thermocouple fed through ad8495 thermocouple amplifier.
	return (v - 1.25) / 0.005

def c_to_f(Tc):
	return (Tc * 9./5.) + 32.

def update_obd_values(init=False):
	global mail_time
	global channelnames
	global limit_dict

	# response = urllib.request.urlopen('http://128.138.141.105/')
	response = urllib.request.urlopen('http://128.138.141.111/')
	html = response.read()
	rawdata = re.sub('nan', '0', str(html, 'ascii'))
	rawdata = re.sub(r'[^\d.]', ' ', rawdata)
	data = arr(rawdata.split()).astype(float).reshape((14, 3))
	# data[:8, 2] = data[:8, 2] / 4096.0
	data[[8, 10, 11], 2] = ad8495_vc(data[[8, 10, 11], 2] / 4096. * 3.3)
	# 1/2 Voltage divider on thermocouple signal:
	data[9, 2] = ad8495_vc(data[9, 2] * 2 / 4096. * 3.3)
	df_row = pd.DataFrame(data = data[:,2][np.newaxis], index = [datetime.datetime.now()], columns = channelnames)
	if init:
		df_row.to_csv(f, mode='w', header=True, index = True, index_label='Time')
	else:
		df_row.to_csv(f, mode='a', header=False, index = True, index_label='Time')

	for key in channelnames:
		value = df_row[key][-1]
		if value >= limit_dict[key] and datetime.datetime.now() > mail_time + mail_delay:
			send_alert(key, value, limit_dict[key], ylabel_dict[key])
			print("Sending Alert!")
			mail_time = datetime.datetime.now()

f = 'dash_data.csv'
if not os.path.isfile(f):
	update_obd_values(init=True)
df_main = pd.read_csv(f, index_col=0)

max_length = 604800 #1 week in seconds
logscale = np.log10(max_length)

app.layout = html.Div([
	html.H2(
		'Kaufman Lab Dashboard', style = {'text-align': 'center'}),

	html.Div([
		dcc.Dropdown(
			id='data-name',
			options=[{'label': s, 'value': s} for s in channelnames],
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

	# # Hidden div inside the app that stores the intermediate value
    # html.Div(id='intermediate-value', style={'display': 'none'}, children = df_main),

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
	global f
	global df_main
	global ylabel_dict
	# print(df_main.index.values)
	graphs = []
	t1 = time.time()
	update_obd_values()
	t2 = time.time()
	update_df()
	t3 = time.time()
	print(t2-t1)
	print(t3-t2)
	if len(data_names)>2:
		class_choice = 'col s12 m6 l4'
	elif len(data_names) == 2:
		class_choice = 'col s12 m6 l6'
	else:
		class_choice = 'col s12'

	for data_name in data_names:
		yname = ylabel_dict[data_name]
		itime = datetime.datetime.strptime(df_main.index.values[-1],'%Y-%m-%d %H:%M:%S.%f')-datetime.timedelta(seconds=(10 ** time_range))
		# ftime = df_main.index.values[-1]
		nstart =time_ind(df_main, itime)

		if unit_dict[data_name] == 'temp':
			if units == 'C':
				ydat = df_main[data_name][nstart:]
			elif units == 'F':
				ydat = df_main[data_name][nstart:].apply(c_to_f)
		else:
			ydat = df_main[data_name][nstart:]

# arr(df_main['Channel 1'][nstart:])

		# itime = itime.strftime("%Y-%m-%d %H:%M:%S.%f")
		ymin = np.min(ydat)
		ymax = np.max(ydat)
		data = go.Scatter(
			x=df_main.index.values[nstart:],
			y=ydat,
			name='Scatter',
			fill="tozeroy",
			fillcolor="#6897bb"
			)
		graphs.append(html.Div(dcc.Graph(
			id=data_name,
			animate=False,
			figure={'data': [data],'layout' : go.Layout(
				xaxis=dict(
					title="Time"
					# range=[itime, ftime]
					),
				yaxis=dict(
					title=yname,
					range = [ymin, ymax]
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
