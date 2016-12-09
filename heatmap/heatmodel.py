# heatmodel.py
# Aidan Holloway-Bidwell
# 12-09-2016

import csv
from datatype import datatype
import datetime as dt
import math
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import sys
from timeunit import timeunit as unit

class heatmodel:

	# Designed to load point-event data with optional value data for overlay.
	# Load csv data file of format timestamp, latitude, longitude, (and optional value)
	# Find min and max lat, lon, and time and save values as instance variables.
	# Return dictionary(s) of lists of values, to be later turned into pandas dataframes
	def load_data(self, point_file_path, value_file_path=None):
		point_output = {}
		point_output['lat'] = []
		point_output['long'] = []
		point_output['time'] = []

		value_output = {}
		value_output['lat'] = []
		value_output['long'] = []
		value_output['time'] = []
		value_output['value'] = []

		with open(point_file_path, 'r') as point_file:
			point_reader = csv.reader(point_file)
			for row in point_reader:

				timestamp = int(row[0])
				lat = float(row[1])
				lon = float(row[2])
				time = dt.datetime.fromtimestamp(timestamp)

				point_output['lat'].append(lat)
				point_output['long'].append(lon)
				point_output['time'].append(time)

				# Update instance variables
				self.min_lat = lat if lat < self.min_lat else self.min_lat
				self.max_lat = lat if lat > self.max_lat else self.max_lat
				self.min_long = lon if lon < self.min_long else self.min_long
				self.max_long = lon if lon > self.max_long else self.max_long
				self.first_timestamp = timestamp if timestamp < self.first_timestamp else self.first_timestamp
				self.last_timestamp = timestamp if timestamp > self.last_timestamp else self.last_timestamp

		if value_file_path:
			with open(value_file_path, 'r') as value_file:
				value_reader = csv.reader(value_file)
				for row in value_reader:

					timestamp = int(row[0])
					lat = float(row[1])
					lon = float(row[2])
					time = dt.datetime.fromtimestamp(timestamp)
					value = float(row[3])

					if self.first_timestamp <= timestamp and timestamp <= self.last_timestamp:
						value_output['lat'].append(lat)
						value_output['long'].append(lon)
						value_output['time'].append(time)
						value_output['value'].append(value)
		else:
			value_output = None

		return point_output, value_output


	# Creates Pandas dataframe object with lat and lon (and value, if datatype.VALUE passed) columns indexed by time
	def create_dataframe(self, llt_dict, datatype=datatype.POINT):
		if datatype == datatype.VALUE:
			df = pd.DataFrame(
				np.array([llt_dict['lat'], llt_dict['long'], llt_dict['value']]).T, 
				index = llt_dict['time'], columns = ['lat', 'long', 'value'])
		else:
			df = pd.DataFrame(
				np.array([llt_dict['lat'], llt_dict['long']]).T, 
				index = llt_dict['time'], columns = ['lat', 'long'])
		return df


	# Splits data frames into smaller, per-hour (or per-time-unit) dataframes and stores as instance variables.
	# Also splits point data into 2D arrays (size determined by self.numcells) representing a mesh over the geographic data \
	# with number of points in each mesh cell.
	def split_data_by_time(self):
		xcells = int(self.numcells)
		ycells = int((self.urcrnrlat - self.llcrnrlat) / ((self.urcrnrlon - self.llcrnrlon) / xcells))

		d = dt.timedelta(microseconds=1)

		x = np.linspace(self.llcrnrlon, self.urcrnrlon, xcells + 1)
		y = np.linspace(self.llcrnrlat, self.urcrnrlat, ycells + 1)

		a, b = 0, 1

		while b < len(self.times):
			points = None
			start = self.times[a]
			end = self.times[b]

			# Slicing just up to but not including end is necessary since
			# Pandas is upper bound inclusive
			points = self.point_dataframe.loc[start:end - d]
			if not self.value_dataframe.empty:
				values = self.value_dataframe.loc[start:end - d]
				self.value_dataframes[start] = values

			# find which cell in the mesh each point falls in and increment the count for that cell
			point_grid = np.zeros((len(y) - 1, len(x) - 1))

			for point in points.itertuples():
				point_grid[np.searchsorted(y, point[1]) - 1][np.searchsorted(x, point[2]) - 1] += 1

			self.point_grids[start] = point_grid
			self.point_dataframes[start] = points

			a += 1
			b += 1

		# Stores computed dictionaries in larger dictionaries based on the number of cells in the mesh
		self.allmaps[self.numcells] = self.point_grids
		self.all_pointframes[self.numcells] = self.point_dataframes
		self.all_valueframes[self.numcells] = self.value_dataframes



	# Get lists of x and y coordinates of points that lie within current time window (self.start, self.end).
	# Also reset self.numpoints to be the number of points in current time window.
	def get_time_window_points(self):
		frames = self.point_dataframes

		points_list = []
		td = dt.timedelta(hours=1)
		if self.time_units == unit.DAY:
			td = dt.timedelta(days=1)
		elif self.time_units == unit.MONTH:
			td = dt.timedelta(months=1)

		curdate = self.start

		while curdate < self.end:
			if curdate in frames:
				frame = frames[curdate]
				map_points = None
				map_points = list(zip(frame['long'], frame['lat']))
					
				points_list += map_points
			else:
				print("No data points for date ", curdate)

			curdate += td

		self.numpoints = len(map_points)

		return points_list


	# Get numpy array of the total number of points falling in each cell of the mesh from self.start to self.end
	def get_point_map(self):
		xcells = int(self.numcells)
		ycells = int((self.urcrnrlat - self.llcrnrlat) / ((self.urcrnrlon - self.llcrnrlon) / xcells))

		td = dt.timedelta(hours=1)
		if self.time_units == unit.DAY:
			td = dt.timedelta(days=1)
		elif self.time_units == unit.MONTH:
			td = dt.timedelta(months=1)
		curdate = self.start

		point_map = np.zeros((ycells, xcells))
		while curdate < self.end:
			if curdate in self.point_grids:
				point_map = np.add(point_map, self.point_grids[curdate])
			curdate += td

		return point_map


	# Makes a mesh (of same dimensions as point mesh) for value data using interpolation.
	def get_value_map(self):
		frame = self.value_dataframes[self.start]

		spatial_resolution = min((self.max_lat - self.min_lat), (self.max_long - self.min_long)) / self.numcells
 
		vx = np.array(frame['long'].tolist())
		vy = np.array(frame['lat'].tolist())
		v = np.array(frame['value'].tolist())
		   
		yinum = (self.urcrnrlat - self.llcrnrlat) / spatial_resolution
		xinum = (self.urcrnrlon - self.llcrnrlon) / spatial_resolution
		vyi = np.linspace(self.llcrnrlat, self.urcrnrlat + spatial_resolution, xinum)
		vxi = np.linspace(self.llcrnrlon, self.urcrnrlon + spatial_resolution, yinum)
		vxi, vyi = np.meshgrid(vxi, vyi)
		xi = np.c_[vx.ravel(),vy.ravel()]
		xx = np.c_[vxi.ravel(),vyi.ravel()]

		return vxi, vyi, griddata((vx, vy), v, (vxi, vyi), method='cubic')


	# Use Haversine Formula to calculate (approximate) cell mesh size in miles or k.
	def get_cell_distance(self, mk):
		lon1_degrees, lon2_degrees = self.llcrnrlon, self.llcrnrlon + (self.urcrnrlon - self.llcrnrlon) / self.numcells
		lat1_degrees, lat2_degrees = (self.urcrnrlat - self.llcrnrlat) / 2, (self.urcrnrlat - self.llcrnrlat) / 2
		lon1, lat1, lon2, lat2 = map(math.radians, [lon1_degrees, lat1_degrees, lon2_degrees, lat2_degrees])
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = (math.sin(dlat/2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2)) ** 2 
		#c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) )
		c = 2 * math.asin(math.sqrt(a))
		if mk == 'MILES':
			return 3959 * c
		elif mk == 'KILOMETERS':
			return 6371 * c


	# Sets self.times array to new array starting at earliest data point time rounded down to nearest specified time unit,
	# to latest data point time, rounded up, incrementing by specified time unit. Resets self.start to first time in 
	# self.times and resets self.end to last time in self.times
	def set_times(self, units):
		self.time_units = units

		start = dt.datetime.fromtimestamp(self.first_timestamp)
		end = dt.datetime.fromtimestamp(self.last_timestamp)

		delta = dt.timedelta(days=1)

		if units == unit.HOUR:
			delta = dt.timedelta(hours=1)
			start = start.replace(minute = 0, second = 0)
			end += delta
			end = end.replace(minute = 0, second = 0)
		elif units == unit.DAY:
			start = start.replace(hour = 0, minute = 0, second = 0)
			end += delta
			end = end.replace(hour = 0, minute = 0, second = 0)
		elif units == unit.MONTH:
			delta = dt.timedelta(months=1)
			start = start.replace(day = 0, hour = 0, minute = 0, second = 0)
			end += delta
			end = end.replace(day = 0, hour = 0, minute = 0, second = 0)

		times = []
		curr = start
		while curr <= end:
			times.append(curr)
			curr += delta

		self.times = times
		self.start = start
		self.end = self.times[-1]


	def change_numcells(self, numcells):
		self.numcells = numcells
		self.split_data_by_time(self)


	def change_time_units(self, units):
		self.set_times(units)
		self.split_data_by_time(self)


	def change_time_window(self, start, end):
		self.start = start
		self.end = end


	def __init__(self, point_file_path, value_file_path=None, time_units=unit.HOUR):

		# size of map buffer around our data
		self.EXTRA = 0.01

		# The number of cells in the mesh will be self.numcells x self.numcells
		# Each cell will be colored based on the count of points that lie within it.
		self.numcells = 50

		# define geographical scope of the data.
		self.min_lat = 91
		self.max_lat = -91
		self.min_long = 181
		self.max_long = -181

		# define temporal scope of all the data.
		self.first_timestamp = sys.maxsize
		self.last_timestamp = 0

		# list of times by which data is divided
		self.times = []

		# timeunit enum object representing time units we are breaking up data by
		self.time_units = time_units

		# define scope of data currently being displayed or examined
		self.start = 0
		self.end = 0

		# Dictionaries to store previously computed point_grids and points for given cell sizes.
		# Key should be numcells.
		# Not currently used.
		self.allmaps = {}
		self.all_pointframes = {}
		self.all_valueframes = {}

		# array of point_grids for current cellsize (Also found in above dictionaries for key=self.numcells)
		self.point_grids = {}
		self.point_dataframes = {}
		self.value_dataframes = {}

		# number of points currently being displayed or examined, based on self.start and self.end
		self.numpoints = 0

		# load data from passed .csv filepaths into dictionaries
		point_data, value_data = self.load_data(point_file_path, value_file_path)

		# create pandas dataframes from dictionaries
		self.point_dataframe = self.create_dataframe(point_data, datatype.POINT)
		self.value_dataframe = pd.DataFrame()
		if value_data:
			self.value_dataframe = self.create_dataframe(value_data, datatype.VALUE)

		# populate instance variables and break up data based on sepcified time units, temporal and geographic scopes
		self.set_times(time_units)
		self.split_data_by_time()

		# calculate coordinates of outer edges of display/examination mesh. 
		# Since size of mesh cells is determined implicitly (by passing a number of cells along the x axis of the mesh),
		# these coordinates along with self.numcells determine size of mesh cells.
		xside = self.max_long - self.min_long
		yside = self.max_lat - self.min_lat
		buf = max(xside * self.EXTRA, yside * self.EXTRA)
		xrad = xside / 2.0 + buf
		yrad = yside / 2.0 + buf 
		self.lat_0 = (self.max_lat + self.min_lat) / 2.0
		self.lon_0 = (self.max_long + self.min_long) / 2.0

		self.llcrnrlon = self.lon_0 - xrad
		self.llcrnrlat = self.lat_0 - yrad
		self.urcrnrlon = self.lon_0 + xrad
		self.urcrnrlat = self.lat_0 + yrad