import csv
import datetime as dt
from enum import Enum
import math
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plot
from matplotlib.collections import PatchCollection
import matplotlib.font_manager as font
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from pylab import rcParams
import pyproj
from shapely.geometry import Point, MultiPoint, MultiPolygon
from matplotlib.patches import Polygon
import sys
from scipy.interpolate import griddata

# Different methods will need to behave differently depending on the type of data they're working with.
# This Enum class makes it clear what sorts of data this class is designed to display.
# datatype.POINT signifies simple coordinate points without values, over time, such as phone calls
# datatype.VALUE signifies coordinate points with values attached over time, such as temperature samples
class datatype(Enum):
	POINT = 1
	VALUE = 2

class heatmap:

	# Designed to load point-event data with optional value data for overlay.
	# Load csv data file of format timestamp, latitude, longitude, (and optional value)
	# Find min and max lat, lon, and time and save values as instance variables.
	# Return dictionary(s) of lists of values.
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


	# Creates Pandas dataframe object with lat and lon columns indexed by time
	def create_dataframe(self, llt_dict, datatype=datatype.POINT):
		if datatype == datatype.VALUE:
			df = pd.DataFrame(np.array([llt_dict['lat'], llt_dict['long'], llt_dict['value']]).T, index = llt_dict['time'], columns = ['lat', 'long', 'value'])
		else:
			df = pd.DataFrame(np.array([llt_dict['lat'], llt_dict['long']]).T, index = llt_dict['time'], columns = ['lat', 'long'])
		return df


	# Initializes basemap with transverse mercator projection large enough to contain all call data points
	def create_basemap(self):
		xside = self.max_long - self.min_long
		yside = self.max_lat - self.min_lat
		buf = max(xside * self.EXTRA, yside * self.EXTRA)
		xrad = xside / 2.0 + buf
		yrad = yside / 2.0 + buf 
		lat_0 = (self.max_lat + self.min_lat) / 2.0
		lon_0 = (self.max_long + self.min_long) / 2.0

		m = Basemap(
				projection = 'tmerc',
				epsg = '4326',
				lon_0 = lon_0,
				lat_0 = lat_0,
				ellps = 'WGS84',
				llcrnrlon = lon_0 - xrad,
				llcrnrlat = lat_0 - yrad,
				urcrnrlon = lon_0 + xrad,
				urcrnrlat = lat_0 + yrad,
				lat_ts = 0,
				resolution = 'i',
				suppress_ticks = True)

		return m


	# Convenience functions for working with colour ramps and bars. Taken from sensitivecities.com.
	def colorbar_index(self, mn, mx, ncolors, cmap, labels=None, **kwargs):
		"""
		This is a convenience function to stop you making off-by-one errors
		Takes a standard colour ramp, and discretizes it,
		then draws a colour bar with correctly aligned labels
		"""
		cmap = self.cmap_discretize(cmap, ncolors + 1)
		mappable = cm.ScalarMappable(cmap=cmap)
		mappable.set_array([])
		mappable.set_clim(-.5, ncolors+.5)
		colorbar = plot.colorbar(mappable, **kwargs)
		colorbar.set_ticks(np.linspace(0, ncolors, ncolors + 1))
		colorbar.set_ticklabels(np.linspace(mn, mx, ncolors + 1))

		if labels:
			colorbar.set_ticklabels(labels)
		return colorbar


	def cmap_discretize(self, cmap, N):
		"""
		Return a discrete colormap from the continuous colormap cmap.

		cmap: colormap instance, eg. cm.jet. 
		N: number of colors.

		Example
		x = resize(arange(100), (5,100))
		djet = cmap_discretize(cm.jet, 5)
		imshow(x, cmap=djet)

		"""
		if type(cmap) == str:
			cmap = cm.get_cmap(cmap)
			colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
			colors_rgba = cmap(colors_i)
			indices = np.linspace(0, 1., N + 1)
			cdict = {}
		for ki, key in enumerate(('red', 'green', 'blue')):
			cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
		return mc.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


	def draw_screen_poly(self, lats, lons, m):
	    x, y = m( lons, lats )
	    xy = list(zip(x,y))
	    poly = Polygon( xy, facecolor='red', alpha=0.4, zorder=7 )
	    plot.gca().add_patch(poly)


	# calculate heatmap and points to display for time window (self.start, self.end).
	# scatter points on basemap and plot heatmap over it.
	def display(self, save_as='data/map.png'):
		extent = [self.basemap.llcrnrlon, self.basemap.urcrnrlon, self.basemap.llcrnrlat, self.basemap.urcrnrlat]
		temp_cmap = 'winter'
		calls_cmap = 'plasma'

		# setting heatmap to show and getting data points that lie within time window
		px, py = [], []
		vx, vy, data, v= None, None, None, None

		self.set_showmap()

		px, py = self.get_time_window_points()

		# clear figure and add subplot
		plot.clf()
		fig = plot.figure()
		ax = fig.add_subplot(111, axisbg = 'w', frame_on = False)

		# load arcgis streetmap image into basemap, if none exists. Else show lower res streetmap image.
		if self.arcgisimage == None:
			self.arcgisimage = self.basemap.arcgisimage(dpi = 300, service='ESRI_StreetMap_World_2D', xpixels = 12000, zorder = 2)
			pass
		else:
			plot.imshow(self.arcgisimage.make_image(), extent = extent, origin = 'lower', zorder = 2)

		# plot red rect

		#sample = ((40.856493, -73.869578), (40.857418, -73.868612))
		#lats, lons = [sample[0][0], sample[1][0], sample[1][0], sample[0][0]], [sample[0][1], sample[0][1], sample[1][1], sample[1][1]]
		#self.draw_screen_poly(lats, lons, self.basemap)

		#self.basemap.scatter(
		#	px,
		#	py,
		#	5, marker='o', lw=.25,
		#	facecolor='#ff0000', edgecolor='w',
		#	alpha=0.9, antialiased=True,
		#	label='Emergencies', zorder=4)

		if not self.value_dataframe.empty:
			frame = self.value_dataframes[self.start]

			spatial_resolution = min((self.max_lat - self.min_lat), (self.max_long - self.min_long)) / self.numcells
	 
			vx = np.array(frame['long'].tolist())
			vy = np.array(frame['lat'].tolist())
			v = np.array(frame['value'].tolist())
			   
			yinum = (self.basemap.urcrnrlat - self.basemap.llcrnrlat) / spatial_resolution
			xinum = (self.basemap.urcrnrlon - self.basemap.llcrnrlon) / spatial_resolution
			vyi = np.linspace(self.basemap.llcrnrlat, self.basemap.urcrnrlat + spatial_resolution, xinum)
			vxi = np.linspace(self.basemap.llcrnrlon, self.basemap.urcrnrlon + spatial_resolution, yinum)
			vxi, vyi = np.meshgrid(vxi, vyi)
			xi = np.c_[vx.ravel(),vy.ravel()]
			xx = np.c_[vxi.ravel(),vyi.ravel()]
			
			data = griddata((vx, vy), v, (vxi, vyi), method='cubic')

			self.basemap.contourf(
				vxi,
				vyi,
				data,
				cmap=temp_cmap, 
				alpha=.3, 
				zorder=3, 
				extent=extent,
				vmin=32,
				antialiased=True)
		
		plot.imshow(
			self.showmap, 
			extent=extent, 
			cmap=calls_cmap, 
			alpha=.3, 
			origin='lower', 
			zorder=5,
			vmin=0,
			vmax=5)

		# copyright and source data info
		ax.text(
			.02, .85,
			'{:%m-%d-%Y\n%H:%M}'.format(self.start, self.numpoints),
			ha='left', va='bottom',
			size=24,
			fontproperties=font.FontProperties(family='sans-serif', weight='bold', size='large'),
			color='#ffffff',
			transform=ax.transAxes,
			zorder = 11)
		# unifinished hexmap implementation.
		#plot.hexbin(np.array(x), np.array(y), extent = extent, cmap = cmap, alpha = .4, gridsize = self.numcells, edgecolors = 'none')

		mn = int(self.showmap.min())
		mx = int(self.showmap.max())

		#cb = self.colorbar_index(mn, mx, mx - mn, cmap=cmap, shrink=0.5)
		#cb.ax.tick_params(labelsize=6)

		#plot.title("")
		plot.tight_layout()
		plot.savefig(save_as, dpi = 300, alpha = True)

		plot.show()

		print("Number of points: ", self.numpoints)
		if datatype == datatype.POINT:
			cell_mi_dist = self.get_cell_distance('MILES')
			cell_km_dist = self.get_cell_distance('KILOMETERS')
			print("Cell size: ", round(cell_mi_dist, 5), "miles, or ", round(cell_km_dist, 5), " kilometers.")


	# Creates 3D array of per-time-unit (1 day, 1 hour, etc.) layers, each layer an n/m cell georgraphic array, as well as corresponding data frames
	# per time-unit.
	# Sets instance variables to these objects
	def split_data_by_time(self, numcells):
		self.numcells = numcells
		xcells = int(numcells)
		ycells = int((self.basemap.urcrnrlat - self.basemap.llcrnrlat) / ((self.basemap.urcrnrlon - self.basemap.llcrnrlon) / xcells))

		d = dt.timedelta(microseconds=1)

		x = np.linspace(self.basemap.llcrnrlon, self.basemap.urcrnrlon, xcells + 1)
		y = np.linspace(self.basemap.llcrnrlat, self.basemap.urcrnrlat, ycells + 1)

		a, b = 0, 1

		while b < len(self.times):
			points = None
			start = self.times[a]
			end = self.times[b]
			# Slicing just up to but not including end is necessary since
			# Pandas is upper bound inclusive
			points = self.point_dataframe.loc[start:end - d]
			heatmap = np.zeros((len(y) - 1, len(x) - 1))

			for point in points.itertuples():
				heatmap[np.searchsorted(y, point[1]) - 1][np.searchsorted(x, point[2]) - 1] += 1

			self.heatmaps[start] = heatmap
			self.point_dataframes[start] = points

			if not self.value_dataframe.empty:
				values = self.value_dataframe.loc[start:end - d]
				self.value_dataframes[start] = values

			a += 1
			b += 1

		self.allmaps[numcells] = self.heatmaps
		self.all_pointframes[numcells] = self.point_dataframes
		self.all_valueframes[numcells] = self.value_dataframes


	# Get lists of data coordinates that lie within current time window (self.start, self.end).
	# Also reset self.numpoints to be the number of points in current time window.
	def get_time_window_points(self):
		frames = self.point_dataframes

		points_list = []
		td = dt.timedelta(hours=1)
		if self.time_units == 'DAY':
			td = dt.timedelta(days=1)
		elif self.time_units == 'MONTH':
			td = dt.timedelta(months=1)

		curdate = self.start
		while curdate < self.end:
			if curdate in frames:
				frame = frames[curdate]
				map_points = None
				map_points = pd.Series([Point(self.basemap(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(frame['long'], frame['lat'])])
					
				points_list += list(map_points.values)

			curdate += td

		raw_points = MultiPoint(points_list)

		x = np.array([geom.x for geom in raw_points])
		y = np.array([geom.y for geom in raw_points])
		self.numpoints = len(raw_points)

		return x, y


	# Set self.showmap to be heatmap corresponding to current time window.
	def set_showmap(self):
		xcells = int(self.numcells)
		ycells = int((self.basemap.urcrnrlat - self.basemap.llcrnrlat) / ((self.basemap.urcrnrlon - self.basemap.llcrnrlon) / xcells))

		td = dt.timedelta(hours=1)
		if self.time_units == 'DAY':
			td = dt.timedelta(days=1)
		elif self.time_units == 'MONTH':
			td = dt.timedelta(months=1)
		curdate = self.start

		heatmap = np.zeros((ycells, xcells))
		while curdate < self.end:
			if curdate in self.heatmaps:
				heatmap = np.add(heatmap, self.heatmaps[curdate])
			curdate += td

		self.showmap = heatmap


	# Sets self.times array to new array starting at earliest data point time rounded down to nearest specified time unit,
	# to latest data point time, rounded up, incrementing by specified time unit.
	def set_times(self, units):
		self.time_units = units

		start = dt.datetime.fromtimestamp(self.first_timestamp)
		end = dt.datetime.fromtimestamp(self.last_timestamp)

		delta = dt.timedelta(days=1)

		if units == 'HOUR':
			delta = dt.timedelta(hours=1)
			start = start.replace(minute = 0, second = 0)
			end += delta
			end = end.replace(minute = 0, second = 0)
		elif units == 'DAY':
			start = start.replace(hour = 0, minute = 0, second = 0)
			end += delta
			end = end.replace(hour = 0, minute = 0, second = 0)
		elif units == 'MONTH':
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


	# Calcualates percentage of data points that get their own cell.
	# Currently unused.
	def cell_point(self):
		return np.count_nonzero(self.showmap) / self.numpoints


	# Use Haversine Formula to calculate (approximate) heatmap cell size in miles or km.
	def get_cell_distance(self, mk):
		lon1_degrees, lon2_degrees = self.basemap.llcrnrlon, self.basemap.llcrnrlon + (self.basemap.urcrnrlon - self.basemap.llcrnrlon) / self.numcells
		lat1_degrees, lat2_degrees = (self.basemap.urcrnrlat - self.basemap.llcrnrlat) / 2, (self.basemap.urcrnrlat - self.basemap.llcrnrlat) / 2
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


	def change_numcells(self, numcells, save_as='data/map.png'):
		self.split_data_by_time(numcells)
		self.set_showmap()
		self.display(save_as)


	def change_time_units(self, units, save_as='data/map.png'):
		self.set_times(units)
		self.split_data_by_time(self.numcells)
		self.set_showmap()
		self.display(save_as)


	def change_time_window(self, start, end, save_as='data/map.png'):
		self.start = start
		self.end = end
		self.display(save_as) 


	def __init__(self, point_file_path, value_file_path=None, time_units='HOUR', winsize=10):
		rcParams['figure.figsize'] = winsize, winsize

		# size of map buffer around our data
		self.EXTRA = 0.01

		# The number of cells in the mesh will be self.numcells x self.numcells
		# Each cell will be colored based on the count of points that lie within it.
		self.numcells = 50
		self.ncolors = 5

		self.min_lat = 91
		self.max_lat = -91
		self.min_long = 181
		self.max_long = -181

		self.first_timestamp = sys.maxsize
		self.last_timestamp = 0

		self.times = []
		self.time_units = time_units
		self.start = 0
		self.end = 0

		# Dictionaries to store computed heatmaps and points for given cell sizes.
		# Key should be numcells.
		# Not currently used.
		self.allmaps = {}
		self.all_pointframes = {}
		self.all_valueframes = {}

		# array of heatmaps for current cellsize
		self.heatmaps = {}
		self.point_dataframes = {}
		self.value_dataframes = {}

		# heatmap for current time window
		self.showmap = []
		self.numpoints = 0

		self.point_data, self.value_data = self.load_data(point_file_path, value_file_path)
		self.point_dataframe = self.create_dataframe(self.point_data, datatype.POINT)
		self.value_dataframe = pd.DataFrame()
		if self.value_data:
			self.value_dataframe = self.create_dataframe(self.value_data, datatype.VALUE)

		self.basemap = self.create_basemap()
		self.arcgisimage = None

		self.set_times(time_units)
		self.split_data_by_time(self.numcells)

		self.display()


