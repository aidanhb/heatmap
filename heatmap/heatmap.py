import csv
import datetime as dt
import math
import numpy as np
import pandas as pd
from pylab import rcParams
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
import sys

class heatmap:

	def load_data(self, file_path):
		output = {}
		output['lat'] = []
		output['long'] = []
		output['time'] = []

		with open(file_path, 'r') as file:
			reader = csv.reader(file)
			for row in reader:
				lat = float(row[0])
				lng = float(row[1])
				timestamp = int(row[2])
				time = dt.datetime.fromtimestamp(timestamp)

				output['lat'].append(lat)
				output['long'].append(lng)
				output['time'].append(time)

				self.min_lat = lat if lat < self.min_lat else self.min_lat
				self.max_lat = lat if lat > self.max_lat else self.max_lat
				self.min_long = lng if lng < self.min_long else self.min_long
				self.max_long = lng if lng > self.max_long else self.max_long
				self.first_timestamp = timestamp if timestamp < self.first_timestamp else self.first_timestamp
				self.last_timestamp = timestamp if timestamp > self.last_timestamp else self.last_timestamp

		return output


	def create_dataframe(self, llt_dict):
		df = pd.DataFrame(np.array([llt_dict['lat'], llt_dict['long']]).T, index = llt_dict['time'], columns = ['lat', 'long'])
		return df


	def create_basemap(self):
		side = max(self.max_long - self.min_long, self.max_lat - self.min_lat)
		rad = side / 2.0 + side * self.EXTRA
		lat_0 = (self.max_lat + self.min_lat) / 2.0
		lon_0 = (self.max_long + self.min_long) / 2.0

		m = Basemap(
				projection = 'tmerc',
				epsg = '4326',
				lon_0 = lon_0,
				lat_0 = lat_0,
				ellps = 'WGS84',
				llcrnrlon = lon_0 - rad,
				llcrnrlat = lat_0 - rad,
				urcrnrlon = lon_0 + rad,
				urcrnrlat = lat_0 + rad,
				lat_ts = 0,
				resolution = 'i',
				suppress_ticks = True)

		return m


	# Convenience functions for working with colour ramps and bars
	def colorbar_index(self, ncolors, cmap, labels=None, **kwargs):
		"""
		This is a convenience function to stop you making off-by-one errors
		Takes a standard colour ramp, and discretizes it,
		then draws a colour bar with correctly aligned labels
		"""
		cmap = self.cmap_discretize(cmap, ncolors)
		mappable = cm.ScalarMappable(cmap=cmap)
		mappable.set_array([])
		mappable.set_clim(-0.5, ncolors+0.5)
		colorbar = plot.colorbar(mappable, **kwargs)
		colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
		colorbar.set_ticklabels(range(ncolors))
		if labels.any():
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


	def display(self, start, end):

		extent = [self.basemap.llcrnrlon, self.basemap.urcrnrlon, self.basemap.llcrnrlat, self.basemap.urcrnrlat]
		self.get_time_window_heatmap()
		x, y = self.get_time_window_points()

		plot.clf()
		fig = plot.figure()
		ax = fig.add_subplot(111, axisbg = 'w', frame_on = False)

		if self.arcgisimage == None:
			self.arcgisimage = self.basemap.arcgisimage(dpi = 300, service='ESRI_StreetMap_World_2D', xpixels = 12000, zorder = 2)
		else:
			plot.imshow(self.arcgisimage.make_image(), extent = extent, origin = 'lower', zorder = 2)

		# we don't need to pass points to basemap() because we calculated using map_points and shapefile polygons
		self.scatter = self.basemap.scatter(
			x,
			y,
			5, marker='o', lw=.25,
			facecolor='#ff0000', edgecolor='w',
			alpha=0.9, antialiased=True,
			label='Emergencies', zorder=4)

		# copyright and source data info
		self.smallprint = smallprint = ax.text(
			1.03, 0,
			'Total points: %s\n$\copyright$ RapidSOS 2016' % self.numpoints,
			ha='right', va='bottom',
			size=4,
			color='#555555',
			transform=ax.transAxes,
			zorder = 5)

		a = np.random.random((16, 16))

		cmap = 'inferno'
		self.heatimage = plot.imshow(self.showmap, extent = extent, cmap = cmap, alpha = .4, interpolation = 'nearest', origin = 'lower', zorder = 3)
		#plot.show()

		cb = self.colorbar_index(ncolors=self.ncolors, cmap=cmap, shrink=0.5, labels = np.linspace(0, self.showmap.max(), self.ncolors))
		cb.ax.tick_params(labelsize=6)

		plot.title("")
		plot.tight_layout()
		plot.savefig('data/scatter.png', dpi = 100, alpha = True)

		plot.show()

		cell_mi_dist = self.get_cell_distance('MILES')
		cell_km_dist = self.get_cell_distance('KILOMETERS')
		print("Cell size: ", round(cell_mi_dist, 5), "miles, or ", round(cell_km_dist, 5), " kilometers.")


	# Creates 3D array of per-time unit layers, each layer an n/m degree georgraphic array, as well as corresponding data frames
	# per time unit
	def set_heatmaps(self, numcells):
		self.numcells = numcells

		heatmaps = []
		dataframes = []

		d = dt.timedelta(microseconds=1)

		x = np.linspace(self.basemap.llcrnrlon, self.basemap.urcrnrlon, numcells + 1)
		y = np.linspace(self.basemap.llcrnrlat, self.basemap.urcrnrlat, numcells + 1)

		a, b = 0, 1

		while b < len(self.times):
			points = None
			start = self.times[a]
			end = self.times[b]
			# Slicing just up to but not including end is necessary since
			# Pandas is upper bound inclusive
			points = self.dataframe.loc[start:end - d]
			heatmap = np.zeros((len(y) - 1, len(x) - 1))

			for point in points.itertuples():
				heatmap[np.searchsorted(y, point[1]) - 1][np.searchsorted(x, point[2]) - 1] += 1

			heatmaps.append(heatmap)
			dataframes.append(points)

			a += 1
			b += 1

		self.currentmaps = heatmaps
		self.currentframes = dataframes


	def get_time_window_points(self):

		points_list = []
		for i in range(self.start, self.end):
			frame = self.currentframes[i]
			map_points = pd.Series([Point(self.basemap(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(frame['long'], frame['lat'])])
			points_list += list(map_points.values)

		raw_points = MultiPoint(points_list)
		x = [geom.x for geom in raw_points]
		y = [geom.y for geom in raw_points]

		self.numpoints = len(raw_points)

		return x, y


	def get_time_window_heatmap(self):

		heatmap = np.zeros((self.numcells, self.numcells))
		for i in range(self.start, self.end):
			if (i < len(self.currentmaps)):
				heatmap = np.add(heatmap, self.currentmaps[i])

		self.showmap = heatmap


	def set_times(self, units):
		start = dt.datetime.fromtimestamp(self.first_timestamp)
		end = dt.datetime.fromtimestamp(self.last_timestamp)

		delta = dt.timedelta(days=1)

		if units == 'HOUR':
			start = start.replace(minute = 0, second = 0)
			end = end.replace(hour = end.hour + 1, minute = 0, second = 0)
			delta = dt.timedelta(hours=1)
		elif units == 'DAY':
			start = start.replace(hour = 0, minute = 0, second = 0)
			end = end.replace(day = end.day + 1, hour = 0, minute = 0, second = 0)
		elif units == 'MONTH':
			start = start.replace(day = 0, hour = 0, minute = 0, second = 0)
			end = end.replace(month = end.month + 1, day = 0, hour = 0, minute = 0, second = 0)
			delta = dt.timedelta(months=1)

		times = []
		curr = start
		while curr <= end:
			times.append(curr)
			curr += delta

		self.times = times
		self.start = 0
		self.end = len(self.times) - 1


	def cell_point(self):
		return np.count_nonzero(self.showmap) / self.numpoints


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


	def change_numcells(self, numcells):
		self.set_heatmaps(numcells)
		self.display(self.start, self.end)


	def change_time_units(self, units):
		self.set_times()
		self.display(self.start, self.end)


	def change_time_window(self, start, end):
		self.start = start
		self.end = end
		self.display(self.start, self.end) 


	def __init__(self, file_path, time_units='HOUR', winsize=8):
		#mpl.rc('savefig', dpi=1200)
		rcParams['figure.figsize'] = winsize, winsize

		self.EXTRA = 0.01

		self.numcells = 30

		self.ncolors = 5
		self.min_lat = 91
		self.max_lat = -91
		self.min_long = 181
		self.max_long = -181
		self.first_timestamp = sys.maxsize
		self.last_timestamp = 0
		self.times = []
		self.start = 0
		self.end = 0
		self.heatmaps = {}
		self.dataframes = {}
		# array of heatmaps for current cellsize
		self.currentmaps = []
		self.currentframes = []
		# heatmap for current time window
		self.showmap = []
		self.numpoints = 0

		self.data = self.load_data(file_path)
		self.dataframe = self.create_dataframe(self.data)
		self.basemap = self.create_basemap()

		self.arcgisimage = None
		self.scatter = None
		self.heatimage = None
		self.smallprint = None

		self.set_times(time_units)

		self.set_heatmaps(self.numcells)

		self.display(self.start, self.end)


