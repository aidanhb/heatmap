import csv
import datetime as dt
import math
import numpy as np
import pandas as pd
from pylab import rcParams
import pyproj
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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

		#print(m.llcrnrlat, m.llcrnrlon)
		#print(m.urcrnrlat, m.urcrnrlon)
		#print()

		return m


	# Convenience functions for working with colour ramps and bars
	def colorbar_index(ncolors, cmap, labels=None, **kwargs):
		"""
		This is a convenience function to stop you making off-by-one errors
		Takes a standard colour ramp, and discretizes it,
		then draws a colour bar with correctly aligned labels
		"""
		cmap = cmap_discretize(cmap, ncolors)
		mappable = cm.ScalarMappable(cmap=cmap)
		mappable.set_array([])
		mappable.set_clim(-0.5, ncolors+0.5)
		colorbar = plot.colorbar(mappable, **kwargs)
		colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
		colorbar.set_ticklabels(range(ncolors))
		if labels:
			colorbar.set_ticklabels(labels)
		return colorbar


	def cmap_discretize(cmap, N):
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
			cmap = get_cmap(cmap)
			colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
			colors_rgba = cmap(colors_i)
			indices = np.linspace(0, 1., N + 1)
			cdict = {}
		for ki, key in enumerate(('red', 'green', 'blue')):
			cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
		return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


	def display(self, dataframe, basemap, numcells, start, end):
		map_points = pd.Series([Point(basemap(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(dataframe['long'], dataframe['lat'])])
		emergency_points = MultiPoint(list(map_points.values))
		heatmaps = self.get_heatmaps(basemap, dataframe, numcells)

		heatmap = np.zeros((numcells, numcells))

		for i in range(start, end):
			if (i < len(heatmaps)):
				heatmap = np.add(heatmap, heatmaps[i])

		x = [geom.x for geom in emergency_points]
		y = [geom.y for geom in emergency_points]

		plot.clf()
		fig = plot.figure()
		ax = fig.add_subplot(111, axisbg = 'w', frame_on = False)

		basemap.arcgisimage(service='ESRI_StreetMap_World_2D', xpixels = 12000, verbose = True, zorder = 2)

		# we don't need to pass points to basemap() because we calculated using map_points and shapefile polygons
		basemap.scatter(
			x,
			y,
			5, marker='o', lw=.25,
			facecolor='#ff0000', edgecolor='w',
			alpha=0.9, antialiased=True,
			label='Emergencies', zorder=3)

		# plot boroughs by adding the PatchCollection to the axes instance
		# ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

		# copyright and source data info
		smallprint = ax.text(
			1.03, 0,
			'Total points: %s\n$\copyright$ RapidSOS 2016' % len(emergency_points),
			ha='right', va='bottom',
			size=4,
			color='#555555',
			transform=ax.transAxes)

		extent = [basemap.llcrnrlon, basemap.urcrnrlon, basemap.llcrnrlat, basemap.urcrnrlat]

		a = np.random.random((16, 16))

		plot.imshow(heatmap, extent = extent, cmap = 'inferno', alpha = .4, interpolation = 'nearest', origin = 'lower')
		#plot.show()

		plot.title("")
		plot.tight_layout()
		plot.savefig('data/scatter.png', dpi = 100, alpha = True)

		plot.show()


	# Creates 3D array of per-hour layers, each layer an n/m degree georgraphic array
	def get_heatmaps(self, basemap, dataframe, numcells):
		#h = basemap.urcrnrlat - basemap.llcrnrlat
		#w = basemap.urcrnrlon - basemap.llcrnrlon
		heatmaps = []
		d = dt.timedelta(microseconds=1)

		#print(basemap.llcrnrlon, basemap.urcrnrlon)

		x = np.linspace(basemap.llcrnrlon, basemap.urcrnrlon, numcells + 1)
		y = np.linspace(basemap.llcrnrlat, basemap.urcrnrlat, numcells + 1)

		#print(self.min_lat, self.min_long)
		#print(self.max_lat, self.max_long)
		#print(x,y)
		#print()

		a, b = 0, 1

		while b < len(self.times):
			points = None
			start = self.times[a]
			end = self.times[b]
				# Slicing just up to but not including end is necessary since
				# Pandas is upper bound inclusive
			points = dataframe.loc[start:end - d]
			heatmap = np.zeros((len(y) - 1, len(x) - 1))

			for point in points.itertuples():
				#print(point)
				#print(point[1], point[2])
				heatmap[np.searchsorted(y, point[1]) - 1][np.searchsorted(x, point[2]) - 1] += 1

			heatmaps.append(heatmap)
			a += 1
			b += 1

			#print(heatmap)

		return heatmaps


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


	def __init__(self, file_path, time_units, win_height, win_width):

		rcParams['figure.figsize'] = win_height, win_width

		self.EXTRA = 0.01

		self.min_lat = 91
		self.max_lat = -91
		self.min_long = 181
		self.max_long = -181
		self.first_timestamp = sys.maxsize
		self.last_timestamp = 0
		self.times = []

		data = self.load_data(file_path)
		dataframe = self.create_dataframe(data)
		basemap = self.create_basemap()

		self.set_times(time_units)

		#print(self.times)
		self.display(dataframe, basemap, 30, 0, 7)


