import datatype
import datetime as dt
import heatmodel as model
import math
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.patches import Polygon
import matplotlib.pyplot as plot
from matplotlib.collections import PatchCollection
import matplotlib.font_manager as font
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from pylab import rcParams
from scipy.interpolate import griddata
from shapely.geometry import Point, MultiPoint, MultiPolygon
import sys
import timeunit as unit

class heatmap:

	# Initializes basemap with transverse mercator projection large enough to contain all call data points
	def create_basemap(self, heatmodel):
		xside = heatmodel.max_long - heatmodel.min_long
		yside = heatmodel.max_lat - heatmodel.min_lat
		buf = max(xside * self.EXTRA, yside * self.EXTRA)
		xrad = xside / 2.0 + buf
		yrad = yside / 2.0 + buf 
		lat_0 = (heatmodel.max_lat + heatmodel.min_lat) / 2.0
		lon_0 = (heatmodel.max_long + heatmodel.min_long) / 2.0

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
	def display(self, heatmodel, save_as='data/map.png'):
		extent = [self.basemap.llcrnrlon, self.basemap.urcrnrlon, self.basemap.llcrnrlat, self.basemap.urcrnrlat]
		temp_cmap = 'winter'
		calls_cmap = 'plasma'

		# setting heatmap to show and getting data points that lie within time window
		px, py = [], []
		vx, vy, data, v= None, None, None, None

		point_map = model.get_showmap()
		px, py = model.get_time_window_points()

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

		self.basemap.scatter(
			px,
			py,
			5, marker='o', lw=.25,
			facecolor='#ff0000', edgecolor='w',
			alpha=0.9, antialiased=True,
			label='Emergencies', zorder=4)

		if not heatmodel.value_dataframe.empty:
			frame = heatmodel.value_dataframes[heatmodel.start]

			spatial_resolution = min((heatmodel.max_lat - heatmodel.min_lat), (heatmodel.max_long - heatmodel.min_long)) / heatmodel.numcells
	 
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
			point_map, 
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
			'{:%m-%d-%Y\n%H:%M}'.format(heatmodel.start, heatmodel.numpoints),
			ha='left', va='bottom',
			size=24,
			fontproperties=font.FontProperties(family='sans-serif', weight='bold', size='large'),
			color='#ffffff',
			transform=ax.transAxes,
			zorder = 11)
		# unifinished hexmap implementation.
		#plot.hexbin(np.array(x), np.array(y), extent = extent, cmap = cmap, alpha = .4, gridsize = self.numcells, edgecolors = 'none')

		mn = int(point_map.min())
		mx = int(point_map.max())

		#cb = self.colorbar_index(mn, mx, mx - mn, cmap=cmap, shrink=0.5)
		#cb.ax.tick_params(labelsize=6)

		#plot.title("")
		plot.tight_layout()
		plot.savefig(save_as, dpi = 300, alpha = True)

		plot.show()

		print("Number of points: ", heatmodel.numpoints)
		if datatype == datatype.POINT:
			cell_mi_dist = model.get_cell_distance('MILES')
			cell_km_dist = model.get_cell_distance('KILOMETERS')
			print("Cell size: ", round(cell_mi_dist, 5), "miles, or ", round(cell_km_dist, 5), " kilometers.")


	def __init__(self, heatmodel, winsize=10):
		rcParams['figure.figsize'] = winsize, winsize

		# size of map buffer around our data
		self.EXTRA = 0.01
		self.model = heatmodel

		self.basemap = self.create_basemap(self.model)
		self.arcgisimage = None

		self.display(self.model)


