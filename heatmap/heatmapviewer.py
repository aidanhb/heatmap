# heatmapviewer.py
# Aidan Holloway-Bidwell
# 12-09-2016

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
from shapely.geometry import Point, MultiPoint, MultiPolygon

# class for displaying a heatmodel object and its data
class heatmapviewer:

	# Initializes basemap with transverse mercator projection large enough to contain all heatmodel data points
	def create_basemap(self):
		m = Basemap(
				projection = 'tmerc',
				epsg = '4326',
				lon_0 = self.model.lon_0,
				lat_0 = self.model.lat_0,
				ellps = 'WGS84',
				llcrnrlon = self.model.llcrnrlon,
				llcrnrlat = self.model.llcrnrlat,
				urcrnrlon = self.model.urcrnrlon,
				urcrnrlat = self.model.urcrnrlat,
				lat_ts = 0,
				resolution = 'i',
				suppress_ticks = True)

		return m


	# Convenience functions for working with colour ramps and bars. Credit to sensitivecities.com.
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


	# draws a red semitransparent rectangle on basemap. Credit to Andrew Shaw on StackOverflow.com
	def draw_screen_poly(self, lats, lons, basemap):
	    x, y = basemap( lons, lats )
	    xy = list(zip(x,y))
	    poly = Polygon(
	    	xy, 
	    	facecolor='red', 
	    	alpha=0.4, 
	    	zorder=7)
	    plot.gca().add_patch(poly)


	# Calculate heatmap and points to display for time window (self.start, self.end).
	# Scatter points on basemap and plot value map over it.
	# Save resulting map in data/
	def display(self, save_as='data/map.png'):
		extent = [self.basemap.llcrnrlon, self.basemap.urcrnrlon, self.basemap.llcrnrlat, self.basemap.urcrnrlat]
		temp_cmap = 'winter'
		calls_cmap = 'plasma'

		# clear figure and add subplot
		plot.clf()
		fig = plot.figure()
		ax = fig.add_subplot(111, axisbg = 'w', frame_on = False)

		# load arcgis streetmap image into basemap, if none exists. Else show lower res streetmap image.
		# This greatly speed up the process of displating the same heatmodel over and over again.
		if self.arcgisimage == None:
			self.arcgisimage = self.basemap.arcgisimage(
				dpi = 300, 
				service='ESRI_StreetMap_World_2D', 
				xpixels = 12000, 
				zorder = 2)
			pass
		else:
			plot.imshow(self.arcgisimage.make_image(), 
				extent = extent, 
				origin = 'lower', 
				zorder = 2)

		# plot red rectangle if desired
		#sample = ((40.856493, -73.869578), (40.857418, -73.868612))
		#lats, lons = [sample[0][0], sample[1][0], sample[1][0], sample[0][0]], [sample[0][1], sample[0][1], sample[1][1], sample[1][1]]
		#self.draw_screen_poly(lats, lons, self.basemap)

		# Set heatmap to show and get data points that lie within heatmodel's current time window
		# Heatmodel's time window can be changed with heatmodel.change_time_window() fn.
		point_map = self.model.get_point_map()
		raw_coords = self.model.get_time_window_points()

		# convert raw point coordinates to basemap projection and get arrays of x and y values
		map_points = pd.Series([Point(self.basemap(mapped_x, mapped_y)) for mapped_x, mapped_y in raw_coords])
		mp = MultiPoint(map_points)
		px = np.array([geom.x for geom in mp])
		py = np.array([geom.y for geom in mp])

		# draw scatter plot and mesh heatmap
		self.basemap.scatter(
			px,
			py,
			5, marker='o', lw=.25,
			facecolor='#ff0000', edgecolor='w',
			alpha=0.9, antialiased=True,
			label='Emergencies', zorder=4)

		plot.imshow(
			point_map, 
			extent=extent, 
			cmap=calls_cmap, 
			alpha=.3, 
			origin='lower', 
			zorder=5,
			vmin=0,
			vmax=5)

		# If there is value data to show over the point heatmap, we show it.
		if not self.model.value_dataframe.empty:
			
			vxi, vyi, data = self.model.get_value_map()

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

		# Display start date and time for data we are displaying (Should be modified to show end date as well!)
		ax.text(
			.02, .85,
			'{:%m-%d-%Y\n%H:%M}'.format(self.model.start, self.model.numpoints),
			ha='left', va='bottom',
			size=24,
			fontproperties=font.FontProperties(
				family='sans-serif', 
				weight='bold', 
				size='large'),
			color='#ffffff',
			transform=ax.transAxes,
			zorder = 11)

		# Show colorbar for heatmap.
		#mn = int(point_map.min())
		#mx = int(point_map.max())
		#cb = self.colorbar_index(mn, mx, mx - mn, cmap=cmap, shrink=0.5)
		#cb.ax.tick_params(labelsize=6)

		plot.tight_layout()
		plot.savefig(
			save_as, 
			dpi = 300, 
			alpha = True)

		plot.show()

		print("Number of points: ", self.model.numpoints)
		cell_mi_dist = self.model.get_cell_distance('MILES')
		cell_km_dist = self.model.get_cell_distance('KILOMETERS')
		print("Cell size: ", round(cell_mi_dist, 5), "miles, or ", round(cell_km_dist, 5), " kilometers.")


	def __init__(self, heatmodel, winsize=10):
		rcParams['figure.figsize'] = winsize, winsize

		# size of map buffer around our data
		self.EXTRA = 0.01
		self.model = heatmodel

		self.basemap = self.create_basemap()
		self.arcgisimage = None

		self.display()