## generate data for RapidSOS data viz demo
import numpy as np
import csv
import datetime as dt
import forecastio as fc
import math
import random

def temp_maps(lat1, lat2, lon1, lon2, times):
	maps = {}

	with open('data/temps.csv', 'w') as csvfile:
		fieldnames = ['time', 'lat', 'lon', 'temp']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		cells = 31
		dl = (lon2 - lon1) / cells
		lats = np.arange(lat1, lat2, dl)
		lons = np.arange(lon1, lon2, dl)

		for time in times:
			m = np.zeros((len(lats), len(lons)))
			for i, lat in enumerate(lats):
				for j, lon in enumerate(lons):
					temp = 54
					ffront = ffront_fn(lat1, lat2, lon1, lon2, time, times[0], times[-1])[0](lon)
					if ffront < lat:
						dlat = lat - ffront
						temp = 28

					writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon), 'temp': str(temp)})
					m[i][j] = temp

			maps[time] = m

	return maps




def call_map(lat_1, lat_2, lon_1, lon_2, times, weathermaps, house_zones, road_zones):
	with open('data/calls.csv', 'w') as csvfile:
		fieldnames = ['time', 'lat', 'lon']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		full_zone = ((lat_1, lon_1), (lat_2, lon_2))

		margin = .1

		for time in times:
			numcalls = 80
			coldcalls = 50 * ((54 - weathermaps[time].mean()) / (54 - 28))
			roadcalls = numcalls * .4
			housecalls = numcalls * .5
			randcalls = numcalls * .1

			# Around rush hour
			if dt.datetime.fromtimestamp(time).hour in [7,9,16,18]:
				numcalls = 100
				roadcalls = numcalls * .8
				housecalls = numcalls * .1
				randcalls = numcalls * .1
			# During rush hour
			elif dt.datetime.fromtimestamp(time).hour in [8,17]:
				numcalls = 120
				roadcalls = numcalls * .9
				housecalls = numcalls * .05
				randcalls = numcalls * .05
			# Early morning
			elif dt.datetime.fromtimestamp(time).hour in [2,3,4,5]:
				numcalls = 50
				roadcalls = numcalls * .2
				housecalls = numcalls * .3
				randcalls = numcalls * .5
			# Workday/evening
			else:
				pass

			for x in range(0,math.ceil(roadcalls)):
				road_zone = random.choice(road_zones)
				lat1, lat2, lon1, lon2 = road_zone[0][0], road_zone[1][0], road_zone[0][1], road_zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})

			for x in range(0,math.ceil(housecalls)):
				house_zone = random.choice(house_zones)
				lat1, lat2, lon1, lon2 = house_zone[0][0], house_zone[1][0], house_zone[0][1], house_zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})

			for x in range(0,math.ceil(randcalls)):
				zone = full_zone
				lat1, lat2, lon1, lon2 = zone[0][0], zone[1][0], zone[0][1], zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})

			for x in range(0,math.ceil(coldcalls)):
				max_lon = ffront_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[1](lat_2)
				zones = [z for z in road_zones if z[1][1] <= max_lon]
				
				if zones:
					zone = random.choice(zones)
					lon1, lon2 = zone[0][1], zone[1][1]
					lon = lon1 + ((lon2 - lon1) * random.random())
					front_intersect = ffront_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](lon)
					lat1, lat2 = max(front_intersect, zone[0][0]), zone[1][0]
					lat = lat1 + (lat2 - lat1) * random.random()
					writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})

def ffront_fn(lat1, lat2, lon1, lon2, time, start, end):
	#window = 3600 * 18
	#pd = (time % window / window)
	pd = (end - time) / (end - start)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	xoffset = lon1 + ((1.0 - pd) * dlon)
	yoffset = lat2 - ((1.0 - pd) * dlat)
	def fnx(x):
		return .75 * (x - xoffset) + yoffset
	def fny(y):
		return (4.0 / 3.0) * (y - yoffset) + xoffset
	return fnx, fny

def get_times(start, end):
	return np.arange(start, end, 3600)

def main():
	api_key = "d0e27f0fd065049b196bd1668e7d55f7"

	tmp_min_lon, tmp_max_lon, tmp_min_lat, tmp_max_lat = -73.874500000, -73.86, 40.853500, 40.862014

	min_lon, max_lon, min_lat, max_lat = -73.874000, -73.862545, 40.854154, 40.861214

	road_zones = [((40.853870, -73.872458),(40.861396, -73.871675)), ((40.855172, -73.873424), (40.857664, -73.871514)), ((40.856503, -73.869658), (40.858037, -73.862545))]
	house_zones = [((40.857448, -73.870377), (40.861214, -73.862695)), ((40.854154, -73.869304), (40.856702, -73.862631))]

	start = 1479445200
	end = 1480050000

	times = get_times(start, end)
	print(times)
	tms = temp_maps(tmp_min_lat, tmp_max_lat, tmp_min_lon, tmp_max_lon, times)
	call_map(min_lat, max_lat, min_lon, max_lon, times, tms, house_zones, road_zones)

if __name__ == "__main__":
	main()