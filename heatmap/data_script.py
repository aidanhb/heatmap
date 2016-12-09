## generate data for RapidSOS data viz demo
import numpy as np
import csv
import datetime as dt
import math
import random

def temp_maps(lat1, lat2, lon1, lon2, times, prediction=False):
	# We'll be not only writing our data to files, but returning it in dictionaries to analyze it
	# using other functions in this program
	maps = {}
	marrays = {}
	filename = 'data/temps.csv'
	if prediction:
		filename = 'data/predicted_temps.csv'

	with open(filename, 'w') as csvfile:
		fieldnames = ['time', 'lat', 'lon', 'temp']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		cells = 31
		dl = (lon2 - lon1) / cells
		lats = np.arange(lat1, lat2, dl)
		lons = np.arange(lon1, lon2, dl)

		for time in times:
			m = []
			marray = np.zeros((len(lats), len(lons)))
			for i, lat in enumerate(lats):
				for j, lon in enumerate(lons):
					temp = 54
					ffront = obs_fn(lat1, lat2, lon1, lon2, time, times[0], times[-1])[0](lon)

					if prediction:
						temp = 28
						ffront = pred_fn(lat1, lat2, lon1, lon2, time, times[0], times[-1])[0](lon)
						bfront = pred_fn(lat1, lat2, lon1, lon2, time, times[0], times[-1])[0](lon) + (lat2 - lat1) * .75
						if ffront < lat and lat < bfront:
							temp = 54
						dlat = min(lat - ffront, bfront - lat)
						if abs(dlat) <= (lat2 - lat1) / 5:
							if dlat > 0:
								temp -= 26 * (1 - abs(dlat) / ((lat2 - lat1) / 5))**4
							else:
								temp += 26 * (1 - abs(dlat) / ((lat2 - lat1) / 5))**4

					else:
						if ffront < lat:
							temp = 28
						dlat = lat - ffront
						if abs(dlat) <= (lat2 - lat1) / 5:
							if dlat < 0:
								temp -= 26 * (1 - abs(dlat) / ((lat2 - lat1) / 5))**4
							else:
								temp += 26 * (1 - abs(dlat) / ((lat2 - lat1) / 5))**4

					writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon), 'temp': str(temp)})
					m.append(((lat, lon), temp))
					marray[i][j] = temp

			maps[time] = m
			marrays[time] = marray

	return maps, marrays




def call_map(lat_1, lat_2, lon_1, lon_2, times, weathermaps, house_zones, road_zones, prediction=False):
	m = {}
	filename = 'data/calls.csv'
	if prediction:
		filename = 'data/predicted_calls.csv'

	with open(filename, 'w') as csvfile:
		fieldnames = ['time', 'lat', 'lon']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		full_zone = ((lat_1, lon_1), (lat_2, lon_2))

		margin = .1

		for time in times:
			numcalls = 50
			coldcalls = 150
			roadcalls = numcalls * .4
			housecalls = numcalls * .5
			randcalls = numcalls * .1

			# Around rush hour
			if dt.datetime.fromtimestamp(time).hour in [7,9,16,18]:
				numcalls = 50
				coldcalls = 200
				roadcalls = numcalls * .8
				housecalls = numcalls * .1
				randcalls = numcalls * .1
			# During rush hour
			elif dt.datetime.fromtimestamp(time).hour in [8,17]:
				numcalls = 75
				coldcalls = 250
				roadcalls = numcalls * .9
				housecalls = numcalls * .05
				randcalls = numcalls * .05
			# Early morning
			elif dt.datetime.fromtimestamp(time).hour in [2,3,4,5]:
				numcalls = 30
				roadcalls = numcalls * .2
				housecalls = numcalls * .3
				randcalls = numcalls * .5
			# Workday/evening
			else:
				pass

			total_rd_A = 0
			frac_rd_A = 0
			cold_zones = [z for z in road_zones if z[0][0] >= obs_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](z[0][1])]
			if prediction:
				cold_zones = [z for z in road_zones if \
				z[0][0] <= pred_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](z[0][1])\
				or z[1][0] >= pred_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](z[0][1]) + (lat_2 - lat_1) * .75]

			for z in road_zones:
				total_rd_A += (z[1][0] - z[0][0]) * (z[1][1] - z[0][1])
			for z in cold_zones:
				frac_rd_A += (z[1][0] - z[0][0]) * (z[1][1] - z[0][1])

			coldcalls *= (frac_rd_A / total_rd_A)

			points = []

			for x in range(0,math.ceil(roadcalls)):
				road_zone = random.choice(road_zones)
				lat1, lat2, lon1, lon2 = road_zone[0][0], road_zone[1][0], road_zone[0][1], road_zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
				points.append((lat, lon))

			for x in range(0,math.ceil(housecalls)):
				house_zone = random.choice(house_zones)
				lat1, lat2, lon1, lon2 = house_zone[0][0], house_zone[1][0], house_zone[0][1], house_zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
				points.append((lat, lon))

			for x in range(0,math.ceil(randcalls)):
				zone = full_zone
				lat1, lat2, lon1, lon2 = zone[0][0], zone[1][0], zone[0][1], zone[1][1]
				lat = lat1 + ((lat2 - lat1) * random.random())
				lon = lon1 + ((lon2 - lon1) * random.random())
				writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
				points.append((lat, lon))

			for x in range(0,math.ceil(coldcalls)):

				if cold_zones:
					zone = random.choice(cold_zones)
					lon1, lon2 = zone[0][1], zone[1][1]
					lon = lon1 + ((lon2 - lon1) * random.random())
					front_intersect = obs_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](lon)
					if prediction:
						front_top = pred_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](lon) + (lat_2 - lat_1) * .75
						front_intersect = pred_fn(lat_1, lat_2, lon_1, lon_2, time, times[0], times[-1])[0](lon)
						if zone[0][0] <= front_intersect:
							lat1, lat2 = zone[0][0], min(zone[1][0], front_intersect)
							lat = lat1 + (lat2 - lat1) * random.random()
							writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
							points.append((lat, lon))
						else: # zone[1][0] >= front_top
							lat1, lat2 = max(zone[0][0], min(front_top, lat_2)), zone[1][0]
							lat = lat1 + (lat2 - lat1) * random.random()
							writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
							points.append((lat, lon))
						
					else:
						lat1, lat2 = max(front_intersect, zone[0][0]), zone[1][0]
						lat = lat1 + (lat2 - lat1) * random.random()
						writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon)})
						points.append((lat, lon))

			m[time] = points

	return m

def ave_hourly_temp(lat1, lat2, lon1, lon2, weathermaps):
	times = []
	temps = []
	for time in weathermaps:
		points = weathermaps[time]
		s = 0
		count = 0
		for point in points:
			if (point[0][0] < lat2 and point[0][0] > lat1) and (point[0][1] < lon2 and point[0][1] > lon1):
				s += point[1]
				count += 1

		ave = s / count

		index = np.searchsorted(times, time)
		times.insert(index, time)
		temps.insert(index, ave)

	return times, temps

def ave_hourly_calls(lat1, lat2, lon1, lon2, points_per_time):
	times = []
	num_points = []
	for time in points_per_time:
		points = points_per_time[time]
		count = 0
		for point in points:
			if (point[0] < lat2 and point[0] > lat1) and (point[1] < lon2 and point[1] > lon1):
				count += 1

		index = np.searchsorted(times, time)
		times.insert(index, time)
		num_points.insert(index, count)

	return times, num_points

def obs_fn(lat1, lat2, lon1, lon2, time, start, end):
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

def pred_fn(lat1, lat2, lon1, lon2, time, start, end):
	pd = (end - time) / (end - start)
	dlon = lon2 - lon1
	dlat = (lat2 - lat1) * 1.75
	xoffset = (lon2 + lon1) / 2
	yoffset = lat2 - ((1.0 - pd) * dlat)
	def fnx(x):
		return 220 * (x - xoffset) ** 2 + yoffset
	def fny(y):
		return (4.0 / 3.0) * (y - yoffset) + xoffset # Incorrect, but not currently used
	return fnx, fny

def get_times(start, end):
	return np.arange(start, end, 3600)

def main(start, end, prediction=False):
	#api_key = "d0e27f0fd065049b196bd1668e7d55f7"

	tmp_min_lon, tmp_max_lon, tmp_min_lat, tmp_max_lat = -73.874500000, -73.86, 40.853500, 40.862014

	min_lon, max_lon, min_lat, max_lat = -73.874000, -73.862545, 40.854154, 40.861214

	# rectangles where calls are more likely to occur
	n_brx_onramp = ((40.858260, -73.872887), (40.859234, -73.872287))
	s_brx_onramp = ((40.854026, -73.872088), (40.854724, -73.871638))
	s_boston_whiteplains_int = ((40.857937, -73.868290), (40.859333, -73.867453))
	n_boston_whiteplains_int = ((40.859333, -73.867453), (40.861313, -73.866681))
	whiteplains_trp_int = ((40.856201, -73.867840), (40.858067, -73.867453))
	pkwys = ((40.856503, -73.869658), (40.858037, -73.862545))
	park_entrance = ((40.856363, -73.870329), (40.857288, -73.868676))
	snow_skid_clover = ((40.855178, -73.873333), (40.857369, -73.871530))

	road_zones = [n_brx_onramp, s_brx_onramp, s_boston_whiteplains_int, n_boston_whiteplains_int, whiteplains_trp_int, pkwys, park_entrance, snow_skid_clover]
	house_zones = [((40.857448, -73.870377), (40.861214, -73.862695)), ((40.854154, -73.869304), (40.856702, -73.862631))]

	times = get_times(start, end)
	#print(times)
	tms, tarrays = temp_maps(tmp_min_lat, tmp_max_lat, tmp_min_lon, tmp_max_lon, times, prediction)
	cms = call_map(min_lat, max_lat, min_lon, max_lon, times, tarrays, house_zones, road_zones, prediction)

	# rectangle of focus for creation of ave_temp/call count line plot
	sample = ((40.856493, -73.869578), (40.857418, -73.868612))
	l, r, b, t = sample[0][1], sample[1][1], sample[0][0], sample[1][0]

	tmp_times, tmps = ave_hourly_temp(b, t, l, r, tms)
	pt_times, pts = ave_hourly_calls(b, t, l, r, cms)

	return tmp_times, tmps, pt_times, pts

if __name__ == "__main__":
	main()