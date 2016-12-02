## generate data for RapidSOS data viz demo
import numpy as np
import csv

def temp_maps(lat1, lat2, lon1, lon2, times):
	with open('data/temps.csv', 'w') as csvfile:
		fieldnames = ['time', 'lat', 'lon', 'temp']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		cells = 50
		dl = (lon2 - lon1) / cells
		lats = np.arange(lat1, lat2, dl)
		lons = np.arange(lon1, lon2, dl)
		width = (lat2 - lat1) * .75

		for time in times:
			m = np.zeros((len(lats), len(lons)))
			for lat in lats:
				for lon in lons:
					temp = 54
					ffront = ffront_fn(lat1, lat2, lon1, lon2, time, times[0], times[-1])(lon)
					bfront = ffront + width
					if ffront < lat and lat < bfront:
						dlat = min(lat - ffront, bfront - lat)
						temp = 28
						if dlat < cells / 40:
							temp += (12 * (1 - ((.5 + (np.random.random()) / 2) * (dlat / (cells / 20)))))

					writer.writerow({'time': str(time), 'lat': str(lat), 'lon': str(lon), 'temp': str(temp)})




def call_map(lat1, lat2, lon1, lon2, times):
	return

def ffront_fn(lat1, lat2, lon1, lon2, time, start, end):
	window = 3600 * 18
	pd = (time % window / window)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	def fn(x):
		xoffset = lon1 + (pd * dlon)
		yoffset = lat2 - (pd * dlat)
		return (x - xoffset)**2 + yoffset
	return fn

def get_times(start, end):
	return np.arange(start, end, 3600)

def main():
	min_lat = 40.841461
	min_lon = -73.900074
	max_lat = 40.864605
	max_lon = -73.839220
	start = 1479427200
	end = 1480032000

	times = get_times(start, end)
	temp_maps(min_lat, max_lat, min_lon, max_lon, times)

if __name__ == "__main__":
	main()