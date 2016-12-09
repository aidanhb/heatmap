**Heat Map for Geotemporal Data Visualization**

This repository contains classes for interpreting simple coordinate-based data. Currently, the data must be in .csv
files and be of the format (unix timestamp, latitude, longitude, value (optional)). Initilizing a heatmodel python class
with a .csv file (or files) will load the data into dataframe instance variables and allow access to subsets of the data via the API.
Currently supported methods are:
> **change_time_window(start, end)** changes the time window of the subset of the total data in the heatmodel we are displaying or examining.

> **change_numcells(numcells)** the heatmodel partitions the geographic data temporally, but also based on a grid mesh laid over the data.
Point data counts will be summed and returned for each cell in the mesh, and for value data a griddata cubic interpolation will be returned.

> **change_time_units(units)** repartitions data based on the new time unit passed.

This heatmodel object can be viewed using the heatmapviewer class, specifically the display() function, which just requires 
initialization with a heatmodel.

Finally, there is a Jupyter notebook (HeatMap.ipynb) with a demo on creating and visualizing fake data.
