from datetime import datetime
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
# Basic Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from datetime import timedelta
from data_extractor import DataExtractor
import copy

# Mapping

import geopy
from geopy.geocoders import Nominatim
import folium
from geopy.extra.rate_limiter import RateLimiter
from folium import plugins
from folium.plugins import MarkerCluster
# Statistical OLS Regression Analysis
import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
#Scipy sklearn Predictions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor



class DataAnalyzer(object):

    def __init__(self, dataset):
        self.dataset = dataset.copy()
        pass


    def select_data(self, lat_borders, long_borders):
        print(self.dataset.groupby(['height_idx', 'width_idx', 'weekday'])[['height_idx', 'width_idx', 'weekday', 'timeofday']].head(35))
        print(self.dataset['timeofday'][(self.dataset['height_idx'] == 12) & (self.dataset['width_idx'] == 6)].head())
        print((lat_borders[12] + lat_borders[13] )/ 2, (long_borders[6] + long_borders[7]) / 2)


if __name__ == '__main__':

    # map_range = (13.978658, 14.879769, 49.896474, 50.260075)
    map_range = (50.07, 50.1, 14.42, 14.45)
    data_extractor = DataExtractor(map_range)
    data_analyzer = DataAnalyzer(data_extractor.dataset)
    data_analyzer.select_data(data_extractor.lat_borders, data_extractor.long_borders)
