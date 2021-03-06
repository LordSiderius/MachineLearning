from datetime import datetime
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import folium
from IPython.display import display, HTML
from folium import plugins
import copy
from math import cos

class DataExtractor(object):

    def __init__(self, map_range, resolution=200):

        dataset = pd.read_csv('database', names=['UTC', 'latitude', 'longitude'], sep=',')


        dataset = dataset.groupby(['UTC', 'longitude', 'latitude']).size().reset_index(name='duration')

        dataset = dataset.drop_duplicates()#.reset_index(drop=True, inplace=True)

        dataset['duration'] = dataset['duration'].apply(lambda x: x * 15)
        dataset['UTC'] = dataset['UTC'].apply(lambda x: x / 1000)
        dataset['day'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%d')))
        dataset['month'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%m')))
        dataset['year'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%Y')))
        dataset['weekday'] = dataset['UTC'].apply(lambda x: datetime.utcfromtimestamp(x).weekday())
        dataset['hour'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%H')))
        dataset['minute'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%M')))
        dataset['timeofday'] = (dataset['hour'] + dataset['minute'] / 60.0) / 24.0

        dataset = dataset.reset_index(drop=True)

        dataset = dataset[dataset['latitude'] > map_range[0]]
        dataset = dataset[dataset['latitude'] < map_range[1]]
        dataset = dataset[dataset['longitude'] > map_range[2]]
        dataset = dataset[dataset['longitude'] < map_range[3]]


        lat_min = dataset['latitude'].min()
        lat_max = dataset['latitude'].max()
        long_min = dataset['longitude'].min()
        long_max = dataset['longitude'].max()


        lat_step, lon_step = self.coordinates_step(lat_min, lat_max, long_min, long_max, resolution)

        n = np.floor((lat_max - lat_min) / lat_step)
        m = np.floor((long_max - long_min) / lon_step)

        self.lat_borders = np.linspace(lat_min, lat_max, int(n))
        self.long_borders = np.linspace(long_min, long_max, int(m))

        dataset['height_idx'] = dataset['latitude'].apply(lambda x: self.get_index(self.lat_borders, x))
        dataset['width_idx'] = dataset['longitude'].apply(lambda x: self.get_index(self.long_borders, x))

        self.dataset = dataset


    def mapping_data(self, gps_data):
        x, y = [], []
        for i in range(len(gps_data)):
            x.append(gps_data[i][0])
            y.append(gps_data[i][1])

        return x, y


    def get_index(self, borders, value):
        for i in range(len(borders)):
            if value <= borders[i]:
                return i

        return None

    def coordinates_step(self, lat0, lat1, lon0, lon1, step_meters=200):
        '''Function to calculate approximate size of step on map from meters to degrees'''
        # earth's radius in meters
        R = 6378000

        lat_step = step_meters / R / 3.1416 * 180
        lon_step = step_meters / (R * cos(abs(lat0 + lat1)/180 * 3.1416 / 2)) / 3.1416 * 180

        return lat_step, lon_step

    def data_visualisation(self, range=(13.978658, 14.879769, 49.896474, 50.260075) ):
        gps_data = tuple(zip(self.dataset['latitude'].values, self.dataset['longitude'].values))
        #

        fig, ax = plt.subplots()

        y, x = self.mapping_data(gps_data)

        ax.scatter(x, y, edgecolors='red', linewidths=2, zorder=2)
        ax.imshow(mpimg.imread('map.png'), extent=range)

        plt.show()

    def data_visualisation2(self, range=(13.978658, 14.879769, 49.896474, 50.260075), day=0):

        dataset = copy.copy(self.dataset[self.dataset['weekday'] == day])
        # Mark events with names on map
        map = folium.Map([(range[2]+range[3])/2, (range[0]+range[1])/2], zoom_start=14)
        a = 0
        # for index, franchise in self.dataset.iterrows():
        #     a += 1
        #     location = [franchise['latitude'], franchise['longitude']]
        #     folium.Marker(location, popup=f'Name:Police').add_to(map)
        #     if a > 1000:
        #         break

        dfmatrix = dataset[['latitude', 'longitude']].values
        # plot heatmap
        map.add_child(plugins.HeatMap(dfmatrix, radius=15))
        path = 'index%d.html' % day
        map.save(path)

    def visualization_by_sameday(self):

        data = self.dataset['UTC'].groupby([self.dataset['height_idx'], self.dataset['width_idx']]).count().plot(kind='bar')
        plt.show()
        # print(data.head(30))
        # dataset = copy.copy(self.dataset[self.dataset['weekday'] == day])






if __name__ == '__main__':
    map_range = (50.07, 50.1, 14.42, 14.45)
    data_ex = DataExtractor(map_range, 20)
    print(data_ex.dataset[['UTC', 'height_idx', 'width_idx', 'duration']].head(5))
    # data_ex.visualization_by_sameday()
    # data_ex.data_visualisation()
    data_ex.data_visualisation2(day=0)
    # data_ex.data_visualisation2(day=1)
    # data_ex.data_visualisation2(day=2)
    # data_ex.data_visualisation2(day=3)
    # data_ex.data_visualisation2(day=4)
    # data_ex.data_visualisation2(day=5)
    # data_ex.data_visualisation2(day=6)
