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

class Data_extractor(object):

    def shit(self, a, b):
        return a + b / 60.0

    def __init__(self, resolution=200):
        n = 50000 / resolution  # width of map - 50km split by 200 meters
        m = 75000 / resolution  # height of map - 75km split by 200 meters

        dataset = pd.read_csv('database', names=['UTC', 'latitude', 'longitude'], sep=',')

        dataset = dataset.drop_duplicates()

        dataset['UTC'] = dataset['UTC'].apply(lambda x: x / 1000)
        dataset['day'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%d')))
        dataset['month'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%m')))
        dataset['year'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%Y')))
        dataset['weekday'] = dataset['UTC'].apply(lambda x: datetime.utcfromtimestamp(x).weekday())
        dataset['hour'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%H')))
        dataset['minute'] = dataset['UTC'].apply(lambda x: float(datetime.utcfromtimestamp(x).strftime('%M')))
        dataset['timeofday'] = (dataset['hour'] + dataset['minute'] / 60.0) / 24.0


        lat_min = dataset['latitude'].min()
        lat_max = dataset['latitude'].max()
        long_min = dataset['longitude'].min()
        long_max = dataset['longitude'].max()

        lat_borders = list(np.linspace(lat_min, lat_max, int(n)))
        long_borders = np.linspace(long_min, long_max, int(m))

        dataset['height_idx'] = dataset['latitude'].apply(lambda x: self.get_index(lat_borders, x))
        dataset['width_idx'] = dataset['longitude'].apply(lambda x: self.get_index(long_borders, x))

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
    data_ex = Data_extractor(200)
    # data_ex.visualization_by_sameday()
    # data_ex.data_visualisation()
    data_ex.data_visualisation2(day=0)
    # data_ex.data_visualisation2(day=1)
    # data_ex.data_visualisation2(day=2)
    # data_ex.data_visualisation2(day=3)
    # data_ex.data_visualisation2(day=4)
    # data_ex.data_visualisation2(day=5)
    # data_ex.data_visualisation2(day=6)
