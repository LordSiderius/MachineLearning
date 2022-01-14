'''Script for visualisation of data with events in grid'''

import numpy as np
from data_extractor import DataExtractor
import folium
import matplotlib as mpl
import webbrowser
from math import cos

class MapGenerator(object):

    def __init__(self, df, map_range=(50.07, 50.1, 14.42, 14.45)):

        # calculate values for the grid
        self.x = df.copy()
        self.lat0 = (self.x['latitude'].max() + self.x['latitude'].min()) / 2
        self.lon0 = (self.x['longitude'].max() + self.x['longitude'].min()) / 2

        self.lat_step, self.lon_step = self.coordinates_step(*map_range, 200)

        self.x['lat'] = np.floor(self.x['latitude'] / self.lat_step) * self.lat_step
        self.x['lon'] = np.floor(self.x['longitude'] / self.lon_step) * self.lon_step

        self.x = self.x.groupby(['lat', 'lon'])['UTC'].count()
        self.x /= self.x.max()
        self.x = self.x.reset_index()

    def surface_distance(self, lat0, lat1, lon0, lon1):
        '''Function to calcualte approximate size of map square by coordinates'''
        # earth's radius in meters
        R = 6378000

        # latitude distance calc
        lat_len = abs(lat0 - lat1)/180 * 3.1416 * R
        # longtitude  distance cal / approximation on avergae latitude
        lon_len = abs(lon0 - lon1)/180 * 3.1416 * R * cos(abs(lat0 + lat1)/180 * 3.1416 / 2)

        return lat_len, lon_len

    def coordinates_step(self, lat0, lat1, lon0, lon1, step_meters=200):
        '''Function to calculate approximate size of step on map from meters to degrees'''
        # earth's radius in meters
        R = 6378000

        lat_step = step_meters / R / 3.1416 * 180
        lon_step = step_meters / (R * cos(abs(lat0 + lat1)/180 * 3.1416 / 2)) / 3.1416 * 180

        return lat_step, lon_step


    # geo_json returns a single square
    def geo_json(self, lat, lon, value, lat_step, lon_step):
        cmap = mpl.cm.RdBu
        return {
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": {
                'color': 'white',
                'weight': 1,
                'fillColor': mpl.colors.to_hex(cmap(value)),
                'fillOpacity': 0.5,
              },
              "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon, lat],
                    [lon, lat + lat_step],
                    [lon + lon_step, lat + lat_step],
                    [lon + lon_step, lat],
                    [lon, lat],
                  ]]}}]}

    def generate_map(self):

        # generating a map...
        m = folium.Map(location=[self.lat0, self.lon0], zoom_start=11)

        # ...with squares...
        for _, xi in self.x.iterrows():
            folium.GeoJson(self.geo_json(xi['lat'], xi['lon'], xi['UTC'], self.lat_step, self.lon_step),
                           lambda x: x['properties']).add_to(m)

        # # ...and the original points
        # for elt in list(zip(df.latitude, df.longitude, df.UTC)):
        #     folium.Circle(elt[:2], color="white", radius=elt[2]).add_to(m)

        m.save('squares.html')

        webbrowser.register('chrome', None,	webbrowser.BackgroundBrowser(
            "C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
        webbrowser.get('chrome').open(
            'file:///E://Python_project//Machine_Learning//prague-police-heatmap-main//squares.html')


if __name__ == '__main__':

    map_range = (50.07, 50.1, 14.42, 14.45)
    data_ex = DataExtractor(map_range, 200)

    map_gen = MapGenerator(data_ex.dataset, map_range)
    map_gen.generate_map()
