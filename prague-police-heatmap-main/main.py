from data_extractor import DataExtractor
from grid_map import MapGenerator

if __name__ == '__main__':
    map_range = (50.07, 50.1, 14.42, 14.45)


    data_ex = DataExtractor(200)
    map_gen = MapGenerator(data_ex.dataset, map_range)
    map_gen.generate_map()
