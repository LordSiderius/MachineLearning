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
from data_extractor import Data_extractor
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


if __name__ == '__main__':

    # map_range = (13.978658, 14.879769, 49.896474, 50.260075)
    map_range = (14.42, 14.45, 50.07, 50.1)

    data_extractor = Data_extractor()
    dataset = copy.copy(data_extractor.dataset)

    print(dataset.head())

    # map rastering
    n = 50000/200 # width of map - 50km split by 200 meters
    m = 75000/200 # height of map - 75km split by 200 meters




    # dataset['UTC'].groupby([dataset['month'], dataset['day']]).count().plot(kind="bar")

    # dataset.pivot_table(index='day', columns='month', aggfunc=sum).plot(kind="bar")
    # plt.show()
    #
    # # scatter_matrix(dataset)
    # # plt.show()
    #
    # print(dataset.head())
    #
    # print(y)
    #
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X,y, test_size=0.20, random_state=1)
    #
    #  First the noiseless case

    dataset = dataset[dataset['latitude'] > map_range[2]]
    dataset = dataset[dataset['latitude'] < map_range[3]]
    dataset = dataset[dataset['longitude'] > map_range[0]]
    dataset = dataset[dataset['longitude'] < map_range[1]]

    monday_data = dataset[dataset['weekday'] == 0]
    monday_data = dataset[dataset['timeofday'] > 0.75]
    monday_data = dataset[dataset['month'] > 10]
    monday_data_2 = monday_data[['weekday', 'day', 'month']].copy().drop_duplicates()


    sample_count = round(monday_data['weekday'].count())
    monday_count = monday_data_2['weekday'].count()
    avg_controls_cnt = round(sample_count / monday_count)
    print('average controls count:', avg_controls_cnt)

    X = np.atleast_2d(monday_data['timeofday'].values).T
    print('length of X:', len(X))

    # Observations
    y = monday_data['latitude'].values
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(0, 1, avg_controls_cnt)).T
    xx = xx.astype(np.float32)

    alpha = 0.95
    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                    n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=3,
                                    min_samples_split=3)

    # clf = KNeighborsRegressor(n_neighbors=5)

    clf.fit(X, y)


    # Make the prediction on the meshed x-axis
    y_upper = clf.predict(xx)
    clf.set_params(alpha=1.0 - alpha)
    clf.fit(X, y)
    # Make the prediction on the meshed x-axis
    y_lower = clf.predict(xx)
    #
    clf.set_params(loss='ls')
    clf.fit(X, y)


    # Make the prediction on the meshed x-axis

    y_pred1 = clf.predict(xx)

    # # Plot the function, the prediction and the 90% confidence interval based on
    # # the MSE
    # fig = plt.figure()
    # plt.figure(figsize=(20,10))
    # plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
    # plt.plot(xx, y_pred, 'r-', label=u'Prediction')
    # plt.plot(xx, y_upper, 'k-')
    # plt.plot(xx, y_lower, 'k-')
    # plt.fill(np.concatenate([xx, xx[::-1]]),
    #          np.concatenate([y_upper, y_lower[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='90% prediction interval')
    # plt.xlabel('$Time of Day by Fraction$')
    # plt.ylabel('$Latitude$')
    # plt.ylim(lat_min, lat_max)
    # plt.legend(loc='upper right')
    # plt.show()
    # ypred1 = y_pred

    # Observations
    y = monday_data['longitude'].values.T
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(0, 1, avg_controls_cnt)).T
    xx = xx.astype(np.float32)

    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                    n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=3,
                                    min_samples_split=3)

    # clf = KNeighborsRegressor(n_neighbors=5)

    clf.fit(X, y)

    # Make the prediction on the meshed x-axis
    y_upper = clf.predict(xx)
    clf.set_params(alpha=1.0 - alpha)
    clf.fit(X, y)
    # Make the prediction on the meshed x-axis
    y_lower = clf.predict(xx)
    #
    clf.set_params(loss='ls')
    clf.fit(X, y)

    # Make the prediction on the meshed x-axis

    y_pred2 = clf.predict(xx)
    # print('y_pred2', y_pred2)
    # print('y_pred1', y_pred1)


    # Map points of events
    pred_map = folium.Map([(map_range[2]+map_range[3])/2, (map_range[0]+map_range[1])/2], zoom_start=14)
    dfmatrix = monday_data[['latitude', 'longitude']].values
    # plot heatmap
    pred_map.add_child(plugins.HeatMap(dfmatrix, radius=15))
    for i in range(avg_controls_cnt):

        folium.CircleMarker((y_pred1[i], y_pred2[i]),
                            radius=5,
                            popup=str(i),
                            color='red',
                            fill_color="#dc143c",

                            ).add_to(pred_map)

    # convert to (n, 2) nd-array format for heatmap
    # matrix = np.column_stack((y_pred1, y_pred2))
    # plot heatmap
    # m.add_child(plugins.HeatMap(matrix, radius=15))

    pred_map.save('index_predict_monday.html')

