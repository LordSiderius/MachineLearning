import pandas as pd

train_data = pd.read_csv('data/TrainData.csv', sep=";", names=["Book-title", "Rating"], skiprows=1)
my_data = pd.read_csv('data/My_similar_books.csv', sep=";", names=["Book-title", "Rating"], skiprows=1)



print(my_data.corr())