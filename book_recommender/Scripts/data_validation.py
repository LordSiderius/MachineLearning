import pandas as pd

import re

class Book_database(object):
    def __init__(self, books_path='data/BX-Books.csv', ratings_path='data/BX-Book-Ratings.csv'):

        # load books
        self.books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip', dtype=str)

        # load ratings
        ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';',
                              dtype={"User-ID": int, "ISBN": str,
                                     "Book-Rating": int})

        self.ratings = ratings[ratings['Book-Rating'] != 0]

        # merge of the datasets based on ISNB code
        dataset = pd.merge(ratings, self.books, on=['ISBN'])

        # all data in database are set lowercase to make strings comparable
        self.dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    def data_validation(self):

        rat_count_per_user = self.ratings.groupby(['User-ID']).agg('count').reset_index()
        aaa = rat_count_per_user['User-ID'][rat_count_per_user['Book-Rating'] > 2600]

        print(aaa.reset_index()['User-ID'])

        self.ratings = self.ratings[~self.ratings['User-ID'].isin(aaa)]

        print(self.ratings['ISBN'].str.len())

        self.ratings = self.ratings[self.ratings['ISBN'].str.len() == 10]

        print(re.match('^[X][0-9]+$', '056840'))

        self.ratings = self.ratings[bool(re.match('^[0-9]+$', self.ratings['ISBN']))]



        print(self.ratings['ISBN'].str.len())

        pass

if __name__ == '__main__':
    # loading the database
    book_database = Book_database()
    book_database.data_validation()