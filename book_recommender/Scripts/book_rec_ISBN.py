# import
import pandas as pd
import numpy as np


class Book_database(object):
    def __init__(self, books_path='data/BX-Books.csv', ratings_path='data/BX-Book-Ratings.csv'):
        # load books
        self.books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip', dtype=str)

        # load ratings
        ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';',
                              dtype={"User-ID": int, "ISBN": str,
                                     "Book-Rating": int})

        # discard ratings with zero rating of book
        self.ratings = ratings[ratings['Book-Rating'] != 0]

        # self.dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    def get_book_ISBN(self, book_title_input='  The Fellowship of the ring (the lord of the rings, part 1)',
                  book_author='tolkien'):

        book_title_input = book_title_input.strip()
        book_author = book_author.strip()

        books = self.books['Book-Title'].str.lower() == book_title_input.lower() &\
                self.books['Book-Author'].str.lower().str.contains(book_author.lower())

        print(books[books])

        self.book_ISBN = self.books['ISBN'][
            (self.books['Book-Title'].str.lower() == book_title_input.lower()) &
            (self.books['Book-Author'].str.lower().str.contains(book_author.lower()))]

        return self.book_ISBN

    def recommend(self):

        book_readers = self.ratings['User-ID'][self.ratings['ISBN'].isin(self.book_ISBN)].reset_index()
        print('before', book_readers.size)
        book_readers = book_readers.drop_duplicates(subset='User-ID')
        print('after', book_readers.size)


        # all_books_read_by =

        # book_readers = np.unique(book_readers.tolist())
        #
        # # final dataset
        #
        # books_of_given_readers = self.dataset[(self.dataset['User-ID'].isin(book_readers))]
        #
        # # Number of ratings per other books in dataset
        # number_of_rating_per_book = books_of_given_readers.groupby('Book-Title').agg('count').reset_index()
        #
        # # select only books which have actually higher number of ratings than threshold
        # threshold = 8
        # books_to_compare = number_of_rating_per_book['Book-Title'][
        #     number_of_rating_per_book['User-ID'] >= threshold]  # !!! 8 as magic number
        #
        # books_to_compare = books_to_compare.tolist()
        #
        # ratings_data_raw = books_of_given_readers[['User-ID', 'Book-Rating', 'Book-Title']][
        #     books_of_given_readers['Book-Title'].isin(books_to_compare)]
        #
        # # group by User and Book and compute mean
        #
        # ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()
        #
        # # reset index to see User-ID in every row
        # ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()
        #
        # dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
        #
        # # Take out the Lord of the Rings selected book from correlation dataframe
        # dataset_of_other_books = dataset_for_corr.copy(deep=False)
        # dataset_of_other_books.drop([book_title_input], axis=1, inplace=True)
        #
        # # empty lists
        # # print(list(dataset_of_other_books.columns.values))
        #
        # # print(dataset_of_other_books.head(5))
        #
        # book_titles = list(dataset_of_other_books.keys())
        #
        # # print(dataset_of_other_books.head())
        #
        # correlations = []
        # avg_rating = []
        #
        # # corr computation
        # book_input_ratings = dataset_for_corr[book_title_input]
        #
        # # tmp = ratings_data_raw[ratings_data_raw['Book-Title'].isin(book_titles)]
        # # aaa = tmp[['Book-Title','Book-Rating']].groupby(['Book-Title']).mean()
        # # print(aaa.head())
        #
        # for book_title in book_titles:
        #     correlations.append(book_input_ratings.corr(dataset_of_other_books[book_title]))
        #
        #     tab = (ratings_data_raw[ratings_data_raw['Book-Title'] == book_title].groupby(
        #         ratings_data_raw['Book-Title']).mean())
        #
        #     avg_rating.append(tab['Book-Rating'].min())
        #
        # # creating structure of data for output, sorting it from by greatest match and selecting first 10 books
        # book_corr = list(zip(book_titles, correlations, avg_rating))
        # book_corr.sort(key=lambda x: x[1], reverse=True)
        # book_corr = book_corr[0:9]
        #
        # return book_corr


if __name__ == '__main__':
    # loading the database
    book_database = Book_database()
    book_database.get_book_ISBN()
    book_database.recommend()


    # book_title_input = 'the fellowship of the ring (the lord of the rings, part 1)'
    # book_author = 'tolkien'
    #
    # print(book_database.recommend(book_title_input, book_author))

