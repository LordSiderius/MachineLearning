# import
import pandas as pd
import numpy as np
import time

tic = time.perf_counter()

class Book_database(object):
    def __init__(self, books_path='data/BX-Books.csv', ratings_path='data/BX-Book-Ratings.csv'):

        # load books
        books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip', dtype=str)

        # load ratings
        ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';',
                              dtype={"User-ID": int, "ISBN": str,
                                     "Book-Rating": int})

        ratings = ratings[ratings['Book-Rating'] != 0]

        # merge the datasets
        dataset = pd.merge(ratings, books, on=['ISBN'])

        self.dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    def recommend(self, book_title_input='the fellowship of the ring (the lord of the rings, part 1)', book_author='tolkien'):

        book_readers = self.dataset['User-ID'][
            (self.dataset['Book-Title'] == book_title_input) & (self.dataset['Book-Author'].str.contains(book_author))]

        book_readers = np.unique(book_readers.tolist())

        # final dataset

        books_of_given_readers = self.dataset[(self.dataset['User-ID'].isin(book_readers))]


        # Number of ratings per other books in dataset
        number_of_rating_per_book = books_of_given_readers.groupby('Book-Title').agg('count').reset_index()

        print(number_of_rating_per_book.keys())
        print(number_of_rating_per_book['Book-Title'])
        print(number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID']])
        input()

        # select only books which have actually higher number of ratings than threshold
        threshold = 8
        books_to_compare = number_of_rating_per_book['Book-Title'][
            number_of_rating_per_book['User-ID'] >= threshold]  # !!! 8 as magic number


        books_to_compare = books_to_compare.tolist()

        ratings_data_raw = books_of_given_readers[['User-ID', 'Book-Rating', 'Book-Title']][
            books_of_given_readers['Book-Title'].isin(books_to_compare)]


        # group by User and Book and compute mean

        ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()


        # reset index to see User-ID in every row
        ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

        dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')


        # Take out the Lord of the Rings selected book from correlation dataframe
        dataset_of_other_books = dataset_for_corr.copy(deep=False)
        dataset_of_other_books.drop([book_title_input], axis=1, inplace=True)

        # empty lists
        # print(list(dataset_of_other_books.columns.values))

        # print(dataset_of_other_books.head(5))


        book_titles = list(dataset_of_other_books.keys())
        correlations = []
        avg_rating = []
        # corr computation

        for book_title in book_titles:

            correlations.append(dataset_for_corr[book_title_input].corr(dataset_of_other_books[book_title]))


            tab = (ratings_data_raw[ratings_data_raw['Book-Title'] == book_title].groupby(
                ratings_data_raw['Book-Title']).mean())

            avg_rating.append(tab['Book-Rating'].min())

        # final dataframe of all correlation of each book
        corr_fellowship = pd.DataFrame(list(zip(book_titles, correlations, avg_rating)),
                                       columns=['book', 'corr', 'avg_rating'])

        print("Correlation for book:", book_title_input)
        # top 10 books with highest corr
        print(corr_fellowship.sort_values('corr', ascending=False).head(10))

        pass


if __name__ == '__main__':
    book_database = Book_database()
    book_title_input = 'the fellowship of the ring (the lord of the rings, part 1)'
    book_author = 'tolkien'
    book_database.recommend(book_title_input, book_author)
