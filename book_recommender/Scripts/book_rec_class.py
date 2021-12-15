# import
import pandas as pd
import numpy as np
import time


class Book_database(object):
    def __init__(self, books_path='data/BX-Books.csv', ratings_path='data/BX-Book-Ratings.csv'):

        # load books
        books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip', dtype=str)

        # load ratings
        ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';',
                              dtype={"User-ID": int, "ISBN": str,
                                     "Book-Rating": int})

        ratings = ratings[ratings['Book-Rating'] != 0]

        # merge of the datasets based on ISNB code
        dataset = pd.merge(ratings, books, on=['ISBN'])

        # all data in database are set lowercase to make strings comparable
        self.dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    def recommend(self, book_title_input='the fellowship of the ring (the lord of the rings, part 1)',
                  book_author='tolkien', threshold=8):


        # string of book title and book author are stripped and make lowercase to make them comparable
        book_title_input = book_title_input.strip().lower()
        book_author = book_author.strip().lower()

        book_readers = self.dataset['User-ID'][
            (self.dataset['Book-Title'] == book_title_input) &
            (self.dataset['Book-Author'].str.contains(book_author))]

        # removal of duplicities in books readers
        book_readers = np.unique(book_readers.tolist())

        # set of books read by identified readers
        books_of_given_readers = self.dataset[(self.dataset['User-ID'].isin(book_readers))]

        # number of ratings per other books in dataset
        number_of_rating_per_book = books_of_given_readers.groupby('Book-Title').agg('count').reset_index()


        # select only books which have actually higher number of ratings than threshold
        books_to_compare = number_of_rating_per_book['Book-Title'][
            number_of_rating_per_book['User-ID'] >= threshold]  # !!! 8 as magic number

        books_to_compare = books_to_compare.tolist()

        ratings_data_raw = books_of_given_readers[['User-ID', 'Book-Rating', 'Book-Title']][
            books_of_given_readers['Book-Title'].isin(books_to_compare)]

        # group by User and Book and compute mean
        ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

        # reset index to see User-ID in every row
        ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

        # reshape the data and make Book-Title as x-axe
        dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

        # take out given book from selected books from correlation dataframe
        dataset_of_other_books = dataset_for_corr.copy(deep=False)
        dataset_of_other_books.drop([book_title_input], axis=1, inplace=True)

        # average rating calcualtion
        book_titles = list(dataset_of_other_books.keys())

        books_avarage_rating = ratings_data_raw.groupby(['Book-Title'])['Book-Rating'].mean()
        avg_ratings_list = list(books_avarage_rating[book_titles])

        # correlation calculation
        book_input_ratings = dataset_for_corr[book_title_input]

        correlations = []

        for book_title in book_titles:
            # correlation of each book from other books list with given book
            correlations.append(book_input_ratings.corr(dataset_of_other_books[book_title]))

        # creating structure of data for output, sorting it from by greatest match and selecting first 10 books
        book_corr = list(zip(book_titles, correlations, avg_ratings_list))
        book_corr.sort(key=lambda x: x[1], reverse=True)

        # remove books with correlation coef below 0.3
        threshold_match = 0.3
        book_corr = [book_corr for book_corr in book_corr if book_corr[1] > threshold_match]

        # returns first 10 elements or books with corr coef above threshold
        book_corr = book_corr[0:min(9, len(book_corr))]

        return book_corr


if __name__ == '__main__':
    # loading the database
    book_database = Book_database()

    book_title_input = 'The Fellowship of the ring (the lord of the rings, part 1)'
    book_author = 'tolkien'
    threshold = 8

    print(book_database.recommend(book_title_input, book_author, threshold))
