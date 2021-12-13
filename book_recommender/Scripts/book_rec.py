# import
import pandas as pd
import numpy as np

# load ratings
ratings = pd.read_csv('data/BX-Book-Ratings.csv', encoding='cp1251', sep=';')
ratings = ratings[ratings['Book-Rating'] != 0]



#
# load books
names = []
books = pd.read_csv('data/BX-Books.csv', encoding='cp1251', sep=';', error_bad_lines=False)
# !!! add datatype of column
#
#



dataset = pd.merge(ratings, books, on=['ISBN'])

print(dataset.head(20))
#  !!! Merge ignoring multiple rating instances, cannot be merged in this hard way
# maybe put dictionary or list with user ID and rating as pandas atribute


print(dataset["Book-Author"].head(5))
dataset = dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
print(dataset["Book-Author"].head(5))
# !!! Should make lower case, there is function for in in pandas Series.str.lower
#  dataset variable can be recycled, cause is not used anymore further in project



tolkien_readers = dataset['User-ID'][(dataset['Book-Title'] ==
                                                'the fellowship of the ring (the lord of the rings, part 1)') &
                                               (dataset['Book-Author'].str.contains("tolkien"))]

# !!! Book title and author input should be user input based to make script more variable, not single purpose
# !!! It is better to use generic name for variable

tolkien_readers = tolkien_readers.tolist()
tolkien_readers = np.unique(tolkien_readers)

# final dataset
books_of_tolkien_readers = dataset[(dataset['User-ID'].isin(tolkien_readers))]

# number of ratings per other books in dataset
number_of_rating_per_book = books_of_tolkien_readers.groupby(['Book-Title']).agg('count').reset_index()

# select only books which have actually higher number of ratings than threshold
books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= 8]
books_to_compare = books_to_compare.tolist()

ratings_data_raw = books_of_tolkien_readers[['User-ID', 'Book-Rating', 'Book-Title']][books_of_tolkien_readers['Book-Title'].isin(books_to_compare)]

# group by User and Book and compute mean
ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

# reset index to see User-ID in every row
ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

LoR_list = ['the fellowship of the ring (the lord of the rings, part 1)']

result_list = []
worst_list = []

# for each of the trilogy book compute:
for LoR_book in LoR_list:

    # take out the Lord of the Rings selected book from correlation dataframe
    dataset_of_other_books = dataset_for_corr.copy(deep=False)
    dataset_of_other_books.drop([LoR_book], axis=1, inplace=True)

    # empty lists
    book_titles = []
    correlations = []
    avg_rating = []

    # corr computation
    for book_title in list(dataset_of_other_books.columns.values):
        book_titles.append(book_title)
        correlations.append(dataset_for_corr[LoR_book].corr(dataset_of_other_books[book_title]))
        tab=(ratings_data_raw[ratings_data_raw['Book-Title'] == book_title].groupby(ratings_data_raw['Book-Title']).mean())
        avg_rating.append(tab['Book-Rating'].min())
    # final dataframe of all correlation of each book
    corr_fellowship = pd.DataFrame(list(zip(book_titles, correlations, avg_rating)), columns=['book', 'corr', 'avg_rating'])
    print(corr_fellowship.head()) #why is this here without print? remove for production

    # top 10 books with highest corr
    result_list.append(corr_fellowship.sort_values('corr', ascending=False).head(10))

    #worst 10 books
    worst_list.append(corr_fellowship.sort_values('corr', ascending=False).tail(10))

print("Correlation for book:", LoR_list[0])
# print("Average rating of LOR:", ratings_data_raw[ratings_data_raw['Book-Title']=='the fellowship of the ring (the lord of the rings, part 1'].groupby(ratings_data_raw['Book-Title']).mean()))
rslt = result_list[0]
# !!! It is not printing output at all, make it function return or at least print it to console
print(rslt)