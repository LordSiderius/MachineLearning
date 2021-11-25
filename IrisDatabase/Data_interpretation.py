# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv('data/iris.data', names=names)
# print(type(dataset))
#
# # dataset basic line information
# # dataset shape
# print(dataset.shape)
#
# # dataset view
# print(dataset.head(20))
#
# # basic statistical information of dataset
# print(dataset.describe())
#
# print(dataset.groupby('class').size())

#dataset visualisation
# Whisker box
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
#
# histograms of data
# dataset.hist()
# pyplot.show()

...
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()