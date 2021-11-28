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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np


names = ['age', 'workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
         'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

dataset = read_csv('data/adult.data', names=names, delimiter=',', encoding="utf-8")
# print(dataset['workclass'])
# print('dataset:')
# print(dataset.head(20))


mapping = {}
mapping['education'] = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
           '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?']

mapping['workclass'] = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                        'Without-pay', 'Never-worked', '?']

mapping['marital-status'] = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                             'Married-spouse-absent', 'Married-AF-spouse', '?']

mapping['occupation'] = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial," \
                        " Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical," \
                        " Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv," \
                        " Armed-Forces, ?".split(", ")

mapping['relationship'] = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried, ?".split(', ')

mapping['race'] = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black, ?".split(', ')

mapping['sex'] = "Female, Male, ?".split(', ')

mapping['native-country'] = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc)," \
                            " India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland," \
                            " Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador," \
                            " Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia," \
                            " El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands, ?".split(', ')

mapping['salary'] = ">50K, <=50K, ?".split(', ')

for key in mapping:

    le = preprocessing.LabelEncoder()
    le.fit(mapping[key])
    dataset[key] = le.transform(dataset[key].str.strip())
    # print(dataset[key].head(5))


# print(dataset.head())
# print(dataset.shape)
# print(dataset.describe())
#
# scatter_matrix(dataset)
# pyplot.show()
#
array = dataset.values

X = array[:, 0:14]
y = array[:, 14]

#
X_train, X_validation, Y_train, Y_validation = train_test_split(X,y, test_size=0.10, random_state=1)

# kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# cv_results = cross_val_score( GaussianNB(), X_train, Y_train, cv=kfold, scoring='accuracy')
#
# print(cv_results.mean())

#
#
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# models.append(('MLPA', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(250,), random_state=1)))
# models.append(('MLPS', MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(250,), random_state=1)))
# evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(predictions)

# Score evaluation
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# input('go to MLP')
#
# model = MLPClassifier(solver='sgd', alpha=1e-6, hidden_layer_sizes=(250,), random_state=1)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)
# print(predictions)

# Score evaluation
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))