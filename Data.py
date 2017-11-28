
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Read data
data = pd.read_csv("breast-cancer-wisconsin2.csv")

# 2. Preprocessing data
sum_of_miss_value_rows = (data[['Bare_Nuclei']]==0).sum()
print 'Shape of data before drop missing value rows : ', data.shape
print 'Sum of missing value rows : ', sum_of_miss_value_rows
# mark zero values as missing or NaN
data[['Bare_Nuclei']] = data[['Bare_Nuclei']].replace(0, np.NaN)
# count the number of NaN values in each column
print 'count the number of NaN values in each column', data.isnull().sum()
# print the first 30 rows of data too see NaN
print 'The first 30 rows of data', data.head(30)
# drop rows with missing values
data.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print 'Shape of data after drop missing value rows : ', data.shape


# 3. Change data into array
array = data.values
X = array[:,1:10]
Y = array[:,10]
# get the number of instance from dat
num_instances = len(X)

# 4. create model
nb  = GaussianNB()
svc = SVC()

# 5. K-Fold Cross Validation (K = 10)
kfold = KFold(n_splits=10, random_state=0)
nb_results = cross_val_score(nb, X, Y, cv=kfold)
nb_accuracy  = nb_results.mean()*100.0
print("Accuracy of Naive Bayes : %.3f%% " % (nb_accuracy))
svc_results = cross_val_score(svc, X, Y, cv=kfold)
svc_accuracy   = svc_results.mean()*100.0
print("Accuracy of Support Vector Machine: %.3f%% " % (svc_accuracy))

# 6. Confusion Matrix
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.90, test_size=0.10, random_state=0)

nb.fit(X_train, Y_train)
nb_predict = nb.predict(X_test)
nb_confusion_matrix = confusion_matrix(Y_test, nb_predict)
tp, fn, fp, tn = confusion_matrix(Y_test, nb_predict).ravel()
print 'Confusion matrix of data use Naive Bayes   : '
print nb_confusion_matrix
print 'Naive Bayes ( tp, fn, fp, tn ) = ', tp, fn, fp, tn

svc.fit(X_train, Y_train)
svc_predict = svc.predict(X_test)
svc_confusion_matrix = confusion_matrix(Y_test, svc_predict)
tp, fn, fp, tn = confusion_matrix(Y_test, svc_predict).ravel()
print 'Confusion matrix of data use Support Vector Machine: '
print svc_confusion_matrix
print 'Support Vector Machine ( tp, fn, fp, tn ) = ', tp, fn, fp, tn

# 7. Classification report of data
report_nb = classification_report(Y_test, nb_predict)
print 'Report of Naive Bayes : '
print report_nb

report_svc = classification_report(Y_test, svc_predict)
print 'Report of Suport Vector Machine : '
print report_svc