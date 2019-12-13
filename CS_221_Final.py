#!/usr/bin/env python
# coding: utf-8

# Packages to import
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.layers import *
from keras.models import *
from keras.layers import Dense
from keras.regularizers import l1, l2
from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure
import pandas as pd
import operator
from math import log
from pandas import Series, DataFrame
from pylab import rcParams

import sklearn
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#Models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import sys

# Loads data and creates combined dataset, cleaning out maybes for target feature
def loadData(who):
    all_filenames = ['2014_clean1.csv', '2016_clean1.csv', '2017_clean1.csv', '2018_clean1.csv']
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    #export to csv
    combined_csv.to_csv("combined.csv", index=False, encoding='utf-8-sig')
    train=pd.read_csv('2018_clean1.csv')
    combined=pd.read_csv('combined.csv')
    colName = 'discuss_mental_'+who+'_Maybe'
    if colName in combined:
        combined = combined[combined[colName] == 0]
    else: print('malformed')
    return train, combined

# Builds & cleans dataset 
def buildData(dataset, who, train):
  #split dataset in features and target variable
  # feature_cols = ['country', 'state', 'self_employed', 'family_history', 'treatment', 'interfere', 'size', 'tech',	'benefits',	'options', 'wellness_program',	'resources', 'anon', 'leave', 'interview_mental_health', 'interview_physical_health', 'observe_negative']
  root = 'discuss_mental_'+who 
  target_col = root+'_Yes'
  ignoreAdd = [root+'_Maybe', root+'_No', target_col, root+'_Some of them']
  if who != 'supervisor':
        ignoreAdd.extend(['Unnamed: 0', root+'nan'])
  feature_cols = list(train.columns)
  remove_list = ['age', 'country', 'state', 'work_remotely','mental_health_seriously', 'employer_negative_consequence_mental', 'employer_negative_consequence_physical']
  remove_list.extend(ignoreAdd)
  for delete_info in remove_list:
    if delete_info in train.columns:
      feature_cols.remove(delete_info)
  X = dataset[feature_cols] # Features
  for feature in feature_cols:
    X.loc[X[feature] != 1, feature] = 0
  y = dataset[[target_col]] # Target variable
  # for feature in feature_cols:
  y.to_csv("y.csv", index=False, encoding='utf-8-sig')
  y.loc[y[target_col] != 1, target_col] = 0
  return X, y, feature_cols

# Performs logistic regression, print classification report, confusion matrix, accuracy, precision and recall. 
# Plots learning curves
def performLogReg(X,y, test_sz = 0.2, random_state = 42):
  # split X and y into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=random_state)

  # instantiate the model (using the default parameters)
  logreg = LogisticRegression()

  # fit the model with data
  logreg.fit(X_train,y_train)

  # predict
  y_pred=logreg.predict(X_test)

  print(classification_report(y_test,y_pred))

  cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
  print("Confusion matrix: \n" + str(cnf_matrix))
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  print("Precision:", metrics.precision_score(y_test, y_pred))
  print("Recall:", metrics.recall_score(y_test, y_pred))

  y_pred_proba = logreg.predict_proba(X_test)[::,1]
  fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
  auc = metrics.roc_auc_score(y_test, y_pred_proba)
  plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
  plt.legend(loc=4)
  plt.title(y.columns[0])
  plt.close()
  return logreg

# Print Logistic Regression Coefficients and saves figures
def printAndShowCoef(X, y, feature_cols, model):
  weights_dict = {}
  for i, feature in enumerate(feature_cols):
    weights_dict[feature] = model.coef_[0][i]
  weights_dict = sorted(weights_dict.items(), key=operator.itemgetter(1))
  
  top_10 = weights_dict[-10:]
  top_10.reverse()
  zip(*top_10)
  plt.figure(figsize=(15,5))
  plt.scatter(*zip(*top_10))
  plt.title(y.columns[0] + " top positive features")
  plt.savefig(y.columns[0] + ' top_pos6.png')
  plt.close()

  low_10 = weights_dict[:10]
  zip(*low_10)
  plt.figure(figsize=(15,5))
  plt.scatter(*zip(*low_10))
  plt.title(y.columns[0] + " top negative features")
  plt.savefig(y.columns[0] + ' neg_6.png')
  
  printListCoefs(X, model)
  return weights_dict

#Helper to print exact values of the extracted feature coefficients from logistic regression
def printListCoefs(X, logreg):
    features = list(X.columns)
    coefficients = {}
    for i, f in enumerate(features):
      coefficients[f] = logreg.coef_[0][i]
    import operator
    sorted_coefficients = sorted(coefficients.items(), key=operator.itemgetter(1))
    print(sorted_coefficients[::-1])
    print('Number of features: ', len(features))
    print('Number of entries: ',len(X))

# Tests different classifier models - 21 total
def runDifferentModels(X,Y, test_size = 0.3):
  # split X and y into training and testing sets
  X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=test_size)
  X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
  print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  #"Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"
  models = []
  models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
  models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('GP', GaussianProcessClassifier()))
  models.append(('RF', RandomForestClassifier()))
  models.append(('NN', MLPClassifier()))
  models.append(('AB', AdaBoostClassifier()))
  models.append(('QDA', QuadraticDiscriminantAnalysis()))
  models.append(('CART', DecisionTreeClassifier()))
  models.append(('NB', GaussianNB()))
  models.append(('K_M', KNeighborsClassifier(metric = 'minkowski')))
  models.append(('K_C', KNeighborsClassifier(metric = 'chebyshev')))
  models.append(('K_E', KNeighborsClassifier(metric = 'euclidean')))
  models.append(('K_Man', KNeighborsClassifier(metric = 'manhattan')))
  models.append(('K_Mat', KNeighborsClassifier(metric = 'matching')))
  models.append(('K_J', KNeighborsClassifier(metric = 'jaccard')))
  models.append(('K_D', KNeighborsClassifier(metric = 'dice')))
  models.append(('K_K', KNeighborsClassifier(metric = 'kulsinski')))
  models.append(('S_Lin', SVC(kernel='linear')))
  models.append(('S_RBF', SVC(kernel='rbf')))
  models.append(('S_Sig', SVC(kernel="sigmoid")))
  models.append(('S_Pol', SVC(kernel="poly")))
  

  # Evaluate each model through 10-fold cross-validation
  results = []
  names = []
  number = 1
  for name, model in models:
      kfold = KFold(n_splits=10, random_state=42)
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
      results.append(cv_results)
      names.append(name)
      msg = "%i) %s: Mean = %f with std = (%f)" % (number, name, cv_results.mean(), cv_results.std())
      number += 1
      print(msg)
  # Compare Algorithms
  matplotlib.rcParams.update({'font.size': 20})
  fig = plt.figure()
  fig = plt.figure(figsize=(14,10))
  plt.title('Algorithm Comparison')
  ax = fig
  medianprops = {'color': 'magenta', 'linewidth': 3}
  boxprops = {'color': 'black', 'linewidth': 2, 'linestyle': '-'}
  whiskerprops = {'color': 'black', 'linewidth': 2, 'linestyle': '-'}
  capprops = {'color': 'black', 'linewidth': 2, 'linestyle': '-'}
  flierprops = {'color': 'black', 'marker': 'x'}
  plt.boxplot(results,
           medianprops=medianprops,
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops,
           flierprops=flierprops)
#   ax.set_xticklabels(names)
  plt.xlabel('Model Number', fontsize=20)
  plt.ylabel('Accuracy', fontsize=20)
  plt.savefig('Algorithm Comparison')

# Tests solely the KNN and SVM models
def runKNN_SVMModels(X,y):
  # split X and y into training and testing sets
  X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
  X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
  print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
  #"Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"
  models = []
  models.append(('K_M', KNeighborsClassifier(metric = 'minkowski')))
  models.append(('K_C', KNeighborsClassifier(metric = 'chebyshev')))
  models.append(('K_E', KNeighborsClassifier(metric = 'euclidean')))
  models.append(('K_Man', KNeighborsClassifier(metric = 'manhattan')))
  models.append(('K_Mat', KNeighborsClassifier(metric = 'matching')))
  models.append(('K_J', KNeighborsClassifier(metric = 'jaccard')))
  models.append(('K_D', KNeighborsClassifier(metric = 'dice')))
  models.append(('K_K', KNeighborsClassifier(metric = 'kulsinski')))
  models.append(('S_Lin', SVC(kernel='linear')))
  models.append(('S_RBF', SVC(kernel='rbf')))
  models.append(('S_Sig', SVC(kernel="sigmoid")))
  models.append(('S_Pol', SVC(kernel="poly")))
  
  # Evaluate each model through 10-fold cross-validation
  results = []
  names = []
  for name, model in models:
      kfold = KFold(n_splits=10, random_state=42)
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
      results.append(cv_results)
      names.append(name)
      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
      print(msg)
  # Compare Algorithms
  fig = plt.figure()
  
  fig.suptitle('Algorithm Comparison')
  ax = fig.add_subplot(111)
  plt.boxplot(results)
  ax.set_xticklabels(names)
  plt.ylabel('Accuracy')
  plt.savefig('Comparison.png')

# See how logistic regression does after conducting 2 and 3 component PCA on data
def testPCA(X, Y):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    fig = plt.figure(figsize = (14,10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 25)
    ax.set_ylabel('Principal Component 2', fontsize = 25)
    ax.set_title('2 component PCA', fontsize = 30)
    targets = ['discuss_mental_supervisor_Yes']
    colors = ['r']
    for target, color in zip(targets,colors):
        cset = ax.scatter(principalDf['principal component 1']
                   , principalDf['principal component 2']
                   , 16)
    plt.savefig('PCA2.png')
    print(principalDf.columns)
    print(len(X.columns))
    print(Y)

    performLogReg(principalDf,Y)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    print(principalDf.columns)
    print(len(X.columns))
    print(Y)

    performLogReg(principalDf,Y)

    pca = PCA().fit(X)
    #Plotting the Cumulative Summation of the Explained Variance
    fig = plt.figure(figsize = (14,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), )
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Dataset Explained Variance')
    plt.savefig('PCAVariance.png')

    my_model = PCA(n_components=0.99, svd_solver='full')
    my_model.fit_transform(X)
    print(my_model.explained_variance_ratio_)
    print(len(my_model.explained_variance_ratio_))
    print(my_model.explained_variance_ratio_.cumsum())

# Runs Neural Network (SGD and Adam)
def runNeuralNet(X, Y, test_size = 0.3):
    # X_train (10 input features, 70% of full dataset), X_val (10 input features, 15% of full dataset), X_test (10 input features, 15% of full dataset)
    # Y_train (1 label, 70% of full dataset), Y_val (1 label, 15% of full dataset), Y_test (1 label, 15% of full dataset)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=test_size)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    # Hyperparameter turning
    # values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # for param in values:
    #     # model = Sequential([Dense(32, activation='relu', input_shape=(58,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'),]) # described sequentially, layer-by-layer with 32, 32, and 1 neurons; 58 input features
    #     print("Param = " + str(param))
        
    # SGD Optimizer
    param = 1e-2
    model = Sequential([Dense(32, activation='relu', input_shape=(58,), kernel_regularizer=l1(param)), Dense(32, activation='relu', kernel_regularizer=l1(param)), Dense(1, activation='sigmoid'),]) # described sequentially, layer-by-layer with 32, 32, and 1 neurons; 58 input features
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) 
    history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

    # Plot epoch vs. accuracy plot
    fig = plt.figure(figsize=(14,10))
    plt.plot(history.history['acc'], linewidth=3.5)
    plt.plot(history.history['val_acc'], linewidth=3.5)
    plt.plot(history.history['loss'], linewidth=3.5)
    plt.plot(history.history['val_loss'], linewidth=3.5)
    plt.rcParams.update({'font.size': 20})
    plt.title('SGD Model Training Loss and Accuracy')
    plt.ylabel('Accuracy/Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylim(0,4)
    plt.legend(['Acc_Train', 'Acc_Val', 'Loss_Train', 'Loss_Val'], loc='upper right')
    plt.savefig('SGD.png')
    print("SGD Accuracy: " + str(model.evaluate(X_test, Y_test)[1]))

    param = 0.5
    # Adam optimizer
    model_2 = Sequential([Dense(100, activation='relu', input_shape=(58,), kernel_regularizer=l1(param)), Dense(100, activation='relu'), Dense(100, activation='relu'), Dense(100, activation='relu'), Dense(1, activation='sigmoid'),])
    model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

    # Plot epoch vs. accuracy plot
    plt.figure()
    fig = plt.figure(figsize=(14,10))
    plt.plot(history2.history['acc'], linewidth=3.5)
    plt.plot(history2.history['val_acc'], linewidth=3.5)
    plt.plot(history2.history['loss'], linewidth=3.5)
    plt.plot(history2.history['val_loss'], linewidth=3.5)
    plt.rcParams.update({'font.size': 20})
    # plt.figure(figsize=(14,10))
    plt.title('Adam Model Training Loss and Accuracy')
    plt.ylabel('Accuracy/Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylim(0,4)
    plt.legend(['Acc_Train', 'Acc_Val', 'Loss_Train', 'Loss_Val'], loc='upper right')
    plt.savefig('Adam.png')
    print("Adam Accuracy: " + str(model_2.evaluate(X_test, Y_test)[1]))

    print("Success!")

# Creates model for param testing
def create_model(optimizer='sgd', init='glorot_uniform', loss = 'binary_crossentropy'):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(58,), init=init))
    model.add(Dense(32, activation='relu', init=init))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Hyperparameter tuning code
def hyperParamTuning(X,y, test_size):
    start=time.time()
    model = KerasClassifier(build_fn=create_model)
    optimizers = ['rmsprop', 'adam', 'sgd']
    init = ['glorot_uniform', 'normal', 'uniform']
    #'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
    loss = ['categorical_crossentropy', 'binary_crossentropy']
    # epochs = np.array([50, 100, 150])
    epochs = np.array([50])
    batches = np.array([5, 10, 20])
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    end = time.time()
    print("Time", end-start)

 #     history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

    #     # Plot epoch vs. accuracy plot
    #     plt.plot(history.history['acc'])
    #     plt.plot(history.history['val_acc'])
    #     plt.title('Model accuracy')
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Val'], loc='upper left')
    #     plt.show()

    #     # Plot epoch vs. loss plot
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('Model loss')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Val'], loc='upper left')
    #     plt.show()

    #     print("Accuracy: " + str(model.evaluate(X_test, Y_test)[1]))
    # print(grid_result.grid_scores_)
    results = pd.DataFrame.from_dict(grid_result.cv_results_)
    print(results)
    results.to_csv('results.csv', index=False)
    # print(grid_result.cv_results_)

    # for params, mean_score, scores in grid_result.cv_results_:
    #     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    # print("total time:",time()-start)

# Main function for running experiments
def runExperiment(who):
    train, combined = loadData(who)
    #weights_supervisor_yes
    X,y,feature_cols = buildData(combined, who, train)
    logreg = performLogReg(X,y)
    weights = printAndShowCoef(X,y,feature_cols, logreg)
    runDifferentModels(X,y)
    #     runKNN_SVMModels(X,Y)
    testPCA(X, y)
    runNeuralNet(X, y)
    print('Finished!')

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 10})
#     who = sys.argv[1]
    who = 'supervisor'
    if who != 'supervisor' and who != 'coworkers':
        print('Incorrect Inputs')
    runExperiment(who)
