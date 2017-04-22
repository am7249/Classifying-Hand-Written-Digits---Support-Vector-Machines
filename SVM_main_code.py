# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:36:31 2017

@author: Abdullah Mobeen
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import metrics, svm, cross_validation
import time

def normalize(data):
    """Takes as input the name of the file storing data. Returns the feature vectors
    and the vector of labels"""
    
    features = []
    labels = []
    with open(data,'r') as file:
        all_features = file.readlines()
        for i in range(len(all_features)):
            features.append((all_features[i].split(',')[1:]))
            labels.append((all_features[i].split(',')[0]))
        for i in range(len(labels)):
            labels[i] = int(labels[i])
            for j in range(len(features[0])):
                features[i][j] = (((2*(int(features[i][j])))/255)-1)
        features = sp.array(features)
        labels = sp.array(labels)
    return features,labels
    
def classifier(train, test,C,gamma):
    """Uses Support Vector Machine to obtain the optimal hyperplane for Multiple Classes.
    Then tests the optimal hyperplane for new data in mnist_test.txt. 
    Prints the results: errors committed, total images analyzed, classifier, and confusion matrix"""
    
    classifier = svm.SVC(C=C,gamma=gamma)
    features_train,labels_train = normalize(train)
    features_test,labels_test = normalize(test)
    classifier.fit(features_train,labels_train)
    predicted = classifier.predict(features_test)
    error = 0
    total = 0
    for i in range(len(predicted)):
        if int(predicted[i]) != int(labels_test[i]):
            error+=1
        total+=1
    print("Total images analysed:", len(labels_test),'\n',"Errors committed:",error)
    print()
    print("Error Percentage:",(error/len(predicted))*100,'%')
    print()
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(labels_test, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_test, predicted))


def cross_val(C,gamma):
    """Takes as input the value of C and gamma. Performs cross validation
    with 5 folds and returns the mean scores and mean test-errors"""
    
    global features,labels
    classifier = svm.SVC(C=C,gamma=gamma)
    scores = cross_validation.cross_val_score(classifier,features,labels,cv=5)
    scores_mean = np.mean(scores)
    test_error = ((1-scores_mean)*100)
    print("Using the Value of C =",C,"We get: Cross Val Error =",scores_mean,"and Test Error % =", test_error)
    print()
    print("Now computing cross validation results for the next value of C. Please wait.")
    print()
    return test_error,scores_mean
    
if __name__ == '__main__':
    
    features_1, labels_1 = normalize('mnist_train.txt')
    features_2, labels_2 = normalize('mnist_test.txt')
    features = np.concatenate((features_1,features_2))
    labels = np.concatenate((labels_1,labels_2))
    
    classifier('mnist_train.txt','mnist_test.txt',1,(1/len(features_1[0]))) #Also prints the confusion matrix. Concept understood from the sckit website example. 
    print()
    
    print("Now doing the cross validation and plotting the graph to find the optimal value of C, it might take some time")
    print()    
    print("Doing various tests on classifier function for the value of gamma, its optimal value has been found to be 1/200")    
    start_time = time.time()    
    print()   
    
    test_errors = list()
    scores = list()
    C = [0.1,0.5,1,2,3,5,10,30] #Multiple arbitrary values for C
    for c in C:
        errors,sc = cross_val(c,1/200) #Experimenting with classification function gave the optimal gamma = 1/200.
        test_errors.append(errors)
        scores.append(sc)
    print("Mean cross validation scores using the respective values of C =",C,":",scores)
    print()
    print("Mean test errors (%) using the respective values of C =",C,":",test_errors)
    
    plt.plot(C,test_errors,'-b',label = 'Mean Test Errors against values of C')
    plt.legend(loc = 'upper right')
    end_time = time.time()
    print()
    print("Total time taken:",round(end_time-start_time,2),'seconds')
    print()
    
    print("From the graph, it has been found that C = 3 and gamma = 1/200 gives the least error")