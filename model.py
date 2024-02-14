# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
# Importing the dataset
dataset = pd.read_csv('diabetes3.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
y = dataset.iloc[:, 8].values
start=time()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
#helps in improving the performance and convergence of many machine learning algorithms
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# using scikit-learn to create a RandomForestClassifier and fit it to your training data 
from sklearn.ensemble import RandomForestClassifier
#n_estimators: The number of trees in the forest. In this case, you've set it to 10.
#criterion: The function to measure the quality of a split. 'entropy' is used, 
# This parameter ensures reproducibility. If you set random_state to 0, you'll get the same results every time you run the code.
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#the model is trained on the provided training data (X_train features and y_train labels).
classifier.fit(X_train, y_train)

#  Use the trained classifier to make predictions on the test data
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix ,using scikit-learn to compute the confusion matrix for evaluating the performance of your model
from sklearn.metrics import confusion_matrix
#Compute the confusion matrix by comparing the true labels (y_test) with the predicted labels (y_pred)
cm = confusion_matrix(y_test, y_pred)

#You can use this confusion matrix to evaluate various performance metrics such as accuracy, precision, recall, and F1-score.
from sklearn.metrics import confusion_matrix,mean_squared_error
print("the mean square error",mean_squared_error(y_test, y_pred))
#calculate True Positive (TP),False Positive (FP), False Negative (FN), True Negative (TN)
TP=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TN=cm[1][1]

# Calculate accuracy, precision, recall, and F-measure
acc=(TP+TN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
precision=TP/(TP+FP)
F_measure=2*precision*recall/(precision+recall)

#precision,recall,threshold=precision_recall_curve(y_test, y_pred)
print("the accuracy is:",acc*100,"%")
print("the precision is:",precision*100,"%")
print("the recall is:",recall*100,"%")
print("the F-measure:",F_measure*100,"%")

# Record and print the execution time
end=time()
print("the time is "+str(end-start))
print(cm)


#using the pickle module to save your trained classifier to a file (model.pkl) and then load it back for making predictions. 
import pickle 
# This line saves your trained RandomForestClassifier (classifier) to a file named "model.pkl" using the pickle.dump() function.
pickle.dump(classifier,open("model.pkl","wb"))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 138, 62,53,120,33.6,.12,47]]))

