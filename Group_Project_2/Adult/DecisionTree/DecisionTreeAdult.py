import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from IPython.display import Image
import pydotplus


if __name__=="__main__":

        file = "adult.data"
        data = pd.read_csv(file)

        # Get the data
        age = data["age"].values
        workclass = data["workclass"].values
        fnlwgt = data["fnlwgt"].values
        education = data["education"].values
        educationNum = data["education-num"].values
        maritalStatus = data["marital-status"].values
        occupation = data["occupation"].values
        relationship = data["relationship"].values
        race = data["race"].values
        sex = data["sex"].values
        capitalGain = data["capital-gain"].values
        capitalLoss = data["capital-loss"].values
        hoursPerWeek = data["hours-per-week"].values
        nativeCountry = data["native-country"].values

        classification = data["classification"].values


        labelEncoder = preprocessing.LabelEncoder()

        encodedWorkclass = labelEncoder.fit_transform(workclass)
        encodedEducation = labelEncoder.fit_transform(education)
        encodedMaritalStatus = labelEncoder.fit_transform(maritalStatus)
        encodedOccupation = labelEncoder.fit_transform(occupation)
        encodedRelationship = labelEncoder.fit_transform(relationship)
        encodedRace = labelEncoder.fit_transform(race)
        encodedSex = labelEncoder.fit_transform(sex)
        encodedNativeCountry = labelEncoder.fit_transform(nativeCountry)

        encodedClassification = labelEncoder.fit_transform(classification)


        features = []
        for i in range(len(encodedWorkclass)):
                features.append([age[i], encodedWorkclass[i], fnlwgt[i],
                                encodedEducation[i], educationNum[i], encodedMaritalStatus[i],
                                encodedOccupation[i], encodedRelationship[i], encodedRace[i],
                                encodedSex[i], capitalGain[i], capitalLoss[i],
                                hoursPerWeek[i], encodedNativeCountry[i]])



        X_train, X_test, y_train, y_test = train_test_split(features, encodedClassification, random_state=0)
     

        classifier = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.00025)
        classifier = classifier.fit(X_train, y_train)
   
        prediction = classifier.predict(X_test)

        # Predict the Test set results
        prediction = classifier.predict(X_test)

        target = ['<=50k','>50k']

        print("=============================")
        print("Accuracy: ",(100*accuracy_score(y_test, prediction)),"%")
        print("=============================")
        print(classification_report(y_test, prediction,target_names=target))
        print("=============================")