#!/usr/bin/env python

"""
__author__ = "Guillermo Torres, Debora Gil and Pau Cano"
__license__ = "GPLv3"
__email__ = "gtorres,debora,pau@cvc.uab.cat"
__year__ = "2023"
"""

import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn import svm


def make_positive_negative_lists(filename):
    # Make two list from the filename, which is an .xlsx file.
    # And return two list with positive and negative samples each one.
    # Positive = 1 and Negative = 0.

    df = pd.read_excel(filename, 
                       sheet_name='Sheet1', 
                       engine='openpyxl'
        )
    print("Reading the filename: {}".format(filename))

    # Positive (malign) = 1, Negative (benign) = 0.
    positive_samples = df.loc[df['diagnosis'] == 1]
    negative_samples = df.loc[df['diagnosis'] == 0]

    print("Positive samples: {}".format(len(positive_samples)))
    print("Negative samples: {}".format(len(negative_samples)))

    return positive_samples, negative_samples




############ MAIN PROGRAM ########################

print("Begining ...")

################### Parameters to be configured ###################################
path = '/home/willytell/Dropbox/code/ML4PM/Code'
excel_features = os.path.join(path, 'features.xlsx')
proportion = 0.5  # define the training samples proportion



all_positive_samples, all_negative_samples = make_positive_negative_lists(excel_features)

all_positive_samples = shuffle(all_positive_samples)
all_negative_samples = shuffle(all_negative_samples)

# Select manually the features
selection = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy',
             'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
             'original_firstorder_Maximum',
             'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy',
             'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
             'original_firstorder_Maximum'
             ]


# applying the feature selection (it removes any column that is not included in the 'selection' list)
positive_samples = all_positive_samples.loc[:, selection]
negative_samples = all_negative_samples.loc[:, selection]

y_positive_samples = all_positive_samples.loc[:, 'diagnosis']
y_negative_samples = all_negative_samples.loc[:, 'diagnosis']

###################### RANDOM SELECTION OF THE TRAIN AND TEST SET  ##########################################
#### It is necessary to select a proportion of samples from positive and negative cases.

# Random selection of the train and test set from POSITIVE samples
num_positive_samples = len(positive_samples)
num_training_positive_samples = int(num_positive_samples * proportion)
all_positive_indexes = range(0, num_positive_samples)

training_positive_indexes = random.sample(all_positive_indexes, int(num_positive_samples * proportion))
testing_positive_indexes = [x for x in all_positive_indexes if x not in training_positive_indexes]

training_positive_samples = positive_samples.iloc[training_positive_indexes, :]
y_training_positive_samples = y_positive_samples.iloc[training_positive_indexes]

testing_positive_samples = positive_samples.iloc[testing_positive_indexes, :]
y_testing_positive_samples = y_positive_samples.iloc[testing_positive_indexes]

print("Training positive samples: {}.".format(len(training_positive_samples)))
print("Testing positive samples: {}.".format(len(testing_positive_samples)))


# Random selection of the train and test set from NEGATIVE samples
num_negative_samples = len(negative_samples)
num_training_negative_samples = int(num_negative_samples * proportion)
all_negative_indexes = range(0, num_negative_samples)

training_negative_indexes = random.sample(all_negative_indexes, int(num_negative_samples * proportion))
testing_negative_indexes = [x for x in all_negative_indexes if x not in training_negative_indexes]

training_negative_samples = negative_samples.iloc[training_negative_indexes, :]
y_training_negative_samples = y_negative_samples.iloc[training_negative_indexes]

testing_negative_samples = negative_samples.iloc[testing_negative_indexes, :]
y_testing_negative_samples = y_negative_samples.iloc[testing_negative_indexes]

print("Training negative samples: {}.".format(len(training_negative_samples)))
print("Testing negative samples: {}.".format(len(testing_negative_samples)))

# train and test sets
X_training_samples = training_positive_samples.append(training_negative_samples)
y_training_samples = y_training_positive_samples.append(y_training_negative_samples)

X_test_samples = testing_positive_samples.append(testing_negative_samples)
y_test_samples = y_testing_positive_samples.append(y_testing_negative_samples)

#############################################################################################################



# from dataFrame to numpy array
X_training_samples_array = X_training_samples.values
y_training_samples_array = y_training_samples.values

X_test_samples_array = X_test_samples.values
y_test_samples_array = y_test_samples.values


model = svm.SVC(gamma='auto')
model.fit(X_training_samples_array, y_training_samples_array)

# Remember: 1 is positive that is malign, in other case it is negative that is benign.

# to predict only one single sample, use the following two lines
#X_test_sample = X_test_samples_array[0,:].reshape(1, -1)    # it contains a single sample.
#print(model.predict(X_test_sample))

# Predict all the test samples
prediction = model.predict(X_test_samples_array)
print("Predicition:    {}.".format(prediction))
print("y_test_samples: {}.".format(y_test_samples_array))
accuracy = (np.sum(prediction == y_test_samples_array) / y_test_samples_array.size) * 100
print("")
print("Accuracy: {}.".format(accuracy))