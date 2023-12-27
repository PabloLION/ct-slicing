__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn import svm

from ct_slicing.config.data_path import OUTPUT_FOLDER
from ct_slicing.ct_logger import logger

FEATURES_EXCEL_PATH = OUTPUT_FOLDER / "features.xlsx"
# Parameters to be configured #


def as_df(v: pd.Series | pd.DataFrame | list) -> pd.DataFrame:
    return cast(pd.DataFrame, v)


def make_positive_negative_lists(filename: Path):
    # Make two list from the filename, which is an .xlsx file.
    # And return two list with positive and negative samples each one.
    # Positive = 1 and Negative = 0.

    df = pd.read_excel(filename, sheet_name="Sheet1", engine="openpyxl")
    logger.info(f"Reading {filename=}")
    df.head()

    # meaning form l3_3: Positive (malign) = 1, Negative (benign) = 0.
    positive_samples = df.loc[df["diagnosis"] == 1]
    negative_samples = df.loc[df["diagnosis"] == 0]

    logger.info(f"Positive samples: {len(positive_samples)}")
    logger.info(f"Negative samples: {len(negative_samples)}")

    return positive_samples, negative_samples


proportion = 0.5  # define the training samples proportion


all_positive_samples, all_negative_samples = make_positive_negative_lists(
    FEATURES_EXCEL_PATH
)

all_positive_samples = as_df(shuffle(all_positive_samples))
all_negative_samples = as_df(shuffle(all_negative_samples))

# Select manually the features
selection = [
    "original_firstorder_10Percentile",
    "original_firstorder_90Percentile",
    "original_firstorder_Energy",
    "original_firstorder_Entropy",
    "original_firstorder_InterquartileRange",
    "original_firstorder_Kurtosis",
    "original_firstorder_Maximum",
    "original_firstorder_10Percentile",
    "original_firstorder_90Percentile",
    "original_firstorder_Energy",
    "original_firstorder_Entropy",
    "original_firstorder_InterquartileRange",
    "original_firstorder_Kurtosis",
    "original_firstorder_Maximum",
]
DF = pd.DataFrame

# applying the feature selection (it removes any column that is not included in the 'selection' list)
positive_samples = all_positive_samples.loc[:, selection]
negative_samples = all_negative_samples.loc[:, selection]

y_positive_samples = as_df(all_positive_samples.loc[:, "diagnosis"])
y_negative_samples = as_df(all_negative_samples.loc[:, "diagnosis"])

###################### RANDOM SELECTION OF THE TRAIN AND TEST SET  ##########################################
#### It is necessary to select a proportion of samples from positive and negative cases.

# Random selection of the train and test set from POSITIVE samples
num_positive_samples = len(positive_samples)
num_training_positive_samples = int(num_positive_samples * proportion)
all_positive_indexes = range(0, num_positive_samples)

training_positive_indexes = random.sample(
    all_positive_indexes, int(num_positive_samples * proportion)
)
testing_positive_indexes = [
    x for x in all_positive_indexes if x not in training_positive_indexes
]

training_positive_samples = as_df(positive_samples.iloc[training_positive_indexes, :])
y_training_positive_samples = as_df(y_positive_samples.iloc[training_positive_indexes])

testing_positive_samples = as_df(positive_samples.iloc[testing_positive_indexes, :])
y_testing_positive_samples = as_df(y_positive_samples.iloc[testing_positive_indexes])

logger.info("Training positive samples: {}.".format(len(training_positive_samples)))
logger.info("Testing positive samples: {}.".format(len(testing_positive_samples)))


# Random selection of the train and test set from NEGATIVE samples
num_negative_samples = len(negative_samples)
num_training_negative_samples = int(num_negative_samples * proportion)
all_negative_indexes = range(0, num_negative_samples)

training_negative_indexes = random.sample(
    all_negative_indexes, int(num_negative_samples * proportion)
)
testing_negative_indexes = [
    x for x in all_negative_indexes if x not in training_negative_indexes
]

training_negative_samples = as_df(negative_samples.iloc[training_negative_indexes, :])
y_training_negative_samples = as_df(y_negative_samples.iloc[training_negative_indexes])

testing_negative_samples = as_df(negative_samples.iloc[testing_negative_indexes, :])
y_testing_negative_samples = as_df(y_negative_samples.iloc[testing_negative_indexes])

# train and test sets
X_training_samples = pd.concat([training_positive_samples, training_negative_samples])
y_training_samples = pd.concat(
    [y_training_positive_samples, y_training_negative_samples]
)

X_test_samples = pd.concat([testing_positive_samples, testing_negative_samples])
y_test_samples = pd.concat([y_testing_positive_samples, y_testing_negative_samples])

# from dataFrame to numpy array
X_training_samples_array = X_training_samples.values
y_training_samples_array = y_training_samples.values

X_test_samples_array = X_test_samples.values
y_test_samples_array = y_test_samples.values

if np.unique(y_training_samples_array).size == 1:
    print("ERROR: y_training_samples_array has only one class.")  # TODO: log
    exit()

model = svm.SVC(gamma="auto")
model.fit(X_training_samples_array, y_training_samples_array)

# Remember: 1 is positive that is malign, in other case it is negative that is benign.

# to predict only one single sample, use the following two lines
# X_test_sample = X_test_samples_array[0,:].reshape(1, -1)    # it contains a single sample.
# print(model.predict(X_test_sample))

# Predict all the test samples
prediction = model.predict(X_test_samples_array)
logger.info(f"Prediction:     {prediction}")
logger.info(f"y_test_samples: {y_test_samples_array}")
accuracy = np.sum(prediction == y_test_samples_array) / y_test_samples_array.size
logger.info(f"Accuracy: {accuracy*100}%")

"""
# Exercise
## Exercise 1. Using PyRadiomics, make a manual feature selection and
classification.

a) Using PyRadiomics, extract only GLCM features from all slices (of the nodules) in the
database and save them in an EXCEL file. For this purpose modify the
featureExtraction.py file as needed.

    This is done in `l3_3_feature_extraction.py`

b) Why would we make a feature selection?
    Because we want to reduce the number of features to reduce the complexity
    of the model and to avoid overfitting.
    The selection is done with the variable `selection` in this file.

c) Split the already extracted features obtained in a) in a training (70%) and test (30%) set.
In the code featuresExtractionSVM.py, then run the code.
    This is done with the variables `proportion` in this file. We can easily
    change the value of `proportion` to change the proportion of training and
    test set.
    The code for splitting is under the comment 
    `## RANDOM SELECTION OF THE TRAIN AND TEST SET ##`
"""
