# CA05-logistic-regression
1. The Application

Cardiovascular Disease (CVD) kills more people than cancer globally. A dataset of real heart patients
collected from a 15 year heart study cohort is made available for this assignment. The dataset has 16
patient features. Note that none of the features include any Blood Test information.
2. Deliverables

Your job is to:

Part 1: build a binary classifier model to predict the CVD Risk (Yes/No, or 1/0) using a Logistic
Regression Model with the best performance possible (deliverable: Notebook)

Part 2: Display the Feature Importance of all the features sorted in the order of decreasing influence on
the CVD Risk (deliverable: Notebook)

Part 3: Evaluate the performance of your model (including ROC Curve), explain the performance and
draw a meaningful conclusion. (deliverable: Performance outputs in Notebook, explanation and
conclusion in Word/PDF document)
## Instructions
1. Open ipynb file
2. Click "Open in Colab" button or click the Colab link
3. Run code by lines or select runtime in the menu bar - click run all in the dropdown list
## Usage
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from matplotlib import pyplot

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve, roc_auc_score
## Questions answered
Explain the performance and draw a meaningful conclusion
