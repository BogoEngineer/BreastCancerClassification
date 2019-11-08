My first machine learning project that helps predicting whether the breast tumor is recurrent or not. Features that are taken into account are: age, menopause, tumor size, (left-right) breast, (yes-no) node caps, (yes-no) irradiat.
Logistic regression model was trained on a small dataset(~300 samples), hence accuracy is not high.
The model was described and trained in Python(3.7), using modules such as sklearn and numpy. The pickle module was used for saving (and loading in) the best model so far. The pandas module was used for data manipulation.
Dataset used can be found on: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer
