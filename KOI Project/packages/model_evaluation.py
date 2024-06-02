#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier 
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as IPipeline
from xgboost import XGBClassifier
from joblib import parallel_backend
import re

# Calculates the class k recall by counting the true positives and false negatives relative to class k
# Takes in true y labels, predicted y labels, and class
def class_k_recall(y, y_pred, k = 0):
    pairs = list(zip(y, y_pred))
    num_of_variables = len(list(set(y)))
    
    # Correctly predicted on class k
    tp = pairs.count((k, k))
    fn = 0
    
    # Row-wise misses on class k (from confusion matrix perspective)
    for i in range(num_of_variables):
        if i == k:
            continue
        fn += pairs.count((k,i))
    
    return tp / (fn + tp)

# Calculates the class k precision by counting the true positives and false positives relative to class k
# Takes in true y labels, predicted y labels, and class
def class_k_precision(y, y_pred, k = 0):
    pairs = list(zip(y, y_pred))
    num_of_variables = len(list(set(y)))
    
    # Correctly predicted on class k
    tp = pairs.count((k, k))
    fp = 0
    
    # Column-wse misses on class k (from confusion matrix perspective)
    for i in range(num_of_variables):
        if i == k:
            continue
        fp += pairs.count((i, k))
        
    return tp / (tp + fp)

# Calculates the class k f1_score using class k recall and class k precision
# Takes in true y labels, predicted y labels, and class
def class_k_f1(y, y_pred, k = 0):
    recall = class_k_recall(y, y_pred, k)
    precision = class_k_precision(y, y_pred, k)
    
    return 2 * recall * precision / (recall + precision)

# Calculates the class k accuracy by counting the true positives and misses relative to class k
# Takes in true y labels, predicted y labels, and class
def class_k_acc(y, y_pred, k = 0):
    pairs = list(zip(y, y_pred))
    num_of_variables = len(list(set(y)))
    
    # Correctly predicted on class k
    tp = pairs.count((k, k))
    miss = 0
    
    # Row-wise and column-wise misses on class k (from confusion matrix perspective)
    for i in range(num_of_variables):
        if i == k:
            continue
        miss += pairs.count((i, k)) + pairs.count((k, i))
        
    return tp / (tp + miss)

# Takes in X_test data, y_test data, and a fitted model
# Extracts feature importances and classification report metrics, as well as (weighted) ROC-AUC
# Returns two Pandas DataFrames, one for classification metrics and ROC-AUC, one for feature importances
def evaluate_model(X_test, y_test, model):
    
    num_classes = len(y_test.value_counts())
    
    # If the model is a GridSearch, walk down the best estimator to get feature_names and feature_importances
    # Else, walk down the Pipeline to get feature_names and feature_importances
    # Hardcoded to work with Pipeline framework: (preprocessing, resampler, classifier)
    # Where preprocessing = (StandardScaler(), OneHotEncoder())
    if type(model) == GridSearchCV:
#         feature_names = model.best_estimator_.steps[0][1].transformers_[-1][1].get_feature_names()
#         one_hot_col = model.best_estimator_.steps[0][1].transformers_[-1][-1]
        feature_importances = model.best_estimator_.steps[-1][-1].feature_importances_
    else:
#         feature_names = model.steps[0][1].transformers_[-1][1].categories_[0]
#         one_hot_col = model.steps[0][1].transformers_[-1][-1]
        # Hardcoded since LogisticRegression uses .coef_, most other SKLearn models use .feature_importances_
        if type(model.steps[-1][-1]) == LogisticRegression:
            feature_importances = model.steps[-1][-1].coef_[0]
        else:
            feature_importances = model.steps[-1][-1].feature_importances_
    
    # Get probabilities and predictions on X_test
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # If binary classification, roc_auc_score requires only y_probabilities corresponding to column 1
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # If multi-class classification, use one-versus-one roc_auc_score and weighted average.
    if num_classes > 2:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class = 'ovo', average = 'weighted')
    else:
        auc = roc_auc_score(y_test, y_pred_proba)
    
    # Initialize and populate recall, precision, and f1 scores for all classes
    recalls = []
    precisions = []
    f1s = []
    
    for i in range(num_classes):
        recalls.append(class_k_recall(y_test, y_pred, k = i))
        precisions.append(class_k_precision(y_test, y_pred, k = i))
        f1s.append(class_k_f1(y_test, y_pred, k = i))
        
    # Capture accuracy score between true labels and predicted labels
    accuracy = accuracy_score(y_test, y_pred)
    
    # Collect classification and roc_auc info into Pandas DataFrame
    classification_frame = pd.DataFrame(np.array([recalls, precisions, f1s]).T,
                                        columns = ['recall', 'precision', 'f1'])
    
    classification_frame['accuracy'] = [accuracy] + [None] * (num_classes - 1)
    classification_frame['auc'] = [auc] + [None] * (num_classes - 1)
    
    # Clean feature strings for OneHotEncoded features
#     for index, string in enumerate(feature_names):
#         for i in range(len(one_hot_col)):
#             feature_names[index] = re.sub(r'x{}'.format(i), one_hot_col[i], feature_names[index])
    
    # Get all feature names from X_test columns, with categorical columns replaced with OneHotEncoded equivalents
    # Hardcoded for all categorical columns to appear before non-categorical columns
#     feature_names = feature_names.tolist() + [x for x in X_test if x not in one_hot_col]
    feature_names = X_test.columns.values.tolist()
    
    # Collect feature importances and names into Pandas DataFrame
    feature_pairs = list(zip(feature_names, feature_importances))
    
    feature_frame = pd.DataFrame(feature_pairs, columns = ['feature', 'coefficient'])
    
    return classification_frame, feature_frame
    
