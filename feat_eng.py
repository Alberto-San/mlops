# for data processing and manipulation
import pandas as pd
import numpy as np

# scikit-learn modules for feature selection and model evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# libraries for visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def remove_nan_columns(df):
    columns_to_remove = ...
    df.drop(columns_to_remove, axis=1, inplace=True)
    return df

def encode_target_var(df):
    df["diagnosis_int"] = (df["diagnosis"] == 'M').astype('int')
    df.drop(['diagnosis'], axis=1, inplace=True)
    return df

def get_features(X, Y):
    X = df.drop("diagnosis_int", 1)
    Y = df["diagnosis_int"]
    return X, Y

def fit_model(X, Y):
    model = RandomForestClassifier(criterion='entropy', random_state=47)
    model.fit(X, Y)
    return model

def calculate_metrics(model, X_test_scaled, Y_test):
    y_predict_r = model.predict(X_test_scaled)
    roc=roc_auc_score(Y_test, y_predict_r)
    acc = accuracy_score(Y_test, y_predict_r)
    prec = precision_score(Y_test, y_predict_r)
    rec = recall_score(Y_test, y_predict_r)
    f1 = f1_score(Y_test, y_predict_r)
    return acc, roc, prec, rec, f1

def train_and_get_metrics(X, Y):
    '''Train a Random Forest Classifier and get evaluation metrics'''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = fit_model(X_train_scaled, Y_train)
    roc, acc, prec, rec, f1 = calculate_metrics(model, X_test_scaled, Y_test)
    return acc, roc, prec, rec, f1

def evaluate_model_on_features(X, Y):    
    acc, roc, prec, rec, f1 = train_and_get_metrics(X, Y)
    display_df = pd.DataFrame([[acc, roc, prec, rec, f1, X.shape[1]]], columns=["Accuracy", "ROC", "Precision", "Recall", "F1 Score", 'Feature Count'])    
    return display_df

def get_training_information(data):
    X, Y, index_name = data 
    features_eval_df = evaluate_model_on_features(X, Y)
    all_features_eval_df.index = [index_name]
    return all_features_eval_df

def correlation_features(df, label):
    cor = df.corr() 
    cor_target = abs(cor[label])
    relevant_features = cor_target[cor_target>0.2] # better to mantain variables with high correlation of the label
    names = [index for index, value in relevant_features.iteritems()]
    names.remove(label)
    X, Y, index_name = df[names], Y, 'Strong features'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)

def correlation_other_feats(df):
    cor = df.corr() 
    cor_target = abs(cor[label])
    relevant_features = cor_target[cor_target>0.2] # better to mantain variables with high correlation of the label
    names = [index for index, value in relevant_features.iteritems()]
    names.remove(label)
    new_corr = df[names].corr()
    highly_corr_feats = ...
    subset_feature_corr_names = [x for x in names if x not in highly_corr_feats]
    X, Y, index_name = df[subset_feature_corr_names], Y, 'Subset features'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)

def univariate_selection(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    selector = SelectKBest(f_classif, k=20)
    X_new = selector.fit_transform(X_train_scaled, Y_train)
    feature_idx = selector.get_support()
    feature_names = df.drop("diagnosis_int",1 ).columns[feature_idx]
    X, Y, index_name = df[feature_names], Y, 'F-test'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)

def run_rfe(X, Y, N=20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(criterion='entropy', random_state=47)
    rfe = RFE(model, N)
    rfe = rfe.fit(X_train_scaled, Y_train)
    feature_names = df.drop("diagnosis_int",1 ).columns[rfe.get_support()]
    X, Y, index_name = df[feature_names], Y, 'RFE'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)

def feature_importances_from_tree_based_model_(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier()
    model = model.fit(X_train_scaled,Y_train)
    model = SelectFromModel(model, prefit=True, threshold=0.013)
    feature_idx = model.get_support()
    feature_names = df.drop("diagnosis_int",1 ).columns[feature_idx]
    X, Y, index_name = df[feature_names], Y, 'Feature Importance'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)


def run_l1_regularization(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    selection = SelectFromModel(LinearSVC(C=1, penalty='l1', dual=False))
    selection.fit(X_train_scaled, Y_train)
    feature_names = df.drop("diagnosis_int",1 ).columns[(selection.get_support())]
    X, Y, index_name = df[feature_names], Y, 'L1 Reg'
    return Monad([X, Y, index_name ]).flatMap(get_training_information)

dataset_path = ...

df = (
    Monad(dataset_path) \
    .map(pd.read_csv) \
    .map(remove_nan_columns) \
    .flatMap(encode_target_var)
)

X, Y = Monad(df).flatMap(get_features)
result_df = Monad().reduce(
    [
        Monad([X, Y, 'All feats' ]).flatMap(get_training_information)
        correlation_features(df),
        correlation_other_feats(df),
        univariate_selection((X, Y)),
        run_rfe((X, Y)),
        feature_importances_from_tree_based_model_((X, Y)),
        run_l1_regularization((X, Y))
    ],
    lambda (x, y): pd.concat([x, y])
)


