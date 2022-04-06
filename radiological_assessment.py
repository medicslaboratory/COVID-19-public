# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:33:51 2020

@author: gosek
"""

import os
import pandas as pd
import numpy as np 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
from sklearn.feature_selection import f_classif, VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from dataset import DBDataSet, execute_query

from skimage.measure import block_reduce
from sklearn.linear_model import LinearRegression, Lasso

import matplotlib.pyplot as plt

#We extract 10 crops per CXR
crop_nbr = 10

#Utility functions for multiple crops
def expand_indices(indice):
    return np.reshape([[a*10,a*10+1,a*10+2,a*10+3,a*10+4,a*10+5,a*10+6,a*10+7,a*10+8,a*10+9] for a in indice], (-1))

#Utility functions for multiple crops
def expand_labels(labels):
    return np.reshape([[a,a,a,a,a,a,a,a,a,a] for a in labels], (-1))

#Utility functions for multiple crops
def average_n(arr, n):
    arr = block_reduce(np.reshape(arr,(-1,1)), block_size=(n,1), func=np.mean)
    return arr

# Return AUC plot with confidence intervals (AUCs as an unsorted array with multiple bootstraps) Max 3 curves.
def plot_AUCs(AUCs, ROCs, labels, title):
    plt.close()
    if len(labels) > 3:
        print('Too many curves')
    colors = [(155/255, 199/255, 69/255), (222/255, 95/255, 45/255), (37/255, 58/255, 107/255)]
    i = 0
    for label in labels:
        AUC = AUCs[i]
        ROC = ROCs[i]
        if len(AUC) != 1:
            #sort arrays by AUC and fill the area between middle 95% curves
            s_AUC = np.sort(AUC)
            plt.fill(np.append(ROC[AUC.index(s_AUC[2])][0], ROC[AUC.index(s_AUC[-3])][0][::-1]), np.append(ROC[AUC.index(s_AUC[2])][1], ROC[AUC.index(s_AUC[-3])][1][::-1]), alpha = 0.5, color = colors[i] )
            plt.plot(ROC[-1][0], ROC[-1][1], linestyle='solid', label= label + str("{:.2f}".format(AUC[-1])), color = colors[i])
        else:
            plt.plot(ROC[0], ROC[1], linestyle='solid', label= label + str("{:.2f}".format(AUC[-1])), color = colors[i])

        i += 1
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(title)
    plt.close()

#Inverse of the low_variance_filter. Find and filter normalized high-variance features within random crops
class high_variance_filter():

    def __init__(self, t):

        self.t_ = t
        self.selected_ = []

    def fit_transform(self, features):
        self.fit(features)

        return self.transform(features)

    def fit(self, features):
        average_feat = np.mean(np.array(features),  axis = 0)
        average_var = block_reduce(np.array(features), block_size=(crop_nbr,1), func=np.var)
        #Compute normalized variance when the average is not 0, else default value is 1000.
        variance = np.mean(np.divide(average_var, average_feat, out=np.ones_like(average_var)*1000, where=average_feat!=0), axis = 0)
        selected = np.array([i for i, v in enumerate(variance) if v < self.t_]).astype(int)
        self.selected_ = selected

    def transform(self, features):
        return np.array(features)[:,self.selected_]

#For a given hyperparameters, features and data split, compute leave-patient-out crossvalidation
def get_val_AUC(matched_features, LOPO_cv, labels, params):

    svm = SVC(class_weight = 'balanced', kernel = params[0], C = params[2], degree = params[4], gamma = params[5], probability = False )
    preds_probs = []
    val_indices = np.array([]).astype(int)

    #High-variance filter, followed by feature selection, followed by scaling and training on a fold.
    for fold in LOPO_cv: 
        val_indices = np.concatenate((val_indices, fold[1]))
        f = high_variance_filter(params[3])
        scaler = MinMaxScaler()
        filtered = f.fit_transform(matched_features[fold[0]])
        fpr = SelectKBest(score_func = f_classif, k=params[1])
        deep_features = fpr.fit_transform(filtered, labels[fold[0]])

        svm.fit(scaler.fit_transform(deep_features), labels[fold[0]])
        pred_probas = svm.decision_function(scaler.transform(fpr.transform(f.transform(matched_features[fold[1]]))))
        for ii in pred_probas:
            preds_probs.append(ii)

    return preds_probs, val_indices

#Random search for SVM hyperparameter optimization. 
#Returns the best AUC and the associated hyperparameters.
def hyperparam_search(matched_features, LOPO_cv, labels):

    #hyperparameter grid
    feat_n = np.arange(2, 20)
    kerns = ('sigmoid','poly')
    Cs = [0.1, 0.2, 0.3, 0.4, 1, 2, 3, 5, 6, 10, 20, 30, 50]
    Vs = [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1]
    deg = [2,3]
    gam = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 2, 3, 5, 10]
    score = []
    params = []

    for it in range(6000):
        if it%10 == 0 and it != 0: 
            print(it)
            print(np.max(score))
            print(params[score.index(np.max(score))])

        kern = np.random.choice(kerns,1)
        n = int(np.random.choice(feat_n,1))
        c = np.random.choice(Cs,1)
        v = np.random.choice(Vs,1)
        d = np.random.choice(deg,1)
        g = np.random.choice(gam,1)

        preds_probs, val_indices = get_val_AUC(matched_features, LOPO_cv, labels, [kern,n,c,v,d,g])

        score.append(roc_auc_score(labels[val_indices], preds_probs))
        params.append([kern,n,c,v,d,g])

    print(np.max(score))
    return params[score.index(np.max(score))]

#Random search for linear regression hyperparameter optimization. 
#Returns the lowest error and the associated hyperparameters.
def hyperparam_search_linreg(matched_features, LOPO_cv, regression_value):
    feat_n = np.arange(5,100)
    Vs = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4]
    score = []
    params = []

    for it in range(10000):
        if it%100 == 0 and it != 0: 
            print(it)
            print(np.min(score))
            print(params[score.index(np.min(score))])

        n = int(np.random.choice(feat_n, 1))
        v = np.random.choice(Vs, 1)

        svm = LinearRegression(fit_intercept = True)
        preds = []
        preds_probs = []
        val_indices = np.array([]).astype(int)


        for fold in LOPO_cv:
            val_indices = np.concatenate((val_indices, fold[1]))

            f = high_variance_filter(v)
            scaler = MinMaxScaler()
            filtered = f.fit_transform(matched_features)
            fpr = SelectKBest(score_func = f_regression, k=n)
            deep_features = fpr.fit_transform(filtered[fold[0]], regression_value[fold[0]])
            svm.fit(scaler.fit_transform(deep_features), regression_value[fold[0]])
            pred_probas = svm.predict(scaler.transform(fpr.transform(filtered[fold[1]])))
            for i in pred_probas:
                preds_probs.append(i)

        score.append(np.mean(np.abs((regression_value[val_indices]-preds_probs))))
        params.append([n,v])

    print(np.min(score))
    return params[score.index(np.min(score))]

#Bootstrap the training set 100 times to compute the 95% CI
def get_CI_AUC(training_s, train_features, train_labels, test_features, test_labels, params):

    svm = SVC(class_weight = 'balanced', kernel = params[0], C = params[2], degree = params[4], gamma = params[5], probability = False )

    AUCs_deep = []
    ROCs_deep = []

    #select only images with a non-stable radiological trajectory
    imp_training_set = np.array([i for i, l in zip(training_s, train_labels[training_s]) if l == 0])
    wor_training_set = np.array([i for i, l in zip(training_s, train_labels[training_s]) if l == 1])

    f = high_variance_filter(params[3])
    f.fit(train_features[training_s])
    filtered = f.transform(train_features)
    skb = SelectKBest(score_func = f_classif, k=params[1])
    skb.fit(filtered[training_s], train_labels[training_s])
    s_train_features = skb.transform(filtered)
    scaler = MinMaxScaler()
    test_filtered = f.transform(test_features)

    for i in range(100):

        np.random.seed(i)
        #select all the crops from a given image
        if i != 99:
            chosen_s = np.random.choice(imp_training_set[0::crop_nbr]/crop_nbr, int(len(imp_training_set)/crop_nbr), replace = True)
            chosen_d = np.random.choice(wor_training_set[0::crop_nbr]/crop_nbr, int(len(wor_training_set)/crop_nbr), replace = True)
            training_set = expand_indices(np.concatenate((chosen_s, chosen_d), axis = -1)).astype(int)
        else: 
            #Compute the real AUC using the whole training set once
            training_set = training_s

        svm.fit(scaler.fit_transform(s_train_features[training_set]), train_labels[training_set])
        pred_probas = svm.decision_function(scaler.transform(skb.transform(test_filtered)))
            
        fpr, tpr, threshold = roc_curve(test_labels.astype(bool), pred_probas)
        AUCs_deep.append(roc_auc_score(test_labels.astype(bool), pred_probas))
        ROCs_deep.append([fpr, tpr, threshold])

    print('*AUCS for deep features*:')
    print('Median: ' + str(np.median(AUCs_deep)))
    print('Maximum: ' + str(np.max(AUCs_deep)))
    print('5 CI: ' + str(np.sort(AUCs_deep)[2]))
    print('95 CI: ' + str(np.sort(AUCs_deep)[-3]))
    print()

    return AUCs_deep, ROCs_deep

#Load outcomes and available images from the Quebec study.
codeair_DB_summary = pd.read_csv('codeair_DB_summary_anon.csv')
image_list = codeair_DB_summary['Images'].to_numpy()
IDs_codeair = codeair_DB_summary['Ids'].to_numpy()
outcomes_codeair = expand_labels(codeair_DB_summary['outcomes'].to_numpy())

#Load severity assessment from the Quebec study
qa_file = pd.read_csv('EQA_codeair_anon.csv')
qa_scores = []

for im in image_list:
    qa_scores.append(qa_file.iloc[qa_file.index[qa_file['image_name'] == os.path.split(im)[-1].strip()]+1]['PROGRESSION (0, 1, 2)'].to_numpy())
    if not qa_scores[-1].size:
        print(os.path.split(im)[-1])

#Load the features
deep_features = pd.read_csv('covid_features.zip', header = None)
deep_features_codeair = pd.read_csv('features_progression.zip', header = None)
#Load the progression labels and the test/train split
preds_n_labels = pd.read_csv('compilation_SD.csv')
test_labels = pd.read_csv('test.csv')
#True for the test variable will compute the testing set AUC, ROCs, and confidence intervals
test = True
test_codeair = True
#True if we want to perform the hyperparameter optimization
rdmSearch = False

# Codeair study encoded Stable as 1, Improve as 2 and Worse as 0. 
# We use 0.5 as censoring value for stable trajectories
# We use -0.5 as censoring value when there it is the last image
code_air_dict = {'0': 1, '1': 0.5, '2': 0, '-': -0.5, 'X=0': 1, 'X=1': 0.5, 'X=2': 0}
# No disease:0, Mild:1/3, Severe:2/3, Critical:3/3
severity_dict = {'N': 0, 'M': 1/3, 'S': 2/3, 'C': 3/3}
# Open COVID used 2 as the Stable censoring value
evolution_dict = {'W': 0, 'S': 2, 'I': 1}
severity_labels = ['No disease', 'Mild', 'Severe', 'Critical']

evo_scores = [code_air_dict[value[0]] for index, value in enumerate(qa_scores)]

#We select images from the dataset based on their indices 
#Excluding 'Stable' label
test_codeair_indices = [index for index, value in enumerate(evo_scores) if value in [0,1]]
preds_n_labels_binary = preds_n_labels.loc[preds_n_labels['Evolution'] != 'S']
indices = np.array(preds_n_labels_binary['Indice'].tolist())
#Expanding labels to account for the 10 crops
labels = expand_labels([evolution_dict[a] for a in preds_n_labels_binary['Evolution'].tolist()])
severity_all = np.reshape(expand_labels([severity_dict[a] for a in preds_n_labels_binary['T1'].tolist()]),(-1,1))
IDs = expand_labels(preds_n_labels_binary['ID'].values)
evo_scores = expand_labels(evo_scores)

test_labels_binary = test_labels.loc[test_labels['Evolution'] != 'S']
test_indices = np.array(test_labels_binary['Indice'].tolist())
test_labels = expand_labels([evolution_dict[a] for a in test_labels_binary['Evolution'].tolist()])
test_severity_all = np.reshape(expand_labels([severity_dict[a] for a in test_labels_binary['T1'].tolist()]),(-1,1))
severity_codeair = np.ones_like(outcomes_codeair).reshape(-1,1)

individual_ID = list(dict.fromkeys(IDs))

matched_features = np.reshape(deep_features.iloc[expand_indices(indices),:].to_numpy(), (-1,1024))
test_features = np.reshape(deep_features.iloc[expand_indices(test_indices),:].to_numpy(), (-1,1024))
test_features_codeair = np.reshape(deep_features_codeair.iloc[expand_indices(test_codeair_indices),:].to_numpy(), (-1,1024))

#leave-patient-out cross_validation (COVID-19 image data repository)
#The cross-validation folds contains the indices of the images in the features/labels set
LOPO_cv = []
for ID in individual_ID:
    LOPO_cv.append([[index for index, value in enumerate(IDs) if value != ID], [index for index, value in enumerate(IDs) if value == ID]])

split = pd.read_csv('train_test.csv')
test_set = split.loc[split['set'] == 'test']['subjid'].to_numpy()
train_set = split.loc[split['set'] == 'train']['subjid'].to_numpy()

#leave-patient-out cross_validation (Quebec study)
individual_ID_codeair = list(dict.fromkeys(IDs_codeair))

train_IDs = []
LOPO_cv_codeair = []
test_fold_codeair = []
full_val_codeair = []
for ID in individual_ID_codeair:
    if ID in test_set:
        #All patients in the test set
        fold = [index for index, value in enumerate(IDs_codeair) if value in test_set and index in test_codeair_indices]
        #A single patient in the test set
        unique = [index for index, value in enumerate(IDs_codeair) if value == ID and index in test_codeair_indices]
        if len(unique):
            test_fold_codeair.append(expand_indices(fold))
    else:
        # A fold is the whole training set - one patient as the element 0 and the one patient as the element 1
        train_IDs.append(ID)
        fold = [expand_indices([index for index, value in enumerate(IDs_codeair) if value != ID and value not in test_set and index in test_codeair_indices]), [index for index, value in enumerate(IDs_codeair) if value == ID and index in test_codeair_indices]]
        # Make sure there are elements in the testing set before appending
        fold[1] = expand_indices([index for it, index in enumerate(fold[1])])
        if fold[1].size:
            LOPO_cv_codeair.append(fold)

test_CV_codeair = [i[1] for i in LOPO_cv_codeair]

#the whole training set is the whole first crossvalidation fold
training_set_codeair = np.concatenate((LOPO_cv_codeair[0][0], LOPO_cv_codeair[0][1]))
# We just need the first fold of the testing set as it contains all items already.
testing_set_codeair = np.array(test_fold_codeair[0])

if rdmSearch:
    #Perform hyperparameters search
    params = hyperparam_search(matched_features, LOPO_cv, labels)
else:
    #Those are the best hyperparameters found on the testing set
    params = ['sigmoid', 22, 0.09, 0.05, 1, 1] 

print(params)
svm = SVC(class_weight = 'balanced', kernel = params[0], C = params[2], degree = params[4], gamma = params[5], probability = False )
val_probs, val_indices = get_val_AUC(matched_features, LOPO_cv, labels, params)

preds = []
preds_probs = []
print('**** Evolution prediction ****')
feat_n = params[1]
#Train on each fold and test on the left-out patient
for fold in LOPO_cv: 
    f = high_variance_filter(params[3])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(matched_features)
    fpr = SelectKBest(score_func = f_classif, k=feat_n)
    deep_features = fpr.fit_transform(filtered[fold[0]], labels[fold[0]])

    svm.fit(scaler.fit_transform(deep_features), labels[fold[0]])
    pred = svm.predict(scaler.transform(fpr.transform(filtered[fold[1]])))
    pred_probas = svm.decision_function(scaler.transform(fpr.transform(filtered[fold[1]])))

    for i, ii in zip(pred, pred_probas):
        preds.append(i)
        preds_probs.append(ii)

#Compute ROC curve parameters from the cross-validation results
fpr_val, tpr_val, threshold_val = roc_curve(labels[val_indices].astype(bool), preds_probs)
val_AUC = roc_auc_score(labels[val_indices], preds_probs)

print('Radiological progression prediction: Cross Validation results')

print('F1 score: ' + str(f1_score(labels[val_indices], preds)))

print(confusion_matrix(labels[val_indices], preds))

print('AUC score: ' + str(val_AUC))
print()

if test:
    print('Radiological progression prediction: Test results')

    #Get confidence intervals using the cross-validated hyperparameters
    AUCs, ROCs = get_CI_AUC(np.concatenate((LOPO_cv[0][0], LOPO_cv[0][1]), axis = -1), matched_features, labels, test_features, test_labels, params)

    f = high_variance_filter(params[3])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(matched_features)
    test_filtered = f.transform(test_features)
    fpr = SelectKBest(score_func = f_classif, k=feat_n)
    selected_features = fpr.fit_transform(filtered, labels)

    svm.fit(scaler.fit_transform(selected_features), labels)
    pred = svm.predict(scaler.transform(fpr.transform(test_filtered)))
    pred_probas = svm.decision_function(scaler.transform(fpr.transform(test_filtered)))
        
    fpr, tpr, threshold = roc_curve(test_labels.astype(bool), pred_probas)

    print('F1 score: ' + str(f1_score(test_labels, pred)))
    
    print(confusion_matrix(test_labels, (pred_probas)>0))
    
    print('AUC score: ' + str(roc_auc_score(test_labels, pred_probas)))
    print()
    plot_AUCs([AUCs, [val_AUC]], [ROCs, [fpr_val, tpr_val]], ['SVM, 22 deep features, test AUC: ', 'SVM, 22 deep features, validation AUC: '], 'results/ROC_CI_test.png')

#Reproduce the same method on the Quebec dataset
if rdmSearch:
    params = hyperparam_search(np.reshape(deep_features_codeair.to_numpy(), (-1,1024)), LOPO_cv_codeair, evo_scores)
else:
    params = ['sigmoid', 2, 0.06, 0.001, 5, 1]  

print(params)

svm = SVC(class_weight = 'balanced', kernel = params[0], C = params[2], degree = params[4], gamma = params[5], probability = False)

if test_codeair:
    print('Radiological progression prediction: Test results on Quebec study')

    val_probs, val_indices = get_val_AUC(np.reshape(deep_features_codeair.to_numpy(), (-1,1024)), LOPO_cv_codeair, evo_scores, params)
    AUCs_codeair, ROCs_codeair = get_CI_AUC(training_set_codeair, np.reshape(deep_features_codeair.to_numpy(), (-1,1024)), evo_scores, np.reshape(deep_features_codeair.to_numpy(), (-1,1024))[testing_set_codeair], evo_scores[testing_set_codeair], params)

    print(confusion_matrix(evo_scores[val_indices], np.array(val_probs)>0))   

    f = high_variance_filter(params[3])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(np.reshape(deep_features_codeair.to_numpy(), (-1,1024))[training_set_codeair])
    test_filtered = f.transform(np.reshape(deep_features_codeair.to_numpy(), (-1,1024))[testing_set_codeair])
    fpr = SelectKBest(score_func = f_classif, k=params[1])
    selected_features = fpr.fit_transform(filtered, evo_scores[training_set_codeair])

    svm.fit(scaler.fit_transform(selected_features), evo_scores[training_set_codeair])
    pred = svm.predict(scaler.transform(fpr.transform(test_filtered)),)
    pred_probas = svm.decision_function(scaler.transform(fpr.transform(test_filtered)))

    fpr, tpr, threshold = roc_curve(evo_scores[testing_set_codeair].astype(bool), pred_probas)
    fpr_val, tpr_val, threshold_val = roc_curve(evo_scores[val_indices].astype(bool), val_probs)

    print('F1 score: ' + str(f1_score(evo_scores[testing_set_codeair], pred)))
    
    print(confusion_matrix(evo_scores[testing_set_codeair], (pred_probas)>0))
    
    print('AUC score: ' + str(roc_auc_score(evo_scores[testing_set_codeair], pred_probas)))
    print()
    
    plot_AUCs([AUCs_codeair, [roc_auc_score(evo_scores[val_indices], val_probs)]], [ROCs_codeair, [fpr_val, tpr_val]], ['SVM, 2 deep features, test AUC: ', 'SVM, 2 deep features, validation AUC: '], 'results/ROC_CI_codeair.png')

print('**** Severity evaluation ****')
#Re-scale the severity from 0 to 3
severity_all *= 3
test_severity_all *= 3

if rdmSearch:
    hypers = hyperparam_search_linreg(matched_features, LOPO_cv, severity_all)
else:
    hypers = [5, 0.0001]

print(hypers)

#We use linear regression for severity estimation
linreg = LinearRegression(fit_intercept = True)
preds= []
for fold in LOPO_cv:
    f = high_variance_filter(hypers[1])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(matched_features)
    fpr = SelectKBest(score_func = f_regression, k=hypers[0])
    deep_features = fpr.fit_transform(filtered[fold[0]], severity_all[fold[0]])
    linreg.fit(scaler.fit_transform(deep_features), severity_all[fold[0]])
    pred = linreg.predict(scaler.transform(fpr.transform(filtered[fold[1]])))
    for i in pred:
        preds.append(i)

#Get performances measurements
print('Severity prediction: Cross Validation results')
print('Average error: ' + str(np.mean(np.abs(preds-severity_all))))
print(confusion_matrix(severity_all, np.clip(np.round(preds),a_min = 0, a_max = 3)))
print('Accuracy: ' + str(accuracy_score(severity_all, np.clip(np.round(preds),a_min = 0, a_max = 3))))
print()

#Using the cross-validated hyperparameters from the previous step, test on the remaining images
if test:
    f = high_variance_filter(hypers[1])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(matched_features)
    test_filtered = f.transform(test_features)
    fpr = SelectKBest(score_func = f_regression, k=hypers[0])
    deep_features = fpr.fit_transform(filtered, severity_all)
    linreg.fit(scaler.fit_transform(deep_features), severity_all)
    pred = linreg.predict(scaler.transform(fpr.transform(test_filtered)))
    
    cm = confusion_matrix(test_severity_all, np.clip(np.round(pred),a_min = 0, a_max = 3), normalize = 'true', labels = [0,1,2,3])
    print('Severity prediction: Test results')

    print('Average error: ' + str(np.mean(np.abs(pred-test_severity_all))))
    print(cm)
    print('Accuracy: ' + str(accuracy_score(test_severity_all, np.clip(np.round(pred),a_min = 0, a_max = 3))))
    print()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    ax.set_xticklabels([''] + severity_labels)
    ax.set_yticklabels([''] + severity_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png', dpi = 1000)

#Using the cross-validated hyperparameters from the previous step, apply on the Quebec dataset without retraining
#Here we assume that the severity in all those patients is at 'Critical' considering that they are all in the ICU
#However severity was not formally assessed in those images
if test_codeair:
    f = high_variance_filter(hypers[1])
    scaler = MinMaxScaler()
    filtered = f.fit_transform(np.concatenate((matched_features, test_features)))
    test_filtered = f.transform(np.reshape(deep_features_codeair.to_numpy(), (-1,1024)))
    fpr = SelectKBest(score_func = f_regression, k=hypers[0])
    deep_features = fpr.fit_transform(filtered, np.concatenate((severity_all, test_severity_all)))
    linreg.fit(scaler.fit_transform(deep_features), np.concatenate((severity_all, test_severity_all)))
    pred = linreg.predict(scaler.transform(fpr.transform(test_filtered)))
    
    cm = confusion_matrix(severity_codeair*3, np.clip(np.round(pred),a_min = 0, a_max = 3), normalize = 'true')
    print('Severity prediction: Generalization results on Quebec study')
    print('Average error: ' + str(np.mean(np.abs(pred-severity_codeair*3))))
    print(cm)
    print('Accuracy: ' + str(accuracy_score(severity_codeair*3, np.clip(np.round(pred),a_min = 0, a_max = 3))))
