import pandas as pd
import warnings
import copy
import sys
import pickle

from itertools import cycle
from sklearn import model_selection, metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from util.myplots import plotROCs
from util.settings import *
from util.process import *
from util.const import *

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


from util.utils import datasetname, create_userids, keeporder_split

from measurements.preprocess import get_user_data


# current_dataset: which dataset to evaluate
# dataset_amount: how many data use from the dataset (ALL, FIRST1000)
# num_actions: how many mouse actions to use for decision

def evaluate_dataset( current_dataset, dataset_amount, num_actions, num_training_actions):
    #filename = FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)
    filename1 = "/home/bwbwchen/temp/mouse_dynamics_balabit_chaoshen_dfl/measurements/lee_log"
    filename2 = "/home/bwbwchen/temp/mouse_dynamics_balabit_chaoshen_dfl/measurements/liu_log"
    """
    filename1 = "/home/bwbwchen/temp/mouse_dynamics_balabit_chaoshen_dfl/measurements/mouse_log"
    filename2 = "/home/bwbwchen/temp/mouse_dynamics_balabit_chaoshen_dfl/measurements/liu_log"
    """

    #print(filename1)
    #print(filename2)
    #dataset = pd.read_csv(filename)
    dataset = get_user_data(filename1, filename2)
    #print(dataset.shape)

    # DataFrame
    df = pd.DataFrame(dataset)

    num_features = int(dataset.shape[1])
    #print("Num features: ", num_features)
    array = dataset.values

    X = array[:, 0:num_features - 1]
    y = array[:, num_features - 1]

    userids = create_userids(current_dataset)
    userids = [1]

    #print(userids)


    # Train user-specific classifiers and evaluate them
    items = userids

    # fpr = {} <==> fpr = dict()
    fpr = {}
    tpr = {}
    roc_auc = {}

    correct = df.loc[df.iloc[:, -1].isin([1])]
    wrong = df.loc[df.iloc[:, -1].isin([2])]
    numSamples = min(correct.shape[0], wrong.shape[0])

    for i in userids:
        # print("Training classifier for the user "+str(i))
        # Select all positive samples that belong to current user
        user_positive_data = df.loc[df.iloc[:, -1].isin([i])]

        user_positive_data = user_positive_data.iloc[np.random.choice(user_positive_data.shape[0], numSamples)]
        #numSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
        array_positive[:, -1] = 1

        # negative data for the current user
        user_neagtive_data = select_negatives_from_other_users(dataset, i, numSamples)
        array_negative = copy.deepcopy(user_neagtive_data.values)
        array_negative[:, -1] = 0

        # concatenate negative and positive data
        dataset_user = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
        X = dataset_user[:, 0:-1]
        y = dataset_user[:, -1]

        if CURRENT_SPLIT_TYPE == SPLIT_TYPE.RANDOM:
            X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=TEST_SIZE,random_state= RANDOM_STATE)
            print ("random split")
        else:
            X_train, X_validation, y_train, y_validation = keeporder_split(X, y, test_size=TEST_SIZE)
            #print ("fuck split")

        model = RandomForestClassifier(random_state= RANDOM_STATE)
        model.fit(X_train, y_train)

        # scoring = ['accuracy', 'roc_auc' ]
        # scores = cross_validate(model, X_train, y_train, scoring=scoring, cv = 10, return_train_score = False)
        scores = cross_validate(model, X_train, y_train, cv=25, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        print ("validation shape ", X_validation.shape)
        y_predicted = model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f, y_predicted[0]" % test_accuracy, y_predicted[0])

        # save model
        with open('outmodel.pkl', 'wb') as f:
            pickle.dump(model, f)

        fpr[i], tpr[i], thr = evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions)

        threshold = -1
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr[i], tpr[i])(x), 0., 1.)
            threshold = interp1d(fpr[i], thr)(eer)
        except (ZeroDivisionError, ValueError):
            print("Division by zero")

        roc_auc[i] = auc(fpr[i], tpr[i])
        print(str(i) + ": " + str(roc_auc[i])+" threshold: "+str(threshold))

    #plotROCs(fpr, tpr, roc_auc, items)

def evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions):
    #print ("validation set", X_validation)
    #print ("validation set answer", y_validation)
    # print(len(X_validation))
    if num_actions == 1:
        y_scores = model.predict_proba(X_validation)
        writeCSVa(y_validation, y_scores[:, 1])
        return roc_curve(y_validation, y_scores[:, 1])

    X_val_positive = []
    X_val_negative = []
    for i in range(len(y_validation)):
        if y_validation[i] == 1:
            X_val_positive.append(X_validation[i])
        else:
            X_val_negative.append(X_validation[i])
    pos_scores = model.predict_proba(X_val_positive)
    neg_scores = model.predict_proba(X_val_negative)

    #print ("pos_scores", pos_scores)
    #print ("neg_scores", neg_scores)

    scores =[]
    labels =[]

    n_pos = len(X_val_positive)
    #print(" x val positive" , X_val_positive)
    #print (n_pos)
    for i in range(n_pos-num_actions+1):
        score = 0
        for j in range(num_actions):
            score += pos_scores[i+j][1]
        score /= num_actions
        scores.append(score)
        labels.append(1)

    n_neg = len(X_val_negative)
    for i in range(n_neg - num_actions + 1):
        score = 0
        for j in range(num_actions):
            score += neg_scores[i + j][1]
        score /= num_actions
        scores.append(score)
        labels.append(0)

    # writeCSVa(labels, scores)
    #print ("fucking labels", labels)
    #print ("fucking scores", scores)
    return roc_curve(labels, scores)

def select_negatives_from_other_users( dataset, userid, numsamples ):
    # num_features = dataset.shape[1]
    #other_users_data =  dataset['userid'] != userid
    other_users_data =  dataset[38] != userid
    dataset_negatives = dataset[other_users_data].sample(numsamples, random_state= RANDOM_STATE)
    return dataset_negatives

