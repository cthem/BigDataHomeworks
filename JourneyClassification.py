import numpy as np
import pandas as pd
import sklearn
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import utils
import os


def classify(features, targets, num_folds, classifiers, output_folder, filename_tag = "", classifier_obj = None):
    kf = KFold(n_splits = num_folds)
    folds_idxs = list(kf.split(features))
    trips_array = np.asarray(features)
    targets = np.asarray(targets)

    if type(classifiers) != list:
        classifiers = [classifiers]
    # train/val accuracies
    accuracies = {}
    mean_accuracies = {}
    # classify
    for classifier in classifiers:
        accuracies[classifier] = []
        print("Testing classifier [%s]" % classifier)
        # train & test each classifier
        # for each fold
        for i, (train_idx, val_idx) in enumerate(folds_idxs):
            print("\tClassifing fold %d/%d" % (i + 1, len(folds_idxs)), end=" ")
            train = (trips_array[train_idx], targets[train_idx])
            val = (trips_array[val_idx], targets[val_idx])
            if classifier == "knn":
                k = 5
                accTrain, accVal = knn_classification(train, val, k)
            elif classifier == "logreg":
                accTrain, accVal = logreg_classification(train, val, lr_classifier=classifier_obj)
            elif classifier == "randfor":
                accTrain, accVal = randfor_classification(train, val)
            accuracies[classifier].append((accTrain, accVal))
            print("accuracies train/val:",accuracies[classifier][-1])

        # accuracy across all folds
        mean_accuracies[classifier] = [np.mean([x[0] for x in accuracies[classifier]]), \
                                       np.mean([x[1] for x in accuracies[classifier]])]
        titlestr = "%s, overall accuracy train/val: %s" % (classifier, str(mean_accuracies[classifier]))
        chart_filename = os.path.join(output_folder, classifier + "_" + filename_tag)
        utils.barchart(list(range(1, num_folds + 1)), accuracies[classifier], title=titlestr, ylabel="accuracy", legend=["train","val"],
                    save=chart_filename)

    return mean_accuracies

# TODO improve classification
def preprocess_data(feature_df):
    targets = []
    data = []
    # TODO make an options dict bundle cardundle
    seed = 123
    feature_df = feature_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    print(feature_df[:2])
    for index, row in feature_df.iterrows():
        targets.append(row["journeyId"])
        train_points = row["points"]
        train_points = eval(train_points)
        points = []
        for point in train_points:
            points.append(point[1])
        data.append(points)
    # get maximum length of feature lists
    maxlen = len(max(data, key=lambda x: len(x)))
    data = [[int(d[1:]) for d in dlist] for dlist in data]
    # pad to the maximum length
    for i, datum in enumerate(data):
        if len(datum) < maxlen:
            data[i] = datum + [3 for _ in range(maxlen - len(datum))]

    # convert journey ids to numbers
    num_ids = {}
    targets_nums = []
    for t in targets:
        if t not in num_ids:
            num_ids[t] = len(num_ids)
        targets_nums.append(num_ids[t])

    # count occurences
    hist = []
    for jid in num_ids:
        occurences = sum([1 if t == jid else 0 for t in targets])
        hist.append((jid, occurences))

    hist = sorted(hist, key=lambda x: x[1], reverse= True)
    print("Most frequent 5 jids:")
    for (jid, occ) in hist[:5]:
        print(jid, ":", occ)
    return data, targets_nums


def knn_classification(train, val, k):
    # folds contains 10 tuples (training and test sets)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(train[0], train[1])
    # get training and test accuracy
    res = knn_classifier.predict(train[0])
    accTrain = accuracy_score(train[1], res)
    res = knn_classifier.predict(val[0])
    accVal = accuracy_score(val[1], res)
    return accTrain, accVal
    #print(classification_report(val[1], res))


def logreg_classification(train, val, lr_classifier = None):

    if lr_classifier is None:
        lr_classifier = LogisticRegression()
    lr_classifier.fit(train[0], train[1])

    # get prediction by probabilty argmax for the predicted class
    res_prob = lr_classifier.predict_proba(val[0])
    res = np.argmax(res_prob, axis=1)
    accVal =  accuracy_score(val[1], res)

    res_prob = lr_classifier.predict_proba(train[0])
    res = np.argmax(res_prob, axis=1)
    accTrain =  accuracy_score(train[1], res)
    return accTrain, accVal


def randfor_classification(train, val):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train[0], train[1])

    res = rf_classifier.predict(train[0])
    accTrain = accuracy_score(train[1], res)
    res = rf_classifier.predict(val[0])
    accVal = accuracy_score(val[1], res)
    return accTrain, accVal


# test file in the same format as the features file
def improve_classification(features_file, num_folds, output_folder, classifier):
    print()
    print("Reading features:",features_file)
    train_df = pd.read_csv(features_file)
    grid_features, targets = preprocess_data(train_df)
    # try various techniques for improvement
    mean_accuracies = {}

    # try feature normalization
    norms = ["l2","l1","max"]
    for norm in norms:
        tag = "norm_%s" % norm
        print("Trying strategy: %s" % tag)
        grid_features = sklearn.preprocessing.normalize(grid_features, norm = norm, copy=False)
        mean_acc = classify(grid_features, targets, num_folds,classifier,output_folder, tag)
        mean_accuracies[tag] = mean_acc

    # try various regularization strengths
    Cs = [0.2, 0.5 , 1.5, 2.0]
    for c in Cs:
        tag = "C %s" % c
        print("Trying strategy: %s" % tag)
        classifier_obj = LogisticRegression(C=c)
        mean_acc = classify(grid_features, targets, num_folds,classifier,output_folder, tag, classifier_obj=classifier_obj)
        mean_accuracies[tag] = mean_acc

    return mean_accuracies

def compare_with_baseline(baseline):

    pass
