import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# TODO get training loss, where applicable (else training accuracy itself), to measure overfit
# TODO improve classification
def preprocess_data(feature_df):
    targets = []
    data = []
    for index, row in feature_df.iterrows():
        targets.append(row["journeyId"])
        train_points = row["points"]
        train_points = eval(train_points)
        points = []
        for point in train_points:
            points.append(point[1])
        data.append(points)
    #for i,d in enumerate(data):
    #    print(targets[i], d)
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

    sorted(hist, key=lambda x: x[1])
    print("Most frequent 5 jids:")
    for (jid, occ) in hist[:5]:
        print(jid, ":", occ)
    return data, targets_nums


def knn_classification(train, val, targets, k):
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


def logreg_classification(train, val, targets):
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


def randfor_classification(train, val, targets):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train[0], train[1])

    res = rf_classifier.predict(train[0])
    accTrain = accuracy_score(train[1], res)
    res = rf_classifier.predict(val[0])
    accVal = accuracy_score(val[1], res)
    return accTrain, accVal


# test file in the same format as the features file
def improve_classification(features_file, test_file, output_folder, classifier):
    return
    train_df = pd.read_csv(features_file)
    test_df = pd.read_csv(test_file)
