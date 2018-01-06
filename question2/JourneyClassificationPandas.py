import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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
    for i,d in enumerate(data):
        print(targets[i], d)
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
    res = knn_classifier.predict(val[0])
    print(classification_report(res, val[1], target_names=str(targets)))
    return accuracy_score(res, val[1])


def logreg_classification(train, val, targets):
    lr_classifier = LogisticRegression()
    lr_classifier.fit(train[0], train[1])
    res_prob = lr_classifier.predict_proba(val[0])
    # get probabilty argmax for the predicted class
    res = np.argmax(res_prob, axis=1)
    print(classification_report(res, val[1], target_names=str(targets)))
    return accuracy_score(res, val[1])


def randfor_classification(train, val, targets):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train[0], train[1])
    res = rf_classifier.predict(val[0])
    print(classification_report(res, val[1], target_names=str(targets)))
    return accuracy_score(res, val[1])


# test file in the same format as the features file
def improve_classification(features_file, test_file, output_folder, classifier):
    train_df = pd.read_csv(features_file)
    test_df = pd.read_csv(test_file)
