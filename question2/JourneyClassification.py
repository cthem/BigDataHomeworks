import pickle, os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import utils


def preprocess_data(data):
    targets = [d[1] for d in data]
    data = [d[2] for d in data]

    maxlen = len(max(data, key=lambda x : len(x)))
    data = [[int(d[1:]) for d in dlist] for dlist in data]

    for i, datum in enumerate(data):
        if len(datum) < maxlen:
            data[i] = datum + [3 for _ in range(maxlen - len(datum))]

    # map journey ids to numbers
    num_ids = {}
    targets_nums = []
    for t in targets:
        if t not in num_ids:
            num_ids[t] = len(num_ids)
        targets_nums.append(num_ids[t])
    return data, targets_nums

def question_c(file):
    # get the data ready
    trips_list = get_grid_trips(file)
    kf = KFold(n_splits=10)
    grid_features, targets = preprocess_data(trips_list)

    folds_idxs = list(kf.split(grid_features))
    trips_array = np.asarray(grid_features)
    targets = np.asarray(targets)

    classifiers = ["knn", "logreg", "randfor"]
    accuracies = {}
    # classify
    output_dir = "classification_charts"
    os.makedirs(output_dir, exist_ok=True)
    for classifier in classifiers:
        print("Testing classifier [%s]" % classifier)
        # train & test each classifier
        # for each fold
        accuracies[classifier] = []
        for i, (train_idx, val_idx) in enumerate(folds_idxs):
            print("\tClassifing fold %d/%d" % (i+1, len(folds_idxs)))
            train = (trips_array[train_idx], targets[train_idx])
            val = (trips_array[val_idx], targets[val_idx])
            if classifier == "knn":
                k = 5
                res = knn_classification(train, val, targets, k)
            elif classifier == "logreg":
                res = logreg_classification(train,val)
            elif classifier == "randfor":
                res = randfor_classification(train, val)
            accuracies[classifier].append(res)
        titlestr = "%s, overall accuracy: %2.4f" % (classifier, np.mean(accuracies[classifier]))
        utils.barchart(list(range(1, 11)), accuracies[classifier],title=titlestr, ylabel="accuracy",save=os.path.join(output_dir,classifier))


def get_grid_trips(file):
    with open(file, 'rb') as f:
        trips_list = pickle.load(f)
    return trips_list


def knn_classification(train, val, targets, k):
    # folds contains 10 tuples (training and test sets)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(train[0], train[1])
    res = knn_classifier.predict(val[0])
    return accuracy_score(res, val[1])


def logreg_classification(train, val):
    lr_classifier = LogisticRegression()
    lr_classifier.fit(train[0], train[1])
    res_prob = lr_classifier.predict_proba(val[0])
    # get probabilty argmax for the predicted class
    res = np.argmax(res_prob, axis=1)
    return accuracy_score(res, val[1])


def randfor_classification(train, val):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train[0], train[1])
    res = rf_classifier.predict(val[0])
    return accuracy_score(res, val[1])


def improve_classification():
    pass
