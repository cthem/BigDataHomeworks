import numpy as np
import pandas as pd
import sklearn
import pickle
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import MapInGridView as gvp
from sklearn.metrics import classification_report
import utils
import os


def preprocess_train_data(feature_df, seed):
    print("Preprocessing training data")
    targets = []
    data = []
    # shuffle data
    feature_df = feature_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    for index, row in feature_df.iterrows():
        targets.append(row["journeyId"])
        train_points = row["points"]
        train_points = eval(train_points)
        data.append(train_points)

    # convert journey ids to numbers
    num_ids = {}
    targets_nums = []
    for t in targets:
        if t not in num_ids:
            num_ids[t] = len(num_ids)
        targets_nums.append(num_ids[t])

    # count instances per jid
    jids = [j for j in num_ids]
    counts = [targets_nums.count(num_ids[j]) for j in jids]
    jids_sorted = sorted(zip(jids,counts), key = lambda x : x[1], reverse = True)
    print("10 Jids with most data:")
    for i in range(10):
        print(i,":",jids_sorted[i])

    print("Done preprocessing data")
    return data, num_ids, targets_nums


def train(features, targets, num_folds, classifiers, output_folder, seed=None, filename_tag ="", classifier_obj = None):
    kf = KFold(n_splits = num_folds, random_state=seed)
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
        classif_start = utils.tic()
        accuracies[classifier] = []
        print("\nTesting classifier [%s]" % classifier)
        # train & test each classifier
        # for each fold
        for i, (train_idx, val_idx) in enumerate(folds_idxs):
            print("\tClassifying fold %d/%d" % (i + 1, len(folds_idxs)), end=" ")
            train = (trips_array[train_idx], targets[train_idx])
            val = (trips_array[val_idx], targets[val_idx])
            if classifier == "knn":
                k = 5
                accTrain, accVal = knn_classification(train, val, k,  classifier_obj=classifier_obj)
            elif classifier == "logreg":
                accTrain, accVal = logreg_classification(train, val, classifier_obj=classifier_obj)
            elif classifier == "randfor":
                accTrain, accVal = randfor_classification(train, val,  seed, classifier_obj=classifier_obj)
            accuracies[classifier].append((accTrain, accVal))
            print("- accuracies train/val:",accuracies[classifier][-1])
        elapsed = utils.tictoc(classif_start)
        print("Done in:", elapsed)

        # accuracy across all folds
        mean_accuracies[classifier] = [np.mean([x[0] for x in accuracies[classifier]]), \
                                       np.mean([x[1] for x in accuracies[classifier]])]
        titlestr = "%s, overall accuracy train/val: %s" % (classifier, str(mean_accuracies[classifier]))
        chart_filename = os.path.join(output_folder, classifier + "_" + filename_tag + "_chart")
        utils.barchart(list(range(1, num_folds + 1)), accuracies[classifier], title=titlestr, ylabel="accuracy", legend=["train","val"],
                    save=chart_filename)

    return mean_accuracies


def knn_classification(train, val, k, classifier_obj = None):
    # folds contains 10 tuples (training and test sets)
    if not classifier_obj:
        classifier_obj = KNeighborsClassifier(n_neighbors=k)
    classifier_obj.fit(train[0], train[1])
    # get training and test accuracy
    res = classifier_obj.predict(train[0])
    accTrain = accuracy_score(train[1], res)
    res = classifier_obj.predict(val[0])
    accVal = accuracy_score(val[1], res)
    return accTrain, accVal
    #print(classification_report(val[1], res))


def logreg_classification(train, val, classifier_obj = None):

    if classifier_obj is None:
        classifier_obj = LogisticRegression()
    classifier_obj.fit(train[0], train[1])

    res_prob = classifier_obj.predict_proba(train[0])
    res = np.argmax(res_prob, axis=1)
    accTrain =  accuracy_score(train[1], res)

    # get prediction by probabilty argmax for the predicted class
    res_prob = classifier_obj.predict_proba(val[0])
    res = np.argmax(res_prob, axis=1)
    accVal =  accuracy_score(val[1], res)

    return accTrain, accVal


def randfor_classification(train, val, seed=None, classifier_obj = None):
    if classifier_obj is None:
        classifier_obj = RandomForestClassifier(random_state=seed)
    classifier_obj.fit(train[0], train[1])

    res = classifier_obj.predict(train[0])
    accTrain = accuracy_score(train[1], res)
    res = classifier_obj.predict(val[0])
    accVal = accuracy_score(val[1], res)
    return accTrain, accVal


def improve_randfor(baseline_accuracy, bow_features_file, num_folds, output_folder, classifier, seed):
    best_accuracy, best_classifier, best_technique = -1, None, None
    features = pd.read_csv(bow_features_file)
    features, jid_mapping, targets = preprocess_train_data(features, seed)
    # try various techniques for improvement
    mean_accuracies = {}

    # test different estimators
    estimators = [5,20,25]
    for estimator in estimators:
        tag = "estimator_%s" % estimator
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = RandomForestClassifier(n_estimators=estimator, random_state=seed)
        mean_acc = train(features, targets, num_folds, classifier, output_folder, seed=seed, filename_tag=tag,
                         classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)

        # test different max_features
    max_features = ["sqrt", "log2", None]
    for max in max_features:
        tag = "max_feature_strategy_%s" % max
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = RandomForestClassifier(max_features=max, random_state=seed)
        mean_acc = train(features, targets, num_folds, classifier, output_folder, seed=seed, filename_tag=tag,
                            classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)
    # test max depth strategy
    max_depth = [10,50,100]
    for max in max_depth:
        tag = "max_depth_strategy_%s" % max
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = RandomForestClassifier(max_depth=max, random_state=seed)
        mean_acc = train(features, targets, num_folds, classifier, output_folder, seed=seed, filename_tag=tag,
                            classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)
    min_samples = [4,6,10]
    for min in min_samples:
        tag = "min_samples_strategy_%s" % min
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = RandomForestClassifier(min_samples_split=min, random_state=seed)
        mean_acc = train(features, targets, num_folds, classifier, output_folder, seed=seed, filename_tag=tag,
                         classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)

    print()
    print("Performance comparison:")
    best_accuracy = -1
    for technique in mean_accuracies:
        accuracy, classifier_obj = mean_accuracies[technique]
        # get validation accuracy
        accuracy = accuracy['randfor'][-1]
        print('randfor', ", technique", technique, ", validation accuracy :", accuracy,
              "change over baseline: %2.2f%%" % ((accuracy - baseline_accuracy) / baseline_accuracy * 100))
        if best_accuracy < accuracy:
            best_classifier = classifier_obj
            best_technique = technique
            best_accuracy=accuracy
    return best_classifier, best_technique, best_accuracy


def improve_logreg(clean_trips_file, grid_file, bow_features_file, num_folds, output_folder, classifier, seed):

    print()
    # print("Reading features:",features_file)
    features = pd.read_csv(bow_features_file)
    features, jid_mapping, targets = preprocess_train_data(features, seed)
    # try various techniques for improvement
    mean_accuracies = {}

    # try feature normalization
    norms = ["l2","l1","max"]
    for norm in norms:
        tag = "norm_%s" % norm
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = LogisticRegression()
        proc_features = sklearn.preprocessing.normalize(features, norm = norm, copy=False)
        mean_acc = train(proc_features, targets, num_folds, classifier, output_folder, tag, classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)

    # try feature scaling
    tag = "scaling"
    print("\nTrying strategy: %s" % tag, end='')
    classifier_obj = LogisticRegression()
    features_scaled = sklearn.preprocessing.scale(features)
    mean_acc = train(features_scaled, targets, num_folds, classifier, output_folder, tag, classifier_obj=classifier_obj)
    mean_accuracies[tag] = (mean_acc, classifier_obj)


    # try various regularization strengths
    Cs = [0.2, 0.5 , 1.5, 2.0]
    for c in Cs:
        tag = "C_%1.3f" % c
        print("\nTrying strategy: %s" % tag, end='')
        classifier_obj = LogisticRegression(C=c)
        mean_acc = train(features, targets, num_folds, classifier, output_folder, tag, classifier_obj=classifier_obj)
        mean_accuracies[tag] = (mean_acc, classifier_obj)

    with open(grid_file,"rb") as fg:
        grid =pickle.load(fg)
    features = pd.read_csv(clean_trips_file)

    # vlad_features = gvp.map_to_features_vlad(features, grid, None)
    # classifier_obj = LogisticRegression()
    # tag = "vlad"
    # mean_acc = train(vlad_features, targets, num_folds, classifier, output_folder, tag, classifier_obj=classifier_obj)
    # mean_accuracies[tag] = (mean_acc, classifier_obj)

    return mean_accuracies


def preprocess_test_data(feature_df):
    print("Preprocessing test data")
    data, ids = [], []
    for index, row in feature_df.iterrows():
        ids.append(row["Test_Trip_ID"])
        data.append([r[1] for r in row["Trajectory"]])

    # get maximum length of feature lists
    maxlen = len(max(data, key=lambda x: len(x)))
    print("Padding test features to a max length of", maxlen)
    # keep the numeric part of the text features
    data = [[int(d[1:]) for d in dlist] for dlist in data]
    # pad to the maximum length
    for i, datum in enumerate(data):
        if len(datum) < maxlen:
            data[i] = datum + [3 for _ in range(maxlen - len(datum))]

    return data


def test(classifier_obj, best_technique, test_file, grid_file, jid_mapping, output_file):

    # read and featurify data
    print("Reading test data...")
    test_data_df = pd.read_csv(test_file,delimiter=";")
    with open(grid_file,"rb") as f:
        grid = pickle.load(f)
    print("Transforming test data to features...")
    features = gvp.map_to_features_bow(test_data_df, grid, None)
    # mean subtraction

    jids = [j for j in jid_mapping]
    numeric_ids = [jid_mapping[j] for j in jids]
    if best_technique.startswith("norm"):
        norm = float(best_technique.split("_")[-1])
        features = sklearn.preprocessing.normalize(features, norm = norm, copy=False, classifier_obj=classifier_obj)

    # classify
    # res_prob = logreg_object.predict_proba(features)
    # res = np.argmax(res_prob, axis=1)
    res = classifier_obj.predict(features)
    # write result
    print("Writing results on",output_file)
    with open(output_file, "w") as f:
        lines = []
        lines.append("Test_Trip_ID\tPredicted_JourneyPatternID")
        for i,r in enumerate(res):
            jid = jids[numeric_ids.index(r)]
            lines.append("%d\t%s" % (i,jid))
        f.writelines(lines)