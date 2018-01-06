import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def question_c(features_file, output_folder):
    feature_df = pd.read_csv(features_file)
    kf = KFold(n_splits=10)
    grid_features, targets = preprocess_data(feature_df)
    folds_idxs = list(kf.split(grid_features))

#TODO does not return correct result
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
        # get maximum length of feature lists
        maxlen = len(max(data, key=lambda x: len(x)))
        # convert string to numerics
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