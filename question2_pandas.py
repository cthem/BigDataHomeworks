import question2.NearestNeighboursPandas as nn
import question2.NearestSubroutsPandas as ns
import question2.MapInGridViewPandas as gvp
import question2.JourneyClassificationPandas as jcp
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import utils_pandas as up
import os


def question_a1(output_folder, clean_file, test_file, paropts):
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(clean_file)
    for index, row in test_df.iterrows():
        print("Examining test element %d" % (index + 1))
        outfile_name = os.path.join(output_folder, "nn_%d_" % (index + 1))
        # prepare to count time
        millis_start = up.tic()
        # compute nearest neighbours
        test_points = row["points"]
        test_points = eval(test_points)
        nns_ids_distances = nn.calculate_nns(test_points, train_df, paropts=paropts)
        # get time elapsed
        elapsed = up.tictoc(millis_start)
        # visualize
        nn.preprocessing_for_visualization(test_points, nns_ids_distances, outfile_name, elapsed, index)


def question_a2(output_folder, test_file, train_file):
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    for index, row in test_df.iterrows():
        print("Extracting subroutes for test trip %d/%d" % (index + 1, len(test_df.index)))
        file_name = os.path.join(output_folder, "subroutes_%d_" % (index + 1))
        test_points = row["points"]
        test_points = eval(test_points)
        max_subseqs = ns.find_similar_subroutes_per_test_trip(test_points, train_df)
        ns.preprocessing_for_visualisation(test_points, max_subseqs, file_name, index)


def question_b(train_file, number_of_cells, output_folder):
    # specify files
    output_file = os.path.join(output_folder, "tripFeatures.csv")
    train_df = pd.read_csv(train_file)
    max_lat, max_lon, min_lat, min_lon = gvp.find_min_max_latlong(train_df)
    rows, columns, cell_names = gvp.create_grid(number_of_cells, max_lat, max_lon, min_lat, min_lon, output_folder=output_folder)
    gvp.replace_points(train_df, rows, columns, cell_names, output_file)
    return output_file


def question_c(features_file, output_folder):
    feature_df = pd.read_csv(features_file)
    kf = KFold(n_splits=10)
    grid_features, targets = jcp.preprocess_data(feature_df)
    folds_idxs = list(kf.split(grid_features))
    folds_idxs = list(kf.split(grid_features))
    trips_array = np.asarray(grid_features)
    targets = np.asarray(targets)

    classifiers = ["knn", "logreg", "randfor"]
    accuracies = {}
    # classify
    for classifier in classifiers:
        print("Testing classifier [%s]" % classifier)
        # train & test each classifier
        # for each fold
        accuracies[classifier] = []
        for i, (train_idx, val_idx) in enumerate(folds_idxs):
            print("\tClassifing fold %d/%d" % (i + 1, len(folds_idxs)))
            train = (trips_array[train_idx], targets[train_idx])
            val = (trips_array[val_idx], targets[val_idx])
            if classifier == "knn":
                k = 5
                res = jcp.knn_classification(train, val, targets, k)
            elif classifier == "logreg":
                res = jcp.logreg_classification(train, val, targets)
            elif classifier == "randfor":
                res = jcp.randfor_classification(train, val, targets)
            accuracies[classifier].append(res)
        titlestr = "%s, overall accuracy: %2.4f" % (classifier, np.mean(accuracies[classifier]))
        up.barchart(list(range(1, 11)), accuracies[classifier], title=titlestr, ylabel="accuracy",
                       save=os.path.join(output_folder, classifier))