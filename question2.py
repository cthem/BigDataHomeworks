import NearestNeighbours as nn
import NearestSubroutes as ns
import MapInGridView as gvp
import JourneyClassification as jcp
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import utils as up
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


def question_c(features_file, test_file, output_folder):
    feature_df = pd.read_csv(features_file)
    kf = KFold(n_splits=10)
    grid_features, targets = jcp.preprocess_data(feature_df)
    folds_idxs = list(kf.split(grid_features))
    trips_array = np.asarray(grid_features)
    targets = np.asarray(targets)

    classifiers = ["knn", "logreg", "randfor"]
    # train/val accuracies
    accuracies = {}
    mean_accuracies = {}
    # classify
    for classifier in classifiers:
        accuracies[classifier] = []
        print()
        print("Testing classifier [%s]" % classifier)
        # train & test each classifier
        # for each fold
        for i, (train_idx, val_idx) in enumerate(folds_idxs):
            print("\tClassifing fold %d/%d" % (i + 1, len(folds_idxs)))
            train = (trips_array[train_idx], targets[train_idx])
            val = (trips_array[val_idx], targets[val_idx])
            if classifier == "knn":
                k = 5
                accTrain, accVal = jcp.knn_classification(train, val, targets, k)
            elif classifier == "logreg":
                accTrain, accVal = jcp.logreg_classification(train, val, targets)
            elif classifier == "randfor":
                accTrain, accVal = jcp.randfor_classification(train, val, targets)
            accuracies[classifier].append((accTrain, accVal))

        # accuracy across all folds
        mean_accuracies[classifier] = [np.mean([x[0] for x in accuracies[classifier]]), \
                                       np.mean([x[1] for x in accuracies[classifier]])]
        titlestr = "%s, overall accuracy train/val: %s" % (classifier, str(mean_accuracies[classifier]))
        up.barchart(list(range(1, 11)), accuracies[classifier], title=titlestr, ylabel="accuracy", legend=["train","val"],
                       save=os.path.join(output_folder, classifier))

    # print mean accuracy per classifier
    for classifier in accuracies:
        print(classifier, "accuracy train/val:", mean_accuracies[classifier])

    print("Improving classification")
    accuracies_impr = {}
    jcp.improve_classification(features_file, test_file, output_folder, classifiers[0])
    # print updated accuracies
    for classifier in accuracies_impr:
        print(classifier, "accuracy:", accuracies_impr[classifier])
