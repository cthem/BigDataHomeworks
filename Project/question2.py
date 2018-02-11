import NearestNeighbours as nn
import NearestSubroutes as ns
import MapInGridView as gvp
import JourneyClassification as jcp
import pandas as pd
import utils
import os
import pickle


def question_a1(output_folder, clean_file, test_file, paropts, k):
    test_df = pd.read_csv(test_file, delimiter="\n")
    train_df = pd.read_csv(clean_file)
    print("Extracting %d nearest neighbours out of %d cleaned train data, for each test trip" % (k,len(train_df)))
    print("Using parallelization options:", paropts)
    for index, row in test_df.iterrows():
        print("Examining test element %d / %d" % (index + 1, len(test_df)))
        outfile_name = os.path.join(output_folder, "nn_%d_" % (index + 1))
        # prepare to count time
        millis_start = utils.tic()
        # compute nearest neighbours
        test_points = eval(row["Trajectory"])
        nns_ids_distances = nn.calculate_nns(test_points, train_df, paropts=paropts)
        # get time elapsed
        elapsed = utils.tictoc(millis_start)
        # visualize
        nn.visualize_nns(test_points, nns_ids_distances, outfile_name, elapsed, index)


def question_a2(output_folder, test_file, train_file, conseq_lcss, k, paropts, verbosity, unique_jids = True):
    lcss_type = "consequtive" if conseq_lcss else "non-consequtive"
    print("Extracting %d %s subroutes for each test trip" % (k, lcss_type))
    test_df = pd.read_csv(test_file, delimiter="\n")
    train_df = pd.read_csv(train_file)
    for index, row in test_df.iterrows():
        print("Extracting subroutes for test trip %d/%d" % (index + 1, len(test_df)))
        file_name = os.path.join(output_folder, "subroutes_%d_" % (index + 1))
        test_points = eval(row["Trajectory"])
        max_subseqs = ns.find_similar_subroutes_per_test_trip(test_points, train_df, k, paropts, conseq_lcss, verbosity, unique_jids)
        ns.preprocessing_for_visualisation(test_points, max_subseqs, file_name, index)


def question_b(train_file, number_of_cells, output_folder):
    # specify files
    grid_file = os.path.join(output_folder,"grid.pickle")
    feature_file = os.path.join(output_folder, "tripFeatures.csv")
    # read data and make the grid
    train_df = pd.read_csv(train_file)
    max_lonlat, min_lonlat, all_lats, all_lons = gvp.find_min_max_latlon(train_df, output_folder)
    grid = gvp.create_grid(number_of_cells, max_lonlat, min_lonlat, all_lats, all_lons, output_folder=output_folder)
    # save grid and transform data
    with open(grid_file, "wb") as f:
        pickle.dump(grid, f)
    # gvp.map_to_features(train_df, grid, feature_file)
    gvp.map_to_features_bow(train_df, grid, feature_file)
    return feature_file, grid_file


def question_c(clean_file, features_file, grid_file, test_file, output_folder, seed, classif_file, num_folds):
    df_features = pd.read_csv(features_file)
    features, jid_mapping, targets = jcp.preprocess_train_data(df_features, seed)
    classifiers = ["knn", "logreg", "randfor"]
    # classifiers = ["logreg"]
    mean_accuracies = jcp.train(features, targets, num_folds, classifiers, output_folder)

    # print mean accuracy per classifier
    print()
    for classifier in mean_accuracies:
        print(classifier, "accuracy train/val:", mean_accuracies[classifier])

    # select the logistic regression algorithm to beat the benchmark
    impr_classifier_name = "logreg"
    baseline_accuracy = mean_accuracies[impr_classifier_name][-1]

    best_accuracy, best_classifier, best_technique = -1, None, None
    print()
    print("Improving classification for classifier", impr_classifier_name)
    mean_accuracies = jcp.improve_classification(clean_file, grid_file, features_file, num_folds, output_folder, impr_classifier_name, seed)

    print()
    print("Performance comparison:")
    # print updated accuracies
    for technique in mean_accuracies:
        accuracy, classifier = mean_accuracies[technique]
        # get validation accuracy
        accuracy = accuracy['logreg'][-1]
        print(impr_classifier_name, ", technique",technique,", validation accuracy :", accuracy,
              "change over baseline: %2.2f%%" % ((accuracy-baseline_accuracy) / baseline_accuracy * 100))
        if best_accuracy < accuracy:
            best_classifier = classifier
            best_technique = technique

    # run best classifier on the test data
    # read and featurify data
    print("Reading test data...")
    test_data_df = pd.read_csv(test_file,delimiter=";")
    with open(grid_file,"rb") as f:
        grid = pickle.load(f)
    print("Transforming test data to features...")
    test_features = gvp.map_to_features_bow(test_data_df, grid, None)
    # test_features = jcp.preprocess_test_data(test_features)
    print("Running test on",impr_classifier_name,"-",best_technique)
    jcp.test(best_classifier, impr_classifier_name, best_technique, test_file, grid_file, jid_mapping, classif_file)
    print("Done!")


