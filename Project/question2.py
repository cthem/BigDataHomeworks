import NearestNeighbours as nn
import NearestSubroutes as ns
import MapInGridView as gvp
import JourneyClassification as jcp
import pandas as pd
import utils
import os


def question_a1(output_folder, clean_file, test_file, paropts, k):
    print("Extracting %d nearest neighbours for each test trip" % k)
    test_df = pd.read_csv(test_file, delimiter="\n")
    train_df = pd.read_csv(clean_file)
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
        nn.preprocessing_for_visualization(test_points, nns_ids_distances, outfile_name, elapsed, index)


def question_a2(output_folder, test_file, train_file, conseq_lcss, k, paropts):
    print("Extracting %d subroutes for each test trip" % k)
    test_df = pd.read_csv(test_file, delimiter="\n")
    train_df = pd.read_csv(train_file)
    for index, row in test_df.iterrows():
        print("Extracting subroutes for test trip %d/%d" % (index + 1, len(test_df)))
        file_name = os.path.join(output_folder, "subroutes_%d_" % (index + 1))
        test_points = eval(row["Trajectory"])
        max_subseqs = ns.find_similar_subroutes_per_test_trip(test_points, train_df, k, paropts, conseq_lcss)
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
    df_features = pd.read_csv(features_file)
    features, targets = jcp.preprocess_data(df_features)
    num_folds = 5
    classifiers = ["knn", "logreg", "randfor"]
    # classifiers = ["logreg"]
    mean_accuracies = jcp.classify(features, targets, num_folds, classifiers, output_folder)

    # print mean accuracy per classifier
    for classifier in mean_accuracies:
        print(classifier, "accuracy train/val:", mean_accuracies[classifier])
    impr_classifier = "logreg"
    print()
    print("Improving classification for classifier", impr_classifier)
    mean_accuracies = jcp.improve_classification(features_file, num_folds, output_folder, impr_classifier)
    # print updated accuracies
    for technique in mean_accuracies:
        print(impr_classifier, ", technique",technique,", accuracy train/val:", mean_accuracies[technique])

    # run best classifier on test data, TODO

