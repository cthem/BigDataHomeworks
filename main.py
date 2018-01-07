import sys, os, pkgutil
import question1 as qp1
import question2 as qp2
import random


def question_1(input_file, output_file, output_file_clean, maps_folder):
    # Question 1
    print(">>> Running question 1a - parsing the training data")
    trips_list, df = qp1.create_trips_file(input_file, output_file)
    print(">>> Running question 1b - cleaning the training data")
    trips_list, df = qp1.filter_trips(output_file_clean, df)
    print(">>> Running question 1c - visualizing the training data")
    qp1.visualize_trips(maps_folder, df)
    print("Finished question1")


def question_2(train_file, test_files, test_file, output_folder, maps_folder, class_folder, paropts):
    # Question 2
    print(">>> Running question 2a1 - Nearest neighbours computation")
    qp2.question_a1(maps_folder, train_file, test_files[0], paropts)
    print(">>> Running question 2a2 - Nearest subroutes computation")
    # qp2.question_a2(maps_folder, test_files[1], train_file)
    print(">>> Running question 2b - Cell grid quantization")
    cellgrid = (10, 10)
    print("Using cell grid:", cellgrid)
    features_file = qp2.question_b(train_file, cellgrid, output_folder)
    print(">>> Running question 2c - Classification")
    qp2.question_c(features_file, test_file, class_folder)


def check_dependencies():
    deps = ['numpy', 'scipy', 'pandas']
    for dep in deps:
        res = pkgutil.find_loader(dep)
        if res is None:
            print("The python3 package [%s] is required to run." % dep)
            exit(1)
    if not os.path.exists('rasterize.js'):
        print("The rasterize.js tool from the phantomjs framework is required to run.")
        exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s inputfolder outputfolder" % sys.argv[0])
        exit(1)
    check_dependencies()
    print("Running %s with arguments: %s" % (os.path.basename(sys.argv[0]), sys.argv))
    input_folder  = os.path.abspath(sys.argv[1])
    output_folder = os.path.abspath(sys.argv[2])

    rand_seed = 123123
    random.seed(rand_seed)

    # paropts = ("processes", 10)
    # paropts = ("threads", 10)
    paropts = None

    # question 1
    ############

    # prepare files
    train_file = os.path.join(input_folder, "train_set_dev.csv")
    output_file = os.path.join(output_folder, "trips.csv")
    output_file_clean = os.path.join(output_folder, "trips_clean.csv")
    maps_folder = os.path.join(output_folder, "gmplots")
    class_folder = os.path.join(output_folder, "classification_charts")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(maps_folder, exist_ok=True)
    os.makedirs(class_folder, exist_ok=True)

    # run
    question_1(train_file, output_file, output_file_clean, maps_folder)

    # question 2
    ############

    # prepare files
    test_file = [os.path.join(input_folder, "test_set.csv")]
    test_files = [os.path.join(input_folder, "test_set_a%d.csv" % t) for t in [1,2]]

    # run
    question_2(output_file_clean, test_files, test_file, output_folder, maps_folder, class_folder, paropts)

