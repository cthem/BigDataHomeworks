import sys, os, pkgutil
import FirstQuestion.Preprocessing as prep
import FirstQuestion.CleanData as clean
import FirstQuestion.DataVisualization as visual

import SecondQuestion.NearestNeighbours as nn
import SecondQuestion.NearestSubroutes as ns
import SecondQuestion.MapInGridView as gv
import SecondQuestion.JourneyClassification as jc


def main(output_folder, input_file):
    # Prepare files and folders
    output_file, test_file1, test_file2 = create_files_folders(output_folder, input_file)
    trips_list = question_1(input_file, output_file, output_folder)
    question_2(output_folder, test_file1, test_file2, trips_list)


def create_files_folders(output_folder, input_file):
    # question 1
    output_file = "%s/trips.csv" % output_folder
    os.makedirs("%s/Question1C" % output_folder, exist_ok=True)
    # question 2
    # test_file1 = "test_set_a1.csv"
    # test_file2 = "test_set_a2.csv"
    # TODO correct the test files
    test_file1 = input_file
    test_file2 = input_file
    os.makedirs("%s/Question2A1" % output_folder, exist_ok=True)
    os.makedirs("%s/Question2A2" % output_folder, exist_ok=True)

    return output_file, test_file1, test_file2


def question_1(input_file, output_file, output_folder):
    # Question 1
    print(">>> Running question 1a - parsing the training data")
    trips_list = prep.question_1a(input_file, output_file)
    print(">>> Running question 1b - cleaning the training data")
    trips_list = clean.question_1b(output_folder, trips_list)
    print(">>> Running question 1b - visualizing the training data")
    visual.question_1c(output_folder, trips_list)
    return trips_list


def question_2(output_folder, test_file1, test_file2, trips_list):
    # Question 2
    print(">>> Running question 2a1 - Nearest neighbours computation")
    nn.question_a1(output_folder, test_file1, trips_list)
    print(">>> Running question 2a2 - Nearest subroutes computation")
    ns.question_a2(output_folder, test_file2, trips_list)
    print(">>> Running question 2b - Cell grid quantization")
    cellgrid = (4, 3)
    print("Using cell grid:", cellgrid)
    features_file = gv.subquestion_b(trips_list, cellgrid, output_folder)
    print(">>> Running question 2c - Classification")
    jc.question_c(features_file)


def check_dependencies():
    deps = ['numpy', 'scipy']
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
        print("Usage: %s outputfolder inputcsv" % sys.argv[0])
        exit(1)
    check_dependencies()
    args = sys.argv[1:]
    print("Running %s with arguments: %s" % (sys.argv[0], args))
    main(*args)
