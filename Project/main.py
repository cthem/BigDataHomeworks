import sys, os, pkgutil
from os.path import join, abspath
import question1 as qp1
import question2 as qp2
import random
import numpy as np


def question_1(opts):
    # Question 1
    print("\nRunning question #1")
    print("=====================")
    print("\n>>> Running question 1a - parsing the training data")
    trips_list, df = qp1.create_trips_file(opts["trainfile"], opts["tripsfile"])
    print("\n>>> Running question 1b - cleaning the training data")
    trips_list, df = qp1.filter_trips(opts["cleanfile"], df)
    print("\n>>> Running question 1c - visualizing the training data")
    qp1.visualize_trips(opts["mapsdir"], df)
    print("\nFinished question1!")


def question_2(opts):
    print("\nRunning question #2")
    print("=====================")
    # Question 2
    print("\n>>> Running question 2a1 - Nearest neighbours computation")
    qp2.question_a1(opts["mapsdir"], opts["cleanfile"], opts["testfiles"][0], opts["paropts"], opts["k"], options["unique_subroute_jids"])

    print("\n>>> Running question 2a2 - Nearest subroutes computation")
    qp2.question_a2(opts["mapsdir"], opts["testfiles"][1], opts["cleanfile"],\
                    opts["conseq_lcss"], opts["k"], opts["paropts"], opts["verbosity"], options["unique_subroute_jids"])

    print("\n>>> Running question 2b - Cell grid quantization")
    cellgrid = opts["grid"]
    print("Using cell grid with dimensions", cellgrid)
    features_file, grid_file = qp2.question_b(opts["cleanfile"], cellgrid, opts["outdir"])

    print("\n>>> Running question 2c - Classification")
    qp2.question_c(opts["cleanfile"],features_file, grid_file, opts["testfiles"][2], opts["classifdir"], opts["seed"], opts["classiffile"], opts["folds"])


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
        print("Usage: %s inputfolder outputfolder" % os.path.basename(sys.argv[0]))
        exit(1)
    check_dependencies()
    print("Running %s with arguments: %s" % (os.path.basename(sys.argv[0]), sys.argv))

    # create an options dict object to pass around
    options = {}
    options["verbosity"] = True
    options["indir"] = abspath(sys.argv[1])
    options["outdir"] = abspath(sys.argv[2])
    options["conseq_lcss"] = True

    options["seed"] = 123123
    np.random.seed(options["seed"])
    random.seed(options["seed"])

    # paropts = ("processes", 10)
    # paropts = ("threads", 10)
    paropts = None
    options["paropts"] = paropts

    # question 1
    ############

    # prepare files and parameters
    options["trainfile"]  = join(options["indir"], "train_set.csv")
    options["tripsfile"]  = join(options["outdir"], "trips.csv")
    options["cleanfile"]  = join(options["outdir"], "trips_clean.csv")
    options["mapsdir"]    = join(options["outdir"], "gmplots")
    options["classifdir"] = join(options["outdir"], "classification_charts")
    options["classiffile"] = join(options["outdir"],"​testSet_JourneyPatternIDs.csv")
    options["folds"] = 10
    options["grid"] = (5,5)

    os.makedirs(options["outdir"], exist_ok=True)
    os.makedirs(options["mapsdir"], exist_ok=True)
    os.makedirs(options["classifdir"], exist_ok=True)

    # run
    question_1(options)

    # question 2
    ############

    # prepare files and parameters
    test_files = [join(options["indir"], "test_set_a%d.csv" % t) for t in [1,2]] + ["test_set.csv"]
    test_files = [join(options["indir"],t) for t in test_files]
    options["testfiles"] = test_files
    options["unique_subroute_jids"] = True
    options["k"] = 5

    # run
    question_2(options)

