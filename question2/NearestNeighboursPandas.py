import pandas as pd
import os
import UtilsPandas as up
from multiprocessing.pool import Pool, ThreadPool
import threading
from dtw import dtw as libdtw
import question1.CleanDataPandas as cdp


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
        nns_ids_distances = calculate_nns(test_points, train_df, paropts=paropts)
        # get time elapsed
        elapsed = up.tictoc(millis_start)
        # visualize
        preprocessing_for_visualization(test_points, train_df, nns_ids_distances, outfile_name, elapsed, index)


def calculate_nns(test_points, train_df, paropts=None):
    # parallelization type
    if paropts:
        print("Parallelizing with", paropts)
        partype, numpar = paropts
    else:
        partype, numpar = None, None

    tic = up.tic()
    test_lonlat = up.idx_to_lonlat(test_points, format="tuples")
    nearest_neighbours = [-1 for _ in range(len(train_df.index))]

    if partype:
        # num threads or processes
        if partype == "processes":
            nearest_neighbours = run_with_processes(numpar, test_lonlat, train_df, nearest_neighbours)
        elif partype == "threads":
            nearest_neighbours = run_with_threads(numpar, test_lonlat, train_df)
    else:
        # serial execution
        nearest_neighbours = calculate_dists(test_lonlat, train_df)
    # sort the list to increasing distance
    nearest_neighbours = sorted(nearest_neighbours, key=lambda k: k[1])
    # return the top 5
    print("Elapsed for parallelization: ", str(partype), numpar, " is:", up.tictoc(tic))
    print("Neighbours:")
    nearest_neighbours = nearest_neighbours[:5]
    for neigh in nearest_neighbours:
        print(neigh)
    return nearest_neighbours[:5]


def run_with_processes(numpar, test_lonlat, train_df, nearest_neighbours):
    pool = ThreadPool(processes=numpar)
    # for results
    rres = [[] for _ in range(len(train_df.index))]
    tasks = [[] for _ in range(len(train_df.index))]
    for itrain, rtrain in train_df:
        train_points = rtrain["points"]
        train_points = eval(train_points)
        async_result = pool.apply_async(calculate_dists, (test_lonlat, train_points))
        tasks[itrain] = async_result
    pool.close()
    pool.join()
    print("Joined.")
    for i in range(len(tasks)):
        rres[i] = tasks[i].get()
        # merge results
        nearest_neighbours = []
        for r in rres:
            nearest_neighbours += r
    return nearest_neighbours


#TODO cardu boh8eiaaaa
def run_with_threads(numpar, test_lonlat, train_df, nearest_neighbours):
    # create empty containers and divide data per thread
    res = [[] for _ in range(numpar)]
    num_data_per_thread, rem = divmod(len(train_df.index), numpar)
    data_per_thread = up.sublist(up.get_total_points(train_df), num_data_per_thread)
    if rem:
        data_per_thread = data_per_thread[:numpar]
        # data_per_thread[-1] += trips_list[-rem:]
    # assign data and start the threads
    threads = []
    for i in range(numpar):
        threads.append(threading.Thread(target=calculate_dists, args=(test_lonlat, data_per_thread[i], res[i])))
        threads[i].start()
    # gather and merge results
    rres = []
    for i in range(numpar):
        threads[i].join()
        rres += res[i]
    nearest_neighbours = rres
    return nearest_neighbours


def calculate_dists(test_lonlat, train_df, ret_container = None, paropts = None):
    if ret_container is not None:
        dists = ret_container
    else:
        dists = []
    for index, row in train_df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        trip_lonlat = up.idx_to_lonlat(train_points,format="tuples")
        # calculate distance
        distance = calculate_dynamic_time_warping(test_lonlat, trip_lonlat, paropts)
        print("Calculated distance: %.2f for trip: %d/%d : %s" % (distance, index+1, len(train_df.index), str(row["journeyId"])))
        dists.append((int(row["tripId"]), distance))
    return dists


def calculate_dynamic_time_warping(latlons1, latlons2, paropts = None, impl = "diy_initial"):
    '''
    Calculate the DTW of two points, used in order find the nearest neighbours
    :param latlons1:
    :param latlons2:
    :return:
    '''
    if impl == "diy":
        # for identical inputs, return an ad hoc 0
        if latlons1 == latlons2:
            return 0
        dtw = [ [float('Inf') for _ in range(len(latlons2)+1)] for _ in range(len(latlons1)+1)]
        dtw[0][0] = 0

        pairs = []
        idxs = {}
        for i in range(len(latlons1)):
            for j in range(len(latlons2)):
                pairs.append((latlons1[i],latlons2[j]))
                idxs[(i,j)] = len(pairs)-1

        dists = compute_dists(pairs, paropts)

        for i in range(1,1+len(latlons1)):
            for j in range(1,1+len(latlons2)):
                # cost = clean.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
                if not (i-1,j-1) in idxs:
                    a=2
                cost = dists[idxs[(i-1,j-1)]]
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        return dtw[-1][-1]
    elif impl == "diy_initial":
        dtw = [[float('Inf') for _ in range(len(latlons2) + 1)] for _ in range(len(latlons1) + 1)]
        dtw[0][0] = 0
        for i in range(1, 1 + len(latlons1)):
            for j in range(1, 1 + len(latlons2)):
                cost = cdp.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
                dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
        return dtw[-1][-1]

    elif impl == "lib":
        # https://github.com/pierre-rouanet/dtw/blob/master/examples/simple%20example.ipynb
        ret = libdtw(latlons1, latlons2, lambda x,y : cdp.calculate_lonlat_distance(x,y))
        return ret[0]


def compute_dists(points_list, paropts):
    reslist = [-1 for _ in points_list]
    if paropts:
        partype, numpar = paropts
    else:
        partype = None
    if partype == 'process':
        pool = ThreadPool(processes=numpar)
        for i, (p1,p2) in enumerate(points_list):
            async_result = pool.apply_async(cdp.calculate_lonlat_distance, (p1, p2))
            reslist[i] = async_result.get()
    else:
        for i, (p1,p2) in enumerate(points_list):
            reslist[i] = cdp.calculate_lonlat_distance(p1, p2)
    return reslist



def preprocessing_for_visualization(test_points, nns_ids_distances, train_df, outfile_name, elapsed, i):
    '''
    :param test_trip: the given test trip from the test file
    :param nearest_neighbours: the 5 nearest neighbours of this test trip, in format id, distance
    :param trips_list: all the trips from the training set file
    :param outfile_name: name for the output file
    :param elapsed: time needed in order to process the nearest neighbours
    :param i: the index of the test trip
    :return: calls the visualize_paths function in order to export the maps with the test trip and the 5 nearest neighbours
    '''
    # create lists of stuff to show, to contain both test and neighbour data
    points, labels= [], []
    # add tuple of longitudes, latitudes and visualization params for the test trip
    # list of lists because multi-color gml visualizer expects such
    points.append([up.get_lonlat_tuple(test_points)])
    labels.append("test trip: %d" % i)
    # loop over neighbours
    total_pts = up.get_total_points(train_df)
    for index, row in train_df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        points.append(up.get_lonlat_tuple(train_points))
        str = ["neighbour %d" % index, "jid: %s" % int(row["tripId"]), "DWT: %d" % nns_ids_distances[index][1], "Delta-t: %s " % elapsed]
        labels.append("\n".join(str))
    # set all colors to blue
    colors = [['b'] for _ in range(len(nns_ids_distances) + 1)]
    # visualize!
    print("Points:")
    for pts in points:
        print(pts)
    up.visualize_point_sequences(points, colors, labels, outfile_name)