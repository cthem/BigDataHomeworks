import question1.CleanData as clean
import os
import utils
from multiprocessing.pool import Pool, ThreadPool
import threading
import itertools
from dtw import dtw as libdtw



# TODO check time
# TODO check for threads in plots
def question_a1(output_folder, test_file, trips_list, paropts):
    print("WARNING: Reading training file part because test file is not supplied.")
    test_trip_list = trips_list[:5]
    for i,test_trip in enumerate(test_trip_list):
        print("Examining test element %d/%d" % (i+1, len(test_trip_list)))
        outfile_name = os.path.join(output_folder, "nn_%d_" % (i+1))
        # prepare to count time
        millis_start = utils.tic()
        # compute nearest neighbours
        #parallelize = None
        nns_ids_distances = calculate_nns(test_trip, trips_list, paropts = paropts)
        # get time elapsed
        elapsed = utils.tictoc(millis_start)
        # get the whole data of the returned trips, add distances
        neighbour_trips = [n for n in trips_list if n['id'] in [t[0] for t in nns_ids_distances]]
        # visualize
        preprocessing_for_visualization(test_trip, nns_ids_distances, neighbour_trips, outfile_name, elapsed, i)



def calculate_nns(test_trip, trips_list, paropts = None, ret_container = None):

    '''
    :param test_trip: a trip row
    :param trips_list: a list of trip rows to find NNs of test_trip
    :return: returns a list of the 5 nearest neighbours from trips_list, for the test_trip. Format is (trip_id, distance)
    '''

    # parallelization type
    if paropts:
        print("Parallelizing with",paropts)
        partype, numpar = paropts
    else:
        partype, numpar = None, None

    tic = utils.tic()
    # get test coordinate list
    test_lonlat = test_trip["points"]
    # initialize res
    nearest_neighbours = [-1 for _ in trips_list]

    if partype:
        # num threads or processes
        if partype == "processes":
            # K elements at a time - only works for K=1. else, computation is done multiple times
            K = 1
            trips = utils.sublist(trips_list, K)
            # results here
            rres = [[] for _ in trips]
            pool = ThreadPool(processes=numpar)
            tasks = [[] for _ in trips]
            for i,tlist in enumerate(trips):
                #print(len(tlist),tlist)
                async_result = pool.apply_async(calculate_dists, (test_lonlat, tlist))
                tasks[i] = async_result
                #distances = async_result.get()
                #rres[i] = distances
            pool.close()
            pool.join()
            print("Joined.")
            for i in range(len(tasks)):
                rres[i] = tasks[i].get()

            # merge results
            nearest_neighbours = []
            for r in rres:
                nearest_neighbours += r
        elif partype == "threads":
            # create empty containers and divide data per thread
            res = [[] for _ in range(numpar)]
            num_data_per_thread, rem = divmod(len(trips_list), numpar)
            data_per_thread = utils.sublist(trips_list, num_data_per_thread)
            if rem:
                data_per_thread = data_per_thread[:numpar]
                data_per_thread[-1] += trips_list[-rem:]
            # assign data and start the threads
            threads = []
            for i in range(numpar):
                threads.append(threading.Thread(target = calculate_dists, args=(test_lonlat, data_per_thread[i], res[i])))
                threads[i].start()
            # gather and merge results
            rres = []
            for i in range(numpar):
                threads[i].join()
                rres += res[i]
            nearest_neighbours = rres
    else:
        # serial execution
        nearest_neighbours = calculate_dists(test_lonlat, trips_list)

    # sort the list to increasing distance
    nearest_neighbours = sorted(nearest_neighbours, key=lambda k: k[1])
    # return the top 5
    print("Elapsed for parallelization: ",str(partype), numpar, " is:",utils.tictoc(tic))
    print("Neighbours:")
    nearest_neighbours = nearest_neighbours[:5]
    for neigh in nearest_neighbours:
        print(neigh)
    return nearest_neighbours[:5]

def calculate_dists(test_lonlat, trips_list, ret_container = None, paropts = None):

    if ret_container is not None:
        dists = ret_container
    else:
        dists = []
    for i, trip in enumerate(trips_list):
        trip_lonlat = trip["points"]
        # calculate distance
        distance = calculate_dynamic_time_warping(test_lonlat, trip_lonlat, paropts)
        print("Calculated distance: %.2f for trip: %d/%d : %s" % (distance, i+1, len(trips_list), str(trip['id'])))
        dists.append((trip['id'], distance))
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
                #cost = clean.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
                if not (i-1,j-1) in idxs:
                    a=2
                cost = dists[idxs[(i-1,j-1)]]
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        return dtw[-1][-1]
    elif impl == "diy_initial":
        dtw = [[float('Inf') for _ in range(len(latlons2) + 1)] for _ in range(len(latlons1) + 1)]
        dtw[0][0] = 0
        #pool = ThreadPool(processes=10)
        for i in range(1, 1 + len(latlons1)):
            for j in range(1, 1 + len(latlons2)):
                #async_result = pool.apply_async(clean.calculate_lonlat_distance, (latlons1[i - 1], latlons2[j - 1]))
                #cost = async_result.get()
                cost = clean.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
                dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
        #pool.close()
        #pool.join()
        return dtw[-1][-1]

    elif impl == "lib":
        # https://github.com/pierre-rouanet/dtw/blob/master/examples/simple%20example.ipynb
        ret = libdtw(latlons1, latlons2, lambda x,y : clean.calculate_lonlat_distance(x,y))
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
            async_result = pool.apply_async(clean.calculate_lonlat_distance, (p1, p2))
            reslist[i] = async_result.get()
    else:
        for i, (p1,p2) in enumerate(points_list):
            reslist[i] = clean.calculate_lonlat_distance(p1, p2)
    return reslist

def get_trip_from_id(trips_list, nearest_neighbours):
    '''
    This function keeps the 5 nearest neighbours in the format of the trips_list
    :param trips_list: all the trips from the training set
    :param nearest_neighbours: the 5 nearest neighbours
    :return: trips_list containing only the 5 nearest neighbours
    '''
    new_trips_list=[]
    for trip in trips_list:
        for neighbour in nearest_neighbours:
            if trip[0] == neighbour[0]:
                new_trips_list.append(trip)
    return new_trips_list

def preprocessing_for_visualization(test_trip, nns_ids_distances, trips_list, outfile_name, elapsed, i):
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
    points.append([utils.get_lonlat_tuple(test_trip['points'])])
    labels.append("test trip: %d" % i)
    # loop over neighbours
    for j, trip in enumerate(trips_list):
        # add the neighbour points and params
        points.append([utils.get_lonlat_tuple(trip['points'])])
        str = ["neighbour %d" % j, "jid: %s" % trip['jid'], "DWT: %d" % nns_ids_distances[j][1], "Delta-t: %s " % elapsed]
        labels.append("\n".join(str))
    # set all colors to blue
    colors = [['b'] for _ in range(len(nns_ids_distances) + 1)]
    # visualize!
    print("Points:")
    for pts in points:
        print(pts)
    utils.visualize_point_sequences(points, colors, labels, outfile_name)
