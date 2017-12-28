import question1.CleanData as clean
import os
import utils


def question_a1(output_folder, test_file, trips_list):
    print("WARNING: Reading training file part because test file is not supplied.")
    test_trip_list = trips_list[:5]
    for i,test_trip in enumerate(test_trip_list):
        print("Examining test element %d/%d" % (i+1, len(test_trip_list)))
        outfile_name = os.path.join(output_folder, "nn_%d_" % (i+1))
        # prepare to count time
        millis_start = utils.tic()
        # compute nearest neighbours
        nearest_neighbours = calcluate_nearest_neighbours(test_trip, trips_list)
        # get time elapsed
        elapsed = utils.tictoc(millis_start)
        # get the whole data of the returned trips
        neighbour_trips = get_trip_from_id(trips_list, nearest_neighbours)
        # visualize
        preprocessing_for_visualization(test_trip, nearest_neighbours, neighbour_trips, outfile_name, elapsed, i)


def calcluate_nearest_neighbours(test_trip, trips_list):

    '''
    :param test_trip: a trip row
    :param trips_list: a list of trip rows to find NNs of test_trip
    :return: returns a list of the 5 nearest neighbours from trips_list, for the test_trip. Format is (trip_id, distance)
    '''

    # get test coordinate list
    test_lonlat = utils.idx_to_lonlat(test_trip, format="tuples")
    nearest_neighbours = []
    for trip in trips_list:
        # get candidate coordinate list
        trip_lonlat = utils.idx_to_lonlat(trip, format="tuples")
        # calcluate distance
        distance = calculate_dynamic_time_warping(test_lonlat, trip_lonlat)
        nearest_neighbours.append((trip[0], distance))
    # sort the list to increasing distance
    nearest_neighbours = sorted(nearest_neighbours, key=lambda k: k[1])
    # return the top 5
    return nearest_neighbours[:5]


def calculate_dynamic_time_warping(latlons1, latlons2):
    '''
    Calculate the DTW of two points, used in order find the nearest neighbours
    :param latlons1:
    :param latlons2:
    :return:
    '''
    dtw = [ [float('Inf') for _ in range(len(latlons2)+1)] for _ in range(len(latlons1)+1)]
    dtw[0][0] = 0
    for i in range(1,1+len(latlons1)):
        for j in range(1,1+len(latlons2)):
            cost = clean.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[-1][-1]


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


def preprocessing_for_visualization(test_trip, nearest_neighbours, trips_list, outfile_name, elapsed, i):
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
    points.append([utils.idx_to_lonlat(test_trip)])
    labels.append("test trip: %d" % i)
    # loop over neighbours
    for j, trip in enumerate(trips_list):
        # add the neighbour points and params
        points.append([utils.idx_to_lonlat(trip)])
        str = ["neighbour %d" % j, "jid: %s" % trip[1], "DWT: %d" % nearest_neighbours[j][1], "Delta-t: %s " % elapsed]
        labels.append("\n".join(str))
    # set all colors to blue
    colors = [['b'] for _ in range(len(nearest_neighbours) + 1)]
    # visualize!
    utils.visualize_point_sequences(points, colors, labels, outfile_name)