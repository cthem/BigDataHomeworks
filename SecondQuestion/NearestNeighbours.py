import FirstQuestion.CleanData as clean
import datetime
import os
import utils


def question_a1(output_folder, file, trips_list):
    test_trip_list = utils.create_test_trip_list(file)
    for i,test_trip in enumerate(test_trip_list):
        print("Examining test element %d/%d" % (i+1, len(test_trip_list)))
        outfile_name = os.path.join(output_folder, "Question2A1", "output_%d" % (i+1))
        millis_start = utils.tic()
        nearest_neighbours = find_nearest_neighbours_for_test_trip(test_trip, trips_list)
        elapsed = utils.tictoc(millis_start)
        trips_list = get_updated_trips_list(trips_list, nearest_neighbours)
        preprocessing_for_visualization(test_trip, nearest_neighbours, trips_list, outfile_name, elapsed, i)


def find_nearest_neighbours_for_test_trip(test_trip, trips_list):
    test_lonlat = utils.create_list_tuples(test_trip)
    nearest_neighbours = []
    #print("knn for ",test_lonlat)
    for trip in trips_list:
        trip_lonlat = utils.create_list_tuples(trip)
        distance = calculate_dynamic_time_warping(test_lonlat, trip_lonlat)
        nearest_neighbours.append((trip[1], distance))
    nearest_neighbours = sorted(nearest_neighbours, key=lambda k: k[1])
    nearest_neighbours = nearest_neighbours[:5]
    return nearest_neighbours


def calculate_dynamic_time_warping(latlons1, latlons2):
    dtw = [ [float('Inf') for _ in range(len(latlons2)+1)] for _ in range(len(latlons1)+1)]
    dtw[0][0] = 0
    for i in range(1,1+len(latlons1)):
        for j in range(1,1+len(latlons2)):
            cost = clean.calculate_lonlat_distance(latlons1[i-1], latlons2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[-1][-1]


def get_updated_trips_list(trips_list, nearest_neighbours):
    """
    gets a trips_list as input, in the format of tripsClean.csv file and returns a list of tuples with lats and lons

    :param trips_list:
    :param nearest_neighbours:
    :return:
    """
    new_trips_list=[]
    for trip in trips_list:
        for neighbour in nearest_neighbours:
            if trip[1] == neighbour[0]:
                new_trips_list.append(trip)
    return new_trips_list


def preprocessing_for_visualization(test_trip, nearest_neighbours, trips_list, outfile_name, elapsed, i):
    points = [[utils.idx_to_lonlat(list(range(len(test_trip[2:]))), test_trip)]]
    labels = ["test trip: %d" % i]
    colors = [['b'] for _ in range(len(nearest_neighbours) + 1)]
    for j, trip in enumerate(trips_list):
        points.append([utils.idx_to_lonlat(list(range(len(trip[2:]))), trip)])
        str = ["neighbour %d" % j, "jid: %s" % trip[1], "DWT: %d" % nearest_neighbours[j][1], "Delta-t: %s " % elapsed]
        labels.append("\n".join(str))
    utils.visualize_paths(points, colors, labels, outfile_name)