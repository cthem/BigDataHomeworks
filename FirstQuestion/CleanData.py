from math import sin, cos, sqrt, atan2, radians
import os
import utils


def question_1b(output_folder, trips_list):
    '''
    Runs the second subquestion
    :param output_folder:
    :param trips_list:
    :return:
    '''
    print("Total num of trips in trips.csv file: %d" % len(trips_list))
    trips_list = filter_trips(trips_list)
    print("Final num of trips to be writtern in tripsClean.csv: %d" % len(trips_list))
    output_file = os.path.join(output_folder, "tripsClean.csv")
    pickle_file = os.path.join(output_folder, "tripsClean.pickle")
    utils.write_trips_to_file(output_file, trips_list)
    utils.write_trips_using_pickle(pickle_file, trips_list)
    return trips_list


def filter_trips(trips_list):
    '''
    Removes too long or too short trips
    :param trips_list:
    :return:
    '''
    trips_too_small, trips_too_big = [], []
    for trip in trips_list:
        total_dist = caluclate_total_distance_per_trip(trip)
        # too small journey check
        if total_dist < 2:
            trips_too_small.append(trip)
            trips_list.remove(trip)
        max_dist = calculate_max_dist(trip)
        if max_dist > 2:
            trips_too_big.append(trip)
            trips_list.remove(trip)
    print("Total trips deleted due to total distance less than 2km: %d" % len(trips_too_small))
    print("Total trips deleted due to max distance between two points more than 2km: %d" % len(trips_too_big))
    return trips_list


def caluclate_total_distance_per_trip(trip):
    '''
    Calculates the total distance of a trip
    :param trip:
    :return:
    '''
    total_distance = 0
    lonlatlist = trip[2:]
    for i in range(len(lonlatlist)-1):
        dist =  calculate_lonlat_distance(lonlatlist[i][1:], lonlatlist[i+1][1:])
        total_distance += dist
    return total_distance


def calculate_lonlat_distance(point1, point2):
    """
    Uses the haversine formula
    """
    # print(point1, point2)
    point1 = [radians(point1[0]), radians(point1[1])]
    point2 = [radians(point2[0]), radians(point2[1])]
    R = 6371.0
    a = sin((point2[1]-point1[1])/2)**2 + cos(point1[1]) * cos(point2[1]) * sin((point2[0]-point1[0])/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d


def calculate_max_dist(trip):
    '''
    Calculates the max distance between two points
    :param trip:
    :return:
    '''
    lonlatlist = trip[2:]
    max_dist=2
    for i in range(len(lonlatlist) - 1):
        dist = calculate_lonlat_distance(lonlatlist[i][1:], lonlatlist[i + 1][1:])
        if(dist > max_dist):
            max_dist = dist
    return max_dist
