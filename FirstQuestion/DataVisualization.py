import random
import os
import utils


def question_1c(output_folder, trips_list):
    '''
    Runs the 3rd subquestion
    :param output_folder:
    :param trips_list:
    :return:
    '''
    output_folder = os.path.join(output_folder, "Question1C")
    output_file_base = os.path.join(output_folder, "mymap")
    trips_visualization(output_file_base, trips_list, 5)


def trips_visualization(base_file_name, trips_list, num_of_trips):
    '''
    Exports trips in a map
    :param base_file_name:
    :param trips_list:
    :param num_of_trips:
    :return:
    '''
    non_null_trips = get_trip_list_not_null_jid(trips_list)
    for i in range(num_of_trips):
        trip = non_null_trips[i]
        file_name = base_file_name + str(i) + ".html"
        points = [utils.idx_to_lonlat(list(range(len(trip[2:]))), trip)]
        colors = ['b' for _ in range(len(points) + 1)]
        utils.write_group_gml(points, file_name, colors)


def get_trip_list_not_null_jid(trips_list):
    '''
    Removes trips with null journey_id
    :param trips_list:
    :return:
    '''
    non_null_trips = [t for t in trips_list if t[1] != 'null']
    random.shuffle(non_null_trips)
    return non_null_trips