import random
import os
import utils


def question_1c(output_folder, trips_list):
    '''
    Runs the 3rd subquestion. Visualize 5 random different journeys.
    :param output_folder:
    :param trips_list:
    :return:
    '''
    output_file_base = os.path.join(output_folder, "mapplot")
    num_visualize = 5
    # Apply some restrictions to visualized candidates
    print("Will restrict visualization to a) non-null journey ids and b) having at least 2 points")
    trips_list = [t for t in trips_list if t[1] != "null" and len(t[2:]) > 1]
    # shuffle them up
    random.shuffle(trips_list)
    for i in range(num_visualize):
        trip = trips_list[i]
        file_name = output_file_base + str(i) + ".html"
        # get point coordinates lon-lat
        num_coords = len(trip[2:])
        points_lonlat = [utils.idx_to_lonlat(list(range(num_coords)), trip)]
        # produce output htmls
        utils.write_group_gml(points_lonlat, file_name)
