import os
import random
import UtilsPandas as up


def visualize_trips(output_folder, df):
    output_file_base = os.path.join(output_folder, "mapplot")
    num_visualize = 5
    total_points = up.get_total_points(df)
    random.shuffle(total_points)
    for i in range(num_visualize):
        file_name = output_file_base + str(i) + ".html"
        # get point coordinates lon-lat
        points_lonlat = [up.idx_to_lonlat(total_points[i])]
        # produce output htmls
        up.write_group_gml(points_lonlat, file_name)



