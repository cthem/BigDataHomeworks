import os
import random
import utils


def visualize_trips(output_folder, df):
    output_file_base = os.path.join(output_folder, "mapplot")
    num_visualize = 5
    total_points = []
    for index, row in df.iterrows():
        points = row["points"]
        points = eval(points)
        total_points.append(points)
        total_points = [t for t in total_points if len(t) > 1]
    random.shuffle(total_points)
    for i in range(num_visualize):
        file_name = output_file_base + str(i) + ".html"
        # get point coordinates lon-lat
        points_lonlat = [idx_to_lonlat(total_points[i])]
        # produce output htmls
        utils.write_group_gml(points_lonlat, file_name)


def idx_to_lonlat(points, idx=None, format="tuple"):
    if idx is None:
        idx = list(range(len(points)))
    if type(idx) != list:
        idx = [idx]
    lats, lons = [], []
    for i in idx:
        lons.append(points[i][1])
        lats.append(points[i][2])
    if format == "tuple":
        return (lons, lats)
    elif format == "tuples":
        return list(zip(lons,lats))
