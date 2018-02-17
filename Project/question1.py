import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import os
import random
import utils


# Question 1a
#################

def create_trips_file(input_file, output_file):
    df = pd.read_csv(input_file)
    print("Reading training file", input_file, " - will ignore null journey ids")
    print("Parsing csv...")
    print("Removing nulls and NaNs ...")
    df = df[pd.notnull(df['journeyPatternId'])]
    df = df[df['journeyPatternId'] != "null"].reindex()
    vids = {}
    trips = []
    print("Transforming data to the required format")
    for index, row in df.iterrows():
        vid = row["vehicleID"]
        jid = row["journeyPatternId"]
        data = [row["timestamp"], row["longitude"], row["latitude"]]
        if vid not in vids:
            vids[vid] = []
        if not vids[vid]:
            vids[vid] = [jid, [data]]
        else:
            if vids[vid][0] == row["journeyPatternId"]:
                vids[vid][1].append(data)
            else:
                trips.append(vids[vid])
                vids[vid] = []
                vids[vid] = [jid, [data]]
    points = []
    jids = []
    tripids = []
    print("Producing new dataframe")
    for i, trip in enumerate(trips):
        points.append(trip[1])
        jids.append(trip[0])
        tripids.append(i)
    df1 = pd.DataFrame({'tripId': tripids})
    df2 = pd.DataFrame({'journeyId': jids})
    df3 = pd.DataFrame({'points': points})
    df = pd.concat([df1, df2, df3], axis=1, ignore_index=False)
    df.to_csv(output_file)
    # trips_list = df.to_dict(orient='dict')
    print("Done transforming data!")
    return df


def create_new_dataframe():
    columns = []
    columns.append("tripId")
    new_df = pd.DataFrame(columns=columns)
    return new_df


def sort_timestamps(df):
    for index, row in df.iterrows():
        points = row["points"]
        points = eval(points)
        points.sort(key=lambda x: x[0])
        df.at[index, "points"] = points
    return df


# Question 1b
#############

def filter_trips(output_file, df):
    print("Filtering %d trips..." % len(df))
    trips_too_small, trips_too_big = [], []
    for index, row in df.iterrows():
        jounreyId = row["journeyId"]
        points = row["points"]
        total_dist = calculate_total_distance_per_trip(points)
        if total_dist < 2000:
            trips_too_small.append(jounreyId)
            df.drop(index, inplace=True)
            continue
        max_dist = calculate_max_dist(points)
        if max_dist > 2000:
            trips_too_big.append(jounreyId)
            df.drop(index, inplace=True)
            continue
    print("Deleted %d due to having total distance less than 2km" % len(trips_too_small))
    print("Deleted %d trips due having to max distance between two points more than 2km" % len(trips_too_big))
    print("Writing",len(df),"cleaned trips to", output_file)
    df.to_csv(output_file)
   # trips_list = df.to_dict(orient='dict')
    return  df


def calculate_total_distance_per_trip(points):
    total_distance = 0
    for i in range(len(points) - 1):
        dist = calculate_lonlat_distance(points[i][1:], points[i + 1][1:])
        total_distance += dist
    return total_distance


def calculate_max_dist(points):
    max_dist = 2000
    for i in range(len(points) - 1):
        dist = calculate_lonlat_distance(points[i][1:], points[i + 1][1:])
        if (dist > max_dist):
            max_dist = dist
    return max_dist


def calculate_lonlat_distance(point1, point2):
    """
    Calculates the distance in meters between two points (lon, lat) using the haversine formula.
    """
    # to rad
    point1 = [radians(point1[0]), radians(point1[1])]
    point2 = [radians(point2[0]), radians(point2[1])]
    delta_lat = point2[1] - point1[1]
    delta_lon = point2[0] - point1[0]
    earth_radius = 6371.0
    a = sin(delta_lat/2)**2 + cos(point1[1]) * cos(point2[1]) * sin(delta_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = earth_radius * c
    # return to meters
    return d * 1000

# Question 1c
###############

def visualize_trips(output_folder, df):
    output_file_base = os.path.join(output_folder, "mapplot")
    num_visualize = 5
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    # total_points = utils.get_total_points(df)
    print("Randomly selected %d trips to visualize, with indexes" % num_visualize, idxs[:num_visualize])
    for i in range(num_visualize):
        idx = idxs[i]
        total_pts = utils.get_total_points(df[idx : idx + 1])[0]
        file_name = output_file_base + str(i) + ".html"
        # get point coordinates lon-lat
        # points_lonlat = [utils.idx_to_lonlat(total_points[i])]
        points_lonlat = [utils.idx_to_lonlat(total_pts)]
        # produce output htmls
        utils.write_group_gml(points_lonlat, file_name)
        # produce output jpg
        utils.html_to_png(file_name, file_name + ".jpg")
