import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import os
import random
import utils


# Question 1a
#################

def create_trips_file(input_file, output_file):
    print("Reading training file with pandas ignoring null jids", input_file)
    df = pd.read_csv(input_file)
    print("Read %d lines." % len(df))
    # remove potential NaNs and "null" jids
    df = df[pd.notnull(df['journeyPatternId'])]
    df = df[df['journeyPatternId'] != "null"].reindex()
    timestamps = df.groupby(["vehicleID", "journeyPatternId"])["timestamp"].apply(list)
    lons = df.groupby(["vehicleID", "journeyPatternId"])["longitude"].apply(list)
    lats = df.groupby(["vehicleID", "journeyPatternId"])["latitude"].apply(list)
    df = pd.concat([timestamps, lons, lats], axis=1, ignore_index=False)
    print("Start processing data")
    for index, row in df.iterrows():
        tslist, lonslist, latslist = [], [], []
        tslist = row["timestamp"]
        lonslist = row["longitude"]
        latslist = row["latitude"]
        data_list = []
        for i,timestamp in enumerate(tslist):
            new_list = []
            new_list.append(timestamp)
            new_list.append(lonslist[i])
            new_list.append(latslist[i])
            data_list.append(new_list)
        row["timestamp"] = data_list
    new_df = create_new_dataframe()
    new_df = pd.concat([new_df, df["timestamp"]], axis=1, ignore_index=False)
    new_df.to_csv(output_file)
    df = pd.read_csv(output_file)
    df = df.drop(["vehicleID", "tripId"], axis=1)
    df.index.name = "tripId"
    df.columns = ["journeyId", "points"]
    df = sort_timestamps(df)
    df.to_csv(output_file)
    trips_list = df.to_dict(orient='dict')
    return trips_list, df


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
    print("Total trips deleted due to total distance less than 2km: %d" % len(trips_too_small))
    print("Total trips deleted due to max distance between two points more than 2km: %d" % len(trips_too_big))
    print("Writing",len(df),"cleaned trips to", output_file)
    df.to_csv(output_file)
    trips_list = df.to_dict(orient='dict')
    return trips_list, df


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
    total_points = utils.get_total_points(df)
    random.shuffle(total_points)
    for i in range(num_visualize):
        file_name = output_file_base + str(i) + ".html"
        # get point coordinates lon-lat
        points_lonlat = [utils.idx_to_lonlat(total_points[i])]
        # produce output htmls
        utils.write_group_gml(points_lonlat, file_name)
