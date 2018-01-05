from math import sin, cos, sqrt, atan2, radians


def filter_trips_pandas(output_file, df):
    trips_too_small, trips_too_big = [], []
    for index, row in df.iterrows():
        jounreyId = row["journeyId"]
        points = row["points"]
        total_dist = calculate_total_distance_per_trip_pandas(points)
        if total_dist < 2000:
            trips_too_small.append(jounreyId)
            df.drop(index, inplace=True)
            continue
        max_dist = calculate_max_dist_pandas(points)
        if max_dist > 2000:
            trips_too_big.append(jounreyId)
            df.drop(index, inplace=True)
            continue
    print("Total trips deleted due to total distance less than 2km: %d" % len(trips_too_small))
    print("Total trips deleted due to max distance between two points more than 2km: %d" % len(trips_too_big))
    df.to_csv(output_file)
    trips_list = df.to_dict(orient='dict')
    return trips_list, df


def calculate_total_distance_per_trip_pandas(points):
    total_distance = 0
    for i in range(len(points) - 1):
        dist = calculate_lonlat_distance(points[i][1:], points[i + 1][1:])
        total_distance += dist
    return total_distance


def calculate_max_dist_pandas(points):
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
