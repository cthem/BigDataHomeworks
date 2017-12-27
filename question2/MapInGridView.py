import matplotlib.pyplot as plt
import utils
import os


def subquestion_b(trips_list, number_of_cells, output_folder):
    # specify files
    output_file = os.path.join(output_folder, "tripFeatures.csv")
    pickle_file = os.path.join(output_folder, "tripFeatures.pickle")
    # use a given data collection to judge min/max data points
    max_lat, max_lon, min_lat, min_lon = find_min_max_latlong(trips_list)
    rows,columns,cell_names = create_grid(number_of_cells,max_lat,max_lon,min_lat,min_lon, output_folder=output_folder)
    replace_points(trips_list, rows, columns, cell_names, output_file, pickle_file)
    return pickle_file


def find_min_max_latlong(trips_list):
    max_lat=trips_list[0][2][2]
    max_lon=trips_list[0][2][1]
    min_lat=trips_list[0][2][2]
    min_lon=trips_list[0][2][1]
    for trip in trips_list:
        if trip[2][1]>max_lon:
            max_lon=trip[2][1]
        if trip[2][1]<min_lon:
            min_lon=trip[2][1]
        if trip[2][2]>max_lat:
            max_lat=trip[2][2]
        if trip[2][2]<min_lat:
            min_lat=trip[2][2]
    return max_lat, max_lon, min_lat, min_lon


def create_grid(number_of_cells, max_lat,max_lon,min_lat,min_lon, output_folder):
    cell_lat_dist, cell_lon_dist = get_distance_per_cell(number_of_cells, min_lat, min_lon, max_lat, max_lon)
    rows = create_grid_lines(number_of_cells[0], min_lat, cell_lat_dist)
    columns=create_grid_lines(number_of_cells[1], min_lon, cell_lon_dist)
    cell_names = create_cell_names(number_of_cells)

    visualize_grid(rows,columns,min_lat,min_lon,max_lat,max_lon, output_folder=output_folder)
    return rows,columns,cell_names


def get_distance_per_cell(number_of_cells, min_lat, min_lon, max_lat, max_lon):
    total_lat_dist = max_lat - min_lat
    total_lon_dist = max_lon - min_lon
    cell_lat_dist = total_lat_dist / number_of_cells[0]
    cell_lon_dist = total_lon_dist / number_of_cells[1]
    return cell_lat_dist, cell_lon_dist


def create_grid_lines(number_of_cells, min_point, dist):
    lines_list = []
    new_min = min_point
    for i in range(number_of_cells -1):
        new_point = new_min + dist
        lines_list.append(new_point)
        new_min = new_point
    return lines_list


def create_cell_names(number_of_cells):
    cell_names = []
    for i in range(number_of_cells[0]):
        cell_names.append([])
        for j in range(number_of_cells[1]):
            cell_names[-1].append(str(i) + str(j))
    return cell_names


def replace_points(trips_list, rows, columns, cell_names, output_file, pickle_file):
    new_trips_list = []
    for trip in trips_list:
        trip_lonlat = utils.idx_to_lonlat(trip, format = "tuples")
        new_trips_list.append([])
        new_trips_list[-1].append(trip[0])
        new_trips_list[-1].append(trip[1])
        new_names=[]
        for lonlat in trip_lonlat:
            lon = lonlat[0]  # for columns
            lat = lonlat[1]  # for rows
            lat_idx = find_index(rows, lat)
            lon_idx = find_index(columns, lon)
            cell_name = 'C'+cell_names[lat_idx][lon_idx]
            new_names.append(cell_name)
            #print("Point ",lonlat,"mapped to",cell_name)
        new_trips_list[-1].append(new_names)
    utils.write_trips(output_file, new_trips_list)
    utils.serialize_trips(pickle_file, new_trips_list)

def find_index(points_list, point):
    count = 0
    for p in points_list:
        if point < p:
            return count
        count += 1
    return count


# Auxiliary function, visualizes the grid created above
def visualize_grid(rows, columns, min_lat=None, min_lon=None, max_lat=None, max_lon=None, points = [], cells = [], output_folder=""):
    # visualize
    min_lat = min(rows) if min_lat is None else min_lat
    min_lon = min(columns) if min_lon is None else min_lon
    max_lat = max(rows) if max_lat is None else max_lat
    max_lon = max(columns) if max_lon is None else max_lon

    fig = plt.figure()
    plt.plot([min_lat, max_lat], [min_lon, min_lon], 'k');
    plt.plot([min_lat, max_lat], [max_lon, max_lon], 'k');
    plt.plot([min_lat, min_lat], [min_lon, max_lon], 'k');
    plt.plot([max_lat, max_lat], [min_lon, max_lon], 'k');
    for x in rows:
        plt.plot([x, x], [min_lon, max_lon], 'r');
    for y in columns:
        plt.plot([min_lat, max_lat], [y, y], 'b');
    for p in points:
        plt.plot(p[1],p[0],".k")
    for p in cells:
        plt.plot(p[1],p[0],"*g")
    plt.xlabel("lat")
    plt.ylabel("lon")
    plt.savefig(os.path.join(output_folder, "grid.png"), dpi = fig.dpi)
