import os
import matplotlib.pyplot as plt
import utils_pandas as up


def find_min_max_latlong(train_df):
    max_lat, max_lon = -1000, -1000
    min_lat, min_lon = 1000, 1000
    for index, row in train_df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        max_lon = max([t[1] for t in train_points] + [max_lon])
        min_lon = min([t[1] for t in train_points] + [min_lon])
        max_lat = max([t[2] for t in train_points] + [max_lat])
        min_lat = min([t[2] for t in train_points] + [min_lat])
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


def replace_points(train_df, rows, columns, cell_names, output_file):
    for index,row in train_df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        timestamps = []
        for point in train_points:
            timestamps.append(point[0])
        train_lonlats = up.idx_to_lonlat(train_points, format="tuples")
        new_points = []
        for i,lonlat in enumerate(train_lonlats):
            new_point = []
            lon = lonlat[0]  # for columns
            lat = lonlat[1]  # for rows
            lat_idx = find_index(rows, lat)
            lon_idx = find_index(columns, lon)
            cell_name = 'C'+cell_names[lat_idx][lon_idx]
            new_point.append(timestamps[i])
            new_point.append(cell_name)
            new_points.append(new_point)
        train_df.at[index, "points"] = new_points
    train_df.to_csv(output_file)


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
    plt.close()
