import os
import matplotlib.pyplot as plt
import utils
from collections import OrderedDict


def find_min_max_latlon(train_df, output_folder):
    max_lat, max_lon = -1000, -1000
    min_lat, min_lon = 1000, 1000
    llats, llons = [],[]
    for index, row in train_df.iterrows():
        train_points = eval(row["points"])
        lats = [t[2] for t in train_points]
        lons = [t[1] for t in train_points]
        llats.extend(lats)
        llons.extend(lons)
        max_lon = max(lons + [max_lon])
        min_lon = min(lons + [min_lon])
        max_lat = max(lats + [max_lat])
        min_lat = min(lats + [min_lat])
    plt.plot(llons,llats,"b.")
    plt.savefig(os.path.join(output_folder,"data_minmax_grid_extent.png"))
    plt.close()
    return (max_lon,max_lat),(min_lon, min_lat),llats, llons #max_lat, max_lon, min_lat, min_lon


def create_grid(number_of_cells, max_lonlat,min_lonlat,all_lats, all_lons, output_folder):
    total_dists = [ m-n for (m,n) in zip(max_lonlat, min_lonlat)]
    cell_dists = [m/n for (m,n) in zip(total_dists, number_of_cells)]
    cell_lat_dist, cell_lon_dist = cell_dists

    rows = create_grid_lines(number_of_cells[0], min_lonlat[1], cell_lat_dist)
    columns=create_grid_lines(number_of_cells[1], min_lonlat[0], cell_lon_dist)
    cell_names = create_cell_names(number_of_cells)

    visualize_grid(rows,columns,min_lonlat,max_lonlat,output_folder=output_folder)
    visualize_grid(rows,columns,min_lonlat,max_lonlat,output_folder=output_folder, points=(all_lons, all_lats))
    return (rows, columns, cell_names)


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


def map_to_features_bow(data_df, grid, output_file):
    rows, columns, cell_names = grid

    points_header = "points" if "points" in data_df else "Trajectory"

    features = []
    for index,row in data_df.iterrows():
        bow_vector = [0 for cc in cell_names for c in cc]
        train_points = row[points_header]
        train_points = eval(train_points)

        train_lonlats = utils.idx_to_lonlat(train_points, format="tuples")
        for i,lonlat in enumerate(train_lonlats):
            lon = lonlat[0]  # for columns
            lat = lonlat[1]  # for rows
            row_idx = find_index(rows, lat)
            col_idx = find_index(columns, lon)
            linear_idx = row_idx * len(columns) + col_idx
            bow_vector[linear_idx] += 1

        features.append(bow_vector)
    # show stats
    # TODO

    for i,feats in enumerate(features):
        data_df.at[i,points_header] = feats
    if output_file is not None:
        data_df.to_csv(output_file)
    else:
        return features



def map_to_features(data_df, grid, output_file):
    rows, columns, cell_names = grid
    raw_features, timestamps = map_to_features_pointwise(data_df, grid)
    headername = "points" if "points" in data_df else "Trajectory"
    maxlen = -1
    for i,(featlist,ts) in enumerate(zip(raw_features,timestamps)):
        # squeeze duplicate points
        sq_feats = []
        for f,t in zip(featlist,ts):
            if (not sq_feats) or (f not in [ v[-1] for v in sq_feats]): # alternate check
            # if (not sq_feats) or (f != sq_feats[-1][-1]): # alternate check
                sq_feats.append([t, f])
        if len(sq_feats) > maxlen:
            maxlen = len(sq_feats)
        data_df.at[i,headername] = sq_feats
    print("Max squeezed feature length:",maxlen)
    if output_file is not None:
        data_df.to_csv(output_file)
    else:
        return data_df

def map_to_features_pointwise(data_df, grid):
    rows, columns, cell_names = grid
    # measure some statistics
    grid_hist = {}
    total_points = 0
    numcells = (len(rows) + 1) * (len(columns) + 1)
    for cc in cell_names:
        for c in cc:
            grid_hist['C' + str(c)] = 0

    points_header = "points" if "points" in data_df else "Trajectory"
    features, timestamps = [], []
    for index,row in data_df.iterrows():
        train_points = row[points_header]
        train_points = eval(train_points)

        ts = [p[0] for p in train_points]
        timestamps.append(ts)
        train_lonlats = utils.idx_to_lonlat(train_points, format="tuples")
        feature_list = []
        for i,lonlat in enumerate(train_lonlats):
            lon = lonlat[0]  # for columns
            lat = lonlat[1]  # for rows
            row_idx = find_index(rows, lat)
            col_idx = find_index(columns, lon)
            cell_name = 'C'+cell_names[row_idx][col_idx]
            # visualize_grid(rows,columns,None,None,[[lon],[lat]])
            grid_hist[cell_name] += 1
            total_points += 1

            feature_list.append(cell_name)
        features.append(feature_list)
    # show stats
    print()
    print("Grid assignment frequencies of the total of %d points:" % total_points)
    ssum = 0
    for i, name in enumerate(grid_hist):
        print(i,"/", numcells, name, grid_hist[name])
        ssum += grid_hist[name]
    return features, timestamps

def find_index(points_list, point):
    count = 0
    for p in points_list:
        if point < p:
            return count
        count += 1
    return count

# Auxiliary function, visualizes the grid created above
def visualize_grid_gml_print(rows, columns, min_lonlat=None,  max_lonlat=None, points = [], cells = [], output_folder=""):
    # visualize
    min_lat = min(rows + points[1]) if min_lonlat is None else min_lonlat[1]
    min_lon = min(columns + points[0]) if min_lonlat is None else min_lonlat[0]
    max_lat = max(rows + points[1]) if max_lonlat is None else max_lonlat[1]
    max_lon = max(columns + points[0]) if max_lonlat is None else max_lonlat[0]

    fig = plt.figure()
    plt.plot([min_lat, max_lat], [min_lon, min_lon], 'k');
    plt.plot([min_lat, max_lat], [max_lon, max_lon], 'k');
    plt.plot([min_lat, min_lat], [min_lon, max_lon], 'k');
    plt.plot([max_lat, max_lat], [min_lon, max_lon], 'k');

    for x in rows:
        plt.plot([x, x], [min_lon, max_lon], 'r');
    for y in columns:
        plt.plot([min_lat, max_lat], [y, y], 'b');

    if points:
        plt.plot(points[1],points[0],".k")

    for p in cells:
        plt.plot(p[1],p[0],"*g")

    plt.xlabel("lat")
    plt.ylabel("lon")
    plt.savefig(os.path.join(output_folder, "grid.png"), dpi = fig.dpi)
    plt.close()

def visualize_grid(rows, columns, min_lonlat=None,  max_lonlat=None, points = [], cells = [], output_folder=""):
    # visualize
    min_lat = min(rows + points[1]) if min_lonlat is None else min_lonlat[1]
    min_lon = min(columns + points[0]) if min_lonlat is None else min_lonlat[0]
    max_lat = max(rows + points[1]) if max_lonlat is None else max_lonlat[1]
    max_lon = max(columns + points[0]) if max_lonlat is None else max_lonlat[0]

    fig = plt.figure()
    plt.plot([min_lon, min_lon], [min_lat, max_lat], 'k');
    plt.plot([max_lon, max_lon], [min_lat, max_lat], 'k');
    plt.plot([min_lon, max_lon], [min_lat, min_lat], 'k');
    plt.plot([min_lon, max_lon], [max_lat, max_lat], 'k');

    for x in rows:
        plt.plot([min_lon, max_lon], [x, x], 'r');
    for y in columns:
        plt.plot([y, y], [min_lat, max_lat],'b');

    if points:
        plt.plot(points[0],points[1],".k")

    for p in cells:
        plt.plot(p[0],p[1],"*g")

    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(os.path.join(output_folder, "grid.png"), dpi = fig.dpi)
    plt.close()
