import gmplot
import subprocess
import os
import matplotlib.pyplot as plt
import pylab
from scipy.misc import imread
import datetime
import pickle
import json



def tic():
    '''
    Get timestamp
    :return:
    '''
    return datetime.datetime.now()

def tictoc(previous_tic):
    '''
    Get duration from previous timestamp
    :param previous_tic:
    :return:
    '''
    # get duration in usec
    diff = tic() - previous_tic
    # get durations of datetime.now() members
    days = diff.days
    secs = diff.seconds
    usec = diff.microseconds
    # aggregate to larger time units
    msec, usec = divmod(usec, 1000)
    sec_, msec = divmod(msec, 1000)
    secs += sec_
    mins, sec = divmod(secs, 60)
    hrs, mins = divmod(mins,60)
    msg = "%02d:%02d:%02d:%03d" % (hrs, mins, secs, msec)
    if days:
        msg = "%d days! " % days + msg
    return msg

def write_trips(output_file, trips_list):
    '''
    Self explanatory
    :param output_file:
    :param trips_list:
    :return:
    '''
    # write list primitives
    if type(trips_list[0]) == list:
        with open(output_file, "w") as f:
            for trip in trips_list:
                # make csv
                header = [str(t) for t in trip[:2]]
                coords = [str(t) for coord_tuple in trip[2:] for t in coord_tuple]
                trip_str = ",".join(header + coords)
                # write it
                f.write(trip_str+"\n")
    # write list of objects
    if type(trips_list[0]) == dict:
        # use json
        with open(output_file, "w") as f:
            json.dump(trips_list, f)


def serialize_trips(output_file, trips_list):
    '''
    Serialize to python pickle
    :param output_file:
    :param trips_list:
    :return:
    '''
    with open(output_file, "wb") as f:
        pickle.dump(trips_list, f)

def read_trips(filepath):
    '''
    Read a trips file, csv or pickle
    :param filepath:
    :return:
    '''
    if filepath.endswith(".pickle"):
        with open(filepath, "rb") as f:
            trips = pickle.load(f)
    else:
        # read the lines
        trips=[]
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                elems = line.split(",")
                obj={}
                obj["id"] = elems[0]
                obj["jid"] = elems[1]
                elems = elems[2:]
                timestamps, points = [], []
                for i in range(0, len(elems), 3):
                    vals = elems[i:i+3]
                    timestamps.append(float(vals[0]))
                    points.append((float(vals[1]),float(vals[2])))
                obj["timestamps"] = timestamps
                obj["points"] = points
                trips.append(obj)
    return trips

# Functions used for trips visualization
#######################################
def read_train_set(filepath):
    '''
    Reads the training file
    :param filepath:
    :return:
    '''
    print("Reading file", filepath)
    line_objects = []
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            line_contents = line.strip().split(',')
            obj = {}
            obj["jid"] = line_contents[0]
            obj["vid"] = line_contents[1]
            obj["ts"] = line_contents[2]
            obj["lon"] = line_contents[3]
            obj["lat"] = line_contents[4]
            line_objects.append(obj)
    return line_objects


def idx_to_lonlat(trip, idx = None, format = "tuple"):
    '''
    Get the longitudes and latitudes coresponding to the input points of the input trip
    :param idx: the lonlat indexes to get. if None, get all
    :param trip: the trip to get stuff from
    :param format: tuple: ([],[]), tuples: [(),(),()...]
    :return: a (Lo, La) tuple, where Lo,La are lists of floats.
    '''
    if idx is None:
        idx = list(range(len(trip[2:])))
    if type(idx) != list:
        idx = [idx]
    lats, lons = [], []
    for i in idx:
        lons.append(trip[2+i][1])
        lats.append(trip[2+i][2])
    if format == "tuple":
        return (lons, lats)
    elif format == "tuples":
        return list(zip(lons,lats))


def write_group_gml(lonlats_tuplelists, outpath, colors=None):
    '''
     Make a color plot a collection of points.
    :param points:  list of tuples [T1, T2, ...]. Each Ti is a tuple ([lon1,lon2,...], [lat,lat2,...]), and should be
    associated with a color
    :param outpath: output dir
    :param colors: list of colors characters, one per L_i. If none, defaults to blue.
    :return:
    '''
    flattened = [l for tup in lonlats_tuplelists for l in tup]
    maxs = [max(t) for t in flattened ]
    mins = [min(t) for t in flattened ]
    max_lonlat = max(maxs[0::2]), max(maxs[1::2])
    min_lonlat = min(mins[0::2]), min(mins[1::2])
    delta_lonlat = [mx-mn for (mx,mn) in zip(max_lonlat, min_lonlat)]
    center_lonlat = [min_lonlat[i] + delta_lonlat[i] for i in range(2)]
    zoom = 14
    # print("points:",lonlats_tuplelists)
    # print("center:",center_lonlat,"delta:",delta_lonlat,"zoom",zoom)
    gmap = gmplot.GoogleMapPlotter(center_lonlat[1], center_lonlat[0], zoom)
    if colors is None:
        colors = ['b' for _ in lonlats_tuplelists]
    for idx,pts in enumerate(lonlats_tuplelists):
        # expects input in lats, longs
        gmap.plot(pts[1], pts[0], colors[idx], edge_width=5)
    print("Writing plot", outpath)
    gmap.draw(outpath)


def html_to_png(html_path, png_path):
    '''
    Create an image from an html file
    :param html_path:
    :param png_path:
    :return:
    '''
    cmd = ["phantomjs","rasterize.js", html_path, png_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as ex:
        print("Failed to run command %s" % " ".join(cmd))
        print(ex)
        exit(1)

def get_lonlat_tuple(points):
    '''
    :param points: [(lon1,lat1), (lon2,lat2),...]
    :return: ([lon1,lon2,...],[lat1,lat2,...])
    '''
    return ([l[0] for l in points], [l[1] for l in points])

def visualize_point_sequences(all_pts, colors, labels, file_name):
    '''
    Generic geocoordinate multi-plot visualization function with gmplot.
    :param all_pts: list of geocoordinate points
    :param colors: list of list of colors
    :param labels:
    :param file_name:
    :return:

     all_pts is a list [P1,P2,...]. Each Pi corresponds to a figure.
         Pi is a list of [K1,K2...]. Each Ki corresponds to a line segment.
         Ki is a tuple (Lons,Lats), Lons ( resp., Lats) is a list of longitudes (resp., latitudes).
     colors is a list [c1, c2]. Each ci corresponts to a list of colors, corresponding to the line segments Ki.
     labels is a list of strings, one per figure

    '''
    img_names = []
    base_file_name = file_name
    # iterate over collections of points
    print("Producing html and png files...")
    for i, pts in enumerate(all_pts):
        # produce the html of the trip
        file_name = base_file_name + str(i+1) + ".html"
        write_group_gml(pts, file_name, colors[i])
        image_filename = file_name + ".jpg"
        # keep track of the image filenames to read them later
        img_names.append(image_filename)
        # produce the image
        html_to_png(file_name, image_filename)
        # delete the html
        os.remove(file_name)

    # read and display images in a collection of pylab plots
    print("Producing plots...")
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,15))
    plt.grid(False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    imgidx = 0
    for row in ax:
        for col in row:
            # remove axes' value ticks
            col.set_xticks([])
            col.set_yticks([])
            # set labels and display the image
            col.set_xlabel(labels[imgidx])
            img = imread(img_names[imgidx])
            col.imshow(img)
            # delete the image
            os.remove(img_names[imgidx])
            imgidx += 1
            if imgidx >= len(img_names):
                break
    pylab.savefig(base_file_name + ".nnplot.jpg", dpi=300)
    plt.close()


def barchart(xvalues, yvalues, title="",ylabel="", save=None):
    '''
    Function to create and show/save a pylab barchart
    :param xvalues:
    :param yvalues:
    :param title:
    :param ylabel:
    :param save:
    :return:
    '''
    barwidth = 0.2
    fig = plt.figure(figsize=(12.0, 5.0))
    ax = plt.gca()
    plt.xticks([i  for i in xvalues], xvalues)
    ax.grid()
    ax.set_xlabel("fold")

    ax.set_ylim([0.0, 1.1])
    xend = xvalues[-1] + 0.1
    ax.set_xlim([0, xend+1])
    ax.set_title(title)

    ax.bar([i for i in xvalues], yvalues, width=barwidth, color="r", label="accuracy")
    ax.plot([0, xend], [1, 1], "k--")
    ax.legend(loc='upper right')
    ax.set_ylabel(ylabel)
    if save is None:
        plt.show()
    else:
        plt.savefig(save, dpi=fig.dpi)
    plt.close()


if __name__ == '__main__':
   pts = [([25,26.4,25.1],[26,26.1,26.8])]
   write_group_gml(pts,"filefile.html")
   print("Done.")
