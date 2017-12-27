import gmplot
import subprocess
import os
import matplotlib.pyplot as plt
import pylab
from scipy.misc import imread
import datetime
import pickle


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
    with open(output_file, "w") as f:
        for trip in trips_list:
            trip_str = str(trip)
            f.write(trip_str+"\n")

def serialize_trips(output_file, trips_list):
    '''
    Serialize to python pickle
    :param output_file:
    :param trips_list:
    :return:
    '''
    with open(output_file, "wb") as f:
        pickle.dump(trips_list, f)

# Functions used for trips visualization
#######################################
def read_trips_file(filepath):
    '''
    Reads the training file
    :param filepath:
    :return:
    '''
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

def create_list_tuples(trips_list):
    '''
    Create trip tuples from the default format
    :param trips_list:
    :return:
    '''
    trips_list = trips_list[2:]
    new_list_tuples = []
    for item in trips_list:
        new_list_tuples.append((item[1], item[2]))
    return new_list_tuples

def idx_to_lonlat(idx, trip):
    '''
    Get the longitudes and latitudes coresponding to the input points of the input trip
    :param idx: the lonlat indexes to get
    :param trip: the trip to get stuff from
    :return: a (Lo, La) tuple, where Lo,La are lists of floats.
    '''
    if type(idx) != list:
        idx = [idx]
    lats, lons = [], []
    for i in idx:
        lons.append(trip[2+i][1])
        lats.append(trip[2+i][2])
    return (lons, lats)

def write_group_gml(lonlats_tuplelists, outpath, colors=None):
    '''
     Make a color plot a collection of points.
    :param points:  list of lists [L1, L2, ...]. Each Li contains a tuple ([lon1,lon2,...], [lat,lat2,...]), and should be
    associated with a color
    :param outpath: output dir
    :param colors: list of colors characters, one per L_i. If none, defaults to blue.
    :return:
    '''
    max_lat, max_lon = -1,-1
    min_lat, min_lon = 1000, 1000
    for lonlats_tuple in lonlats_tuplelists:
        if not lonlats_tuple[0]:
            continue
        longitude_list = lonlats_tuple[0]
        latitude_list = lonlats_tuple[1]
        if max(longitude_list) > max_lon:
            max_lon = max(longitude_list)
        if max(latitude_list) > max_lat:
            max_lat = max(latitude_list)

        if min(longitude_list) < min_lon:
            min_lon = min(longitude_list)
        if min(latitude_list) < min_lat:
            min_lat = min(latitude_list)

    delta_latlon = max_lat - min_lat , max_lon - min_lon
    center = (min_lat + delta_latlon[0]/2, min_lon + delta_latlon[1]/2)
    zoom = 14
    #print("center:",center,"delta:",delta_latlon,"zoom",zoom)
    gmap = gmplot.GoogleMapPlotter(center[0], center[1], zoom)
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

def visualize_paths(all_pts, colors, labels, file_name):
    '''
    Visualize paths in gmplot.
    :param all_pts:
    :param colors:
    :param labels:
    :param file_name:
    :return:
    '''
    img_names = []
    base_file_name = file_name
    # iterate over collections of points
    print("Producing html and png files...")
    for i, pts in enumerate(all_pts):
        # produce the html of the trip
        file_name = base_file_name + str(i) + ".html"
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
            # col.set_title("...")
    pylab.savefig(file_name + ".plot.jpg", dpi=300)

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
