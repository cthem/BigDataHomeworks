import gmplot
import subprocess
import os
import matplotlib.pyplot as plt
import pylab
from scipy.misc import imread
import datetime
import FirstQuestion.Preprocessing as prep
import pickle


def tic():
    return datetime.datetime.now()


def tictoc(previous_tic):
    # get duration in usec
    diff = tic() - previous_tic
    days = diff.days
    secs = diff.seconds
    usec = diff.microseconds
    msec, usec = divmod(usec, 1000)
    sec_, msec = divmod(msec, 1000)
    secs += sec_
    mins, sec = divmod(secs, 60)
    hrs, mins = divmod(mins,60)
    msg = "%02d:%02d:%02d:%03d" % (hrs, mins, secs, msec)
    if days:
        msg = "%d days! " % days + msg
    return msg


def write_trips_to_file(output_file, trips_list):
    with open(output_file, "w") as f:
        for trip in trips_list:
            trip_str = str(trip)
            f.write(trip_str+"\n")


def write_trips_using_pickle(output_file, trips_list):
    with open(output_file, "wb") as f:
        pickle.dump(trips_list, f)


def create_test_trip_list(file):
    '''
    Test files processing
    '''
    # test_file = path_to_files + file
    # with open(test_file, "r") as f:
    #     test_trip_list=[]
    #     for line in f:
    #         test_trip_list.append(list(line))
    # TODO read from test file
    print("\tNote: Will use the training file from question 1a for now, limited to 5 elements")
    tl =  prep.question_1a(file, "test_clean_trips.csv")
    return tl[:5]


# Functions used for trips visualization

def create_list_tuples(trips_list):
    '''
    get lats and lots as list of tuples
    '''
    trips_list = trips_list[2:]
    new_list_tuples = []
    for item in trips_list:
        new_list_tuples.append((item[1], item[2]))
    return new_list_tuples


def idx_to_lonlat(idx, trip):
    if type(idx) != list:
        idx = [idx]
    lats = []
    lons = []
    for i in idx:
        lats.append(trip[2+i][2])
        lons.append(trip[2+i][1])
    return (lons, lats)


def write_group_gml(points, outpath, colors):
    mean_lat, mean_lon = 0, 0
    llen = 0
    for pts in points:
        if not pts[0]:
            continue
        lat = pts[1]
        lon = pts[0]
        mean_lat += sum(lat)
        mean_lon += sum(lon)
        llen += len(lat)

    mean_lat /= llen
    mean_lon /= llen
    gmap = gmplot.GoogleMapPlotter(mean_lat, mean_lon, 14)
    for idx,pts in enumerate(points):
        # expects input in lats, longs
        gmap.plot(pts[1], pts[0], colors[idx], edge_width=5)
    gmap.draw(outpath)


def html_to_png(html_path, png_path):
    cmd = ["phantomjs","rasterize.js", html_path, png_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as ex:
        print("Failed to run command %s" % " ".join(cmd))
        print(ex)
        exit(1)


def visualize_paths(all_pts, colors, labels, file_name):
    img_names = []
    base_file_name = file_name
    # produce html and png per path
    print("Producing html and png files...")
    for i, pts in enumerate(all_pts):

        file_name = base_file_name + str(i) + ".html"
        write_group_gml(pts, file_name, colors[i])

        image_filename = file_name + ".jpg"
        # keep track of the image filenames to read them later
        img_names.append(image_filename)
        # produce the image
        html_to_png(file_name, image_filename)
        # delete the html
        os.remove(file_name)

    # read and display images in a collection of plots
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
