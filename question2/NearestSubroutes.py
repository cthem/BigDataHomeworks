import question1.CleanData as clean
import os
import datetime
import utils


def question_a2(output_folder, test_file, trips_list):
    print("WARNING: Reading training file part because test file is not supplied.")
    test_list = trips_list[:5]
    k=5
    for i,test_trip in enumerate(test_list):
        print("Extracting subroutes for test trip %d/%d" % (i+1,len(test_list)))
        file_name = os.path.join(output_folder, "subroutes_%d_" % (i + 1))
        max_subseqs = find_similar_subroots_per_test_trip(test_trip, trips_list, k)
        preprocessing_for_visualisation(test_trip, max_subseqs, trips_list, file_name)


def find_similar_subroots_per_test_trip(test_trip, trips_list, k):
    test_lonlat = utils.idx_to_lonlat(test_trip, format="tuples")
    # keep track of the current common subsequences
    max_subseqs = []
    # iterate over the candidates
    for tripidx, trip in enumerate(trips_list):
        # get coordinates
        trip_lonlat = utils.idx_to_lonlat(trip, format="tuples")
        timestart = utils.tic()
        # compute common subsequences between the test trip and the current candidate
        subseqs, subseqs_idx = calc_lcss(test_lonlat, trip_lonlat)
        elapsed = utils.tictoc(timestart)
        # sort by decr. length
        subseqs_idx = sorted(subseqs_idx, key = lambda x : len(x), reverse=True)
        # update the list of the longest subsequences
        max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, tripidx)
    print("Got %d common subsequences" % len(max_subseqs))
    if len(max_subseqs) != k:
        print("WARNING: Specified %d subseqs!" % k)
    return max_subseqs


def calc_lcss(t1, t2):
    print("Have to check the lcss")
    '''

    :param t1: list of lonlat coordinate tuples
    :param t2: same
    :return:
    '''
    L = [ [0 for _ in t2] for _ in t1]
    # store the sequence of similar point values and indexes
    seqs, idxs = [], []
    z = 0
    for i, p1 in enumerate(t1):
        for j, p2 in enumerate(t2):
            # calculate the dist
            dist = clean.calculate_lonlat_distance(p1, p2)
            equal = dist < 200
            if equal:
                # the points are equal enough
                if i == 0 or j == 0:
                    # it's the first point of t1 or t2. Mark the cell to unit cost.
                    L[i][j] = 1
                    # initiate a new similar sequence
                    seqs.append(t2[j])
                    idxs.append([j])
                else:
                    # continue an existing sequence : current len is the previous plus one
                    L[i][j] =  L[i-1][j-1] + 1
                    #
                    if L[i][j] > z:
                        # current cost increases
                        z = L[i][j]
                        seqs.append(t2[j-z+1:j+1])
                        idxs.append(list(range(j-z+1,j+1)))
            else:
                #print(i,j,"-",p1,p2)
                #for ll in L: print(ll)
                L[i][j] = 0
    return seqs, list(idxs)


def update_current_maxsubseq(current, new_seqs, k, elapsed, tripidx):
    """
    Updates current sequence with longest elements of new_seq, if applicable, i.e. if there's space left or a longer subseq
    is available
    :param current:
    :param new_seqs:
    :param k:
    :return:
    """
    count = 0
    for seq in new_seqs:
        should_sort = False
        count += 1
        if len(current) < k:
            # there's space to add a subsequence, add it
            current.append((seq, elapsed, tripidx))
            # altered the list, we should sort it
            should_sort = True
        else:
            # no space - but check if the new guy is large enough - we just have to check if it's longer than the current shortest
            # it is sorted in descending length, so we only have to check the last one.
            if len(current[-1][0]) < len(seq):
                # big enough, replace!
                current[-1] = (seq, elapsed, tripidx)
                # changed the list, so we should sort it
                should_sort = True
        if should_sort:
            current = sorted(current, key = lambda x : len(x[0]), reverse=True)
        # the new_seqs sequence list is itself sorted, so no need to check more than k
        if count > k:
            break
    return current


def preprocessing_for_visualisation(test_trip, max_subseqs, trips_list, file_name):
    # initialize to the test trip data
    labels = ["test trip: %d" % test_trip[0]]  # test jid
    points = [[utils.idx_to_lonlat(test_trip)]]
    colors = [['b']]

    for j, sseq in enumerate(max_subseqs):
        cols, pts = [], []
        # trip jid
        trip = trips_list[sseq[2]]
        # label
        str = ["neighbour %d" % j, "jid: %s" % trip[1], "Matching pts: %d" % len(sseq[0]), "Delta-t: %s " % sseq[1]]
        labels.append("\n".join(str))

        # get the indexes of common points
        point_idxs = sseq[0]

        # color matching points in red. Remaining points are drawn in blue.
        # get the points from the beginning up to the match
        b1 = utils.idx_to_lonlat(idx=list(range(0, point_idxs[0])), trip=trip)
        if b1[0]:
            pts.append(b1)
            cols.append('b')
        # get the matching points
        r = utils.idx_to_lonlat(idx=point_idxs, trip=trip)
        if r[0]:
            pts.append(r)
            cols.append('r')
        # get the points from the last matching point, to the end of the points
        b2 = utils.idx_to_lonlat(idx=list(range(point_idxs[-1], len(trip[2:]) - 1)), trip=trip)
        if b2[0]:
            pts.append(b2)
            cols.append('b')

        # add to the list of points to draw
        points.append(pts)
        # as said above, the color sequence is blue, red, blue
        colors.append(cols)
    # send the whole parameter bundle to be drawn
    utils.visualize_paths(points, colors, labels, file_name)