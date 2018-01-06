import utils_pandas as up
from multiprocessing.pool import ThreadPool
import question1_pandas as qp1


#TODO include only one time a trip
def find_similar_subroutes_per_test_trip(test_points, train_df, k=5):
    test_lonlat = up.idx_to_lonlat(test_points, format="tuples")
    max_subseqs = []
    pool = ThreadPool(processes=10)
    for index, row in train_df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        train_lonlat = up.idx_to_lonlat(train_points, format="tuples")
        timestart = up.tic()
        # compute common subsequences between the test trip and the current candidate
        async_result = pool.apply_async(calc_lcss, (test_lonlat, train_lonlat))
        subseqs, subseqs_idx = async_result.get()
        elapsed = up.tictoc(timestart)
        # sort by decr. length
        subseqs_idx = sorted(subseqs_idx, key=lambda x: len(x), reverse=True)
        # update the list of the longest subsequences
        max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row)
    print("Got %d common subsequences" % len(max_subseqs))
    pool.close()
    pool.join()
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
            dist = qp1.calculate_lonlat_distance(p1, p2)
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
                # print(i,j,"-",p1,p2)
                # for ll in L: print(ll)
                L[i][j] = 0
    return seqs, list(idxs)


def update_current_maxsubseq(current, new_seqs, k, elapsed, row):
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
            current.append((seq, elapsed, row))
            # altered the list, we should sort it
            should_sort = True
        else:
            # no space - but check if the new guy is large enough - we just have to check if it's longer than the current shortest
            # it is sorted in descending length, so we only have to check the last one.
            if len(current[-1][0]) < len(seq):
                # big enough, replace!
                current[-1] = (seq, elapsed, row)
                # changed the list, so we should sort it
                should_sort = True
        if should_sort:
            current = sorted(current, key = lambda x : len(x[0]), reverse=True)
        # the new_seqs sequence list is itself sorted, so no need to check more than k
        if count > k:
            break
    return current


def preprocessing_for_visualisation(test_points, max_subseqs, file_name, index):
    # initialize to the test trip data
    labels = ["test trip: %s" % index]  # test jid
    points = [[up.get_lonlat_tuple(test_points)]]
    colors = [['b']]

    for j, sseq in enumerate(max_subseqs):
        cols, pts = [], []
        # trip jid
        journey_id = sseq[2]["journeyId"]
        # label
        str = ["neighbour %d" % j, "jid: %s" % journey_id, "Matching pts: %d" % len(sseq[0]), "Delta-t: %s " % sseq[1]]
        labels.append("\n".join(str))

        # get the indexes of common points
        point_idxs = sseq[0]
        train_points = sseq[2]["points"]
        train_points = eval(train_points)
        # color matching points in red. Remaining points are drawn in blue.
        # get the points from the beginning up to the match
        b1 = train_points[0:point_idxs[0]+1]
        b1 = up.get_lonlat_tuple(b1)
        if b1[0]:
            pts.append(b1)
            cols.append('b')
        # get the matching points
        r = train_points[point_idxs[0]:point_idxs[-1]+1]
        r = up.get_lonlat_tuple(r)
        if r[0]:
            pts.append(r)
            cols.append('r')
        # get the points from the last matching point, to the end of the points
        b2 = train_points[point_idxs[-1]:]
        b2 = up.get_lonlat_tuple(b2)
        if b2[0]:
            pts.append(b2)
            cols.append('b')

        # add to the list of points to draw
        points.append(pts)
        # as said above, the color sequence is blue, red, blue
        colors.append(cols)
        print("Added pts:", pts)
        print("Added cols:", cols)
    # send the whole parameter bundle to be drawn
    up.visualize_point_sequences(points, colors, labels, file_name)