import utils
from multiprocessing.pool import ThreadPool
import threading
import question1 as qp1
import pprint


def find_similar_subroutes_per_test_trip(test_points, train_df, k, paropts=None, verbosity = False):
    if paropts:
        print("Parallelizing with", paropts)
        partype, numpar = paropts
    else:
        partype, numpar = None, None

    timestart = utils.tic();
    test_lonlat = utils.idx_to_lonlat(test_points, format="tuples")
    max_subseqs = []
    if partype:
        # num threads or processes
        if partype == "processes":
            max_subseqs = exec_with_processes(train_df, numpar, test_lonlat, k)
        elif partype == "threads":
            max_subseqs = exec_with_threads(train_df, numpar, test_lonlat, k)
    else:
        max_subseqs = serial_execution(train_df, test_lonlat, k, verbosity = verbosity)
    if len(max_subseqs) != k:
        print("WARNING: Specified %d subseqs!" % k)
    print("Extracted %d nearest subsequences of a %d-long test tring in: %s" % (len(test_points), k, utils.tictoc(timestart)))
    return max_subseqs


def serial_execution(df, test_lonlat, k, verbosity = False):
    max_subseqs = []
    # for each trip in the training data
    for index, row in df.iterrows():
        if index > 500:
            break
        train_points = row["points"]
        train_points = eval(train_points)
        train_lonlat = utils.idx_to_lonlat(train_points, format="tuples")
        timestart = utils.tic()
        # compute common subsequences between the test trip and the current candidate
        _ , subseqs_idx_list = calc_lcss(test_lonlat, train_lonlat)
        # consider non-consequtive subroutes
        subseqs_idx = list(set([idx for seq in subseqs_idx_list for idx in seq]))
        elapsed = utils.tictoc(timestart)
        # sort by decr. length
        subseqs_idx.sort(reverse=True)
        # update the list of the longest subsequences
        if subseqs_idx:
            max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row)
            print("Max subseq length:",len(max_subseqs))
            print([x[0] for x in max_subseqs])
            print("Updated max subseqs, lens now:",[len(x[0]) for x in max_subseqs])
    if verbosity:
        print("Got %d subseqs:" % len(max_subseqs), [ (x,y,z["tripId"]) for (x,y,z) in max_subseqs])

    #max_subseqs = check_reverse_lcss(max_subseqs, test_lonlat, k)
    if verbosity:
        print("Got %d reversed: subseqs:" % len(max_subseqs), [ (x,y,z["tripId"]) for (x,y,z) in max_subseqs])

    return max_subseqs


def check_reverse_lcss(max_subseqs, test_lonlat, k):
    new_subseqs = []
    for i,mxs in enumerate(max_subseqs):
        (seq_old, elapsed, row) = mxs
        # get reversed points
        train_pts = eval(row["points"])
        train_lonlat = utils.idx_to_lonlat(train_pts, format="tuples")
        _, seq_old_again = calc_lcss(test_lonlat, train_lonlat)
        train_lonlat = train_lonlat[-1::-1]
        _, idxs = calc_lcss(test_lonlat, train_lonlat)
        # re-reverse
        if idxs:
            idxs = [ii[-1::-1] for ii in idxs]
            idxs = list(set([idx for seq in idxs for idx in seq]))
            max_subseqs = update_current_maxsubseq(max_subseqs, idxs, k, elapsed, row)
    return max_subseqs

def exec_with_processes(df, process_num, test_lonlat, k):
    max_subseqs = []
    pool = ThreadPool(processes=process_num)
    for index, row in df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        train_lonlat = utils.idx_to_lonlat(train_points, format="tuples")
        timestart = utils.tic()
        # compute common subsequences between the test trip and the current candidate
        async_result = pool.apply_async(calc_lcss, (test_lonlat, train_lonlat))
        subseqs, subseqs_idx = async_result.get()
        elapsed = utils.tictoc(timestart)
        # sort by decr. length
        subseqs_idx = sorted(subseqs_idx, key=lambda x: len(x), reverse=True)
        # update the list of the longest subsequences
        max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row)
    print("Got %d common subsequences" % len(max_subseqs))
    pool.close()
    pool.join()
    return max_subseqs


def exec_with_threads(df, numpar, test_lonlat, k):
    max_subseqs = []
    res1 = [[] for _ in range(numpar)]
    res2 = [[] for _ in range(numpar)]
    subframes = utils.get_sub_dataframes(df, numpar)
    # assign data and start the threads
    threads = []
    timestart = utils.tic()
    for i in range(numpar):
        train_lonlat = []
        for index, row in subframes[i].iterrows():
            train_points = row["points"]
            train_points = eval(train_points)
            train_lonlat = utils.idx_to_lonlat(train_points, format="tuples")
        threads.append(threading.Thread(target=calc_lcss, args=(test_lonlat, train_lonlat, res1, res2)))
        threads[i].start()
    # gather and merge results
    subseqs = []
    subseqs_idx = []
    for i in range(numpar):
        threads[i].join()
        subseqs += res1[i]
        subseqs_idx += res2[i]
    subseqs_idx = sorted(subseqs_idx, key=lambda x: len(x), reverse=True)
    elapsed = utils.tictoc(timestart)
    max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row)
    return max_subseqs


def calc_lcss(t1, t2, subseqs=None, subseqs_idx=None):
    '''
    :param t1: list of lonlat coordinate tuples
    :param t2: same
    :return:
    '''
    if subseqs is not None:
        seqs = subseqs
    else:
        seqs = []

    if subseqs_idx is not None:
        idxs = subseqs_idx
    else:
        idxs = []


    # initialize lcss matrix to 0
    L = [ [0 for _ in t2] for _ in t1]
    # distance cache
    dist_cache = {}
    # dist matrix for debugging
    D = [ [0 for _ in t2] for _ in t1]

    for i, p1 in enumerate(t1):
        for j, p2 in enumerate(t2):
            # calculate the dist
            if (i,j) not in dist_cache:
                dist = qp1.calculate_lonlat_distance(p1, p2)
                dist_cache[(i,j)] = dist
                dist_cache[(j,i)] = dist
            else:
                dist = dist_cache[(i,j)]
            equal = dist < 200
            if equal:
                # the points are equal enough
                if i == 0 or j == 0:
                    # it's the first point of t1 or t2. Mark the cell to unit cost.
                    L[i][j] = 1
                    # add unitary sequence
                    seqs.append(p2)
                    idxs.append([j])
                else:
                    # continue an existing sequence: current len is the previous plus one
                    L[i][j] =  L[i-1][j-1] + 1
                    # if L[i][j] > curr_len:
                    curr_len = L[i][j]
                    # append new longer sequence
                    new_seq = t2[j-curr_len+1:j+1]
                    new_seq_idxs = list(range(j-curr_len+1,j+1))
                    if new_seq not in seqs:
                        seqs.append(new_seq)
                        idxs.append(new_seq_idxs)
                    # else:
                    #     print("Curr len at",i,j,":",p1,p2,"not gr8r than curr_len:",curr_len," -> not adding to the seqs")
            else:
                L[i][j] = 0

    return seqs, list(idxs)


if __name__ == '__main__':
    s1 = "cardouker"
    s2 = "carder"
    print(s1,"vs",s2)
    seqs, idxs = calc_lcss(list(s1), list(s2))
    print("Consequtive:")
    for s,i in zip(seqs,idxs):
        print(s,i)
    seqs, idxs = calc_lcss(list(s1), list(s2))
    print("Nonconsequtive:")
    for s,i in zip(seqs,idxs):
        print(s,i)
    utils.subsets([1,2,7,1,3,9])


def update_current_maxsubseq(current, new_seq, k, elapsed, row):
    """
    Updates current sequence with longest elements of new_seq, if applicable, i.e. if there's space left or a longer subseq
    is available
    :param current:
    :param new_seq:
    :param k:
    :return:
    """
    should_sort = False
    if len(current) < k:
        # there's space to add a subsequence, add it
        current.append((new_seq, elapsed, row))
        # altered the list, we should sort it
        should_sort = True
    else:
        # no space - but check if the new guy is large enough - we just have to check if it's longer than the current shortest
        # it is sorted in descending length, so we only have to check the last one.
        if len(current[-1][0]) < len(new_seq):
            # big enough, replace!
            current[-1] = (new_seq, elapsed, row)
            # changed the list, so we should sort it
            should_sort = True
    if should_sort:
        current = sorted(current, key = lambda x : len(x[0]), reverse=True)

    return current


def make_consequtive(idxs):
    """
    Split a list of integers into list of sublists, s.t. each sublist contains consequtive integer lists
    :param idxs: list of integers
    :return: list of lists of integers
    """
    idxs.sort()
    res = [[idxs[0]]]
    for idx in idxs[1:]:
        if idx - res[-1][-1] <= 1:
            res[-1].append(idx)
            continue
        res.append([idx])
    return [sorted(t) for t in res]


def preprocessing_for_visualisation(test_points, max_subseqs, file_name, index):
    # initialize to the test trip data
    labels = ["test trip: %s" % index]  # test jid
    points = [[utils.get_lonlat_tuple(test_points)]]
    colors = [['b']]

    for j, sseq in enumerate(max_subseqs):
        cols, print_idxs, pts = [], [], []
        # trip jid
        jid = sseq[2]["journeyId"]
        subseq_idxs = sseq[0]
        num_points = sum([len(x) for x in subseq_idxs])
        # label
        str = ["neighbour %d" % j, "jid: %s" % jid, "Matching pts: %d" % num_points, "Delta-t: %s " % sseq[1]]
        labels.append("\n".join(str))
        print("seq first/last idxs:",[(s[0],s[-1]) for s in subseq_idxs])

        # get the points data from the pandas dataframe to lonlat tuples
        train_points = sseq[2]["points"]
        train_points = utils.idx_to_lonlat(eval(train_points), format='tuples')

        # prepend blue, if list is not starting at first point
        if subseq_idxs[0][0] > 0:
            idxs = list(range(subseq_idxs[0][0]+2))
            print_idxs.append(list(range(idxs[0],idxs[-1])))
            cols.append('b')

        # for each sequence, make the matching red and the following blue, if any
        for seq_idx, idxs in enumerate(subseq_idxs):
            # the match
            if idxs:
                print_idxs.append(idxs)
                cols.append('r')

            # check for a following blue portion: not existent iff last seq idx is last idx of the trip
            if idxs[-1] == len(train_points) -1:
                continue
            # else, either up to first point of next subsequence, or last point in row
            if seq_idx == len(subseq_idxs) -1:
                next_seq_first_pt = len(train_points) -1
            else:
                next_seq_first_pt = subseq_idxs[seq_idx+1][0]
            # blue it up
            b = list(range(idxs[-1],next_seq_first_pt+1))
            if b[0]:
                print_idxs.append(b)
                cols.append('b')

        # append the points corresponding to the indexes
        for i,idx_list in enumerate(print_idxs):
            pts.append(utils.get_lonlat_tuple([train_points[i] for i in idx_list]))
            print("Idx list:",idx_list[0],idx_list[-1],"col:",cols[i])

        # add to the list of points to draw
        points.append(pts)
        colors.append(cols)
        # print("Added pts:", pts)
        # print("Added cols:", cols)
    # send the whole parameter bundle to be drawn
    utils.visualize_point_sequences(points, colors, labels, file_name)
