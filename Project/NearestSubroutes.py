import utils
from multiprocessing.pool import ThreadPool
import threading
import question1 as qp1


def find_similar_subroutes_per_test_trip(test_points, train_df, k, paropts=None, conseq_lcss = True, verbosity = False, unique_jids = True):
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
            max_subseqs = exec_with_processes(train_df, numpar, test_lonlat, k, conseq_lcss, unique_jids)
        elif partype == "threads":
            max_subseqs = exec_with_threads(train_df, numpar, test_lonlat, k, conseq_lcss, unique_jids)
    else:
        max_subseqs = serial_execution(train_df, test_lonlat, k, conseq_lcss, verbosity = verbosity, unique_jids = unique_jids)
    if len(max_subseqs) != k:
        print("WARNING: Specified %d subseqs!" % k)
    print("Extracted %d nearest subsequences in: %s" % (k, utils.tictoc(timestart)))
    return max_subseqs


def serial_execution(df, test_lonlat, k, conseq_lcss, verbosity = False, unique_jids = True):
    max_subseqs = []
    # for each trip in the training data
    for index, row in df.iterrows():
        train_points = row["points"]
        train_points = eval(train_points)
        train_lonlat = utils.idx_to_lonlat(train_points, format="tuples")
        timestart = utils.tic()
        # compute common subsequences between the test trip and the current candidate
        subseqs, subseqs_idx = calc_lcss(test_lonlat, train_lonlat, conseq_lcss= conseq_lcss)
        elapsed = utils.tictoc(timestart)
        # sort by decr. length
        subseqs_idx = sorted(subseqs_idx, key=lambda x: len(x), reverse=True)
        if unique_jids:
            # keep at most one (the longest) subroute for each training trip 
            subseqs_idx = subseqs_idx[0:1]
        # update the list of the longest subsequences
        if subseqs:
            max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row, unique_jids = unique_jids)
            # print("Updated max subseqs, len now:",len(max_subseqs))
    if verbosity:
        print("Got %d subseqs:" % len(max_subseqs), [ (x,y,z["journeyId"]) for (x,y,z) in max_subseqs])
    return max_subseqs


def exec_with_processes(df, process_num, test_lonlat, k, unique_jids = True):
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
        max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row, unique_jids = unique_jids)
    print("Got %d common subsequences" % len(max_subseqs))
    pool.close()
    pool.join()
    return max_subseqs


def exec_with_threads(df, numpar, test_lonlat, k, unique_jids = True):
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


def calc_lcss_noconseq(t1, t2, subseqs=None, subseqs_idx=None):

    if subseqs is not None:
        seqs = subseqs
    else:
        seqs = []

    if subseqs_idx is not None:
        idxs = subseqs_idx
    else:
        idxs = []

    curr_len = 0
    L = [ [0 for _ in t2 + [0]] for _ in t1 + [0]]
    for i in range(1,len(t1) + 1):
        for j in range(1,len(t2) + 1):
            p1, p2 = t1[i-1], t2[j-1]
            dist = qp1.calculate_lonlat_distance(t1[i],t2[j])
            equal = dist < 200
            # equal = p1 == p2
            if equal:
                L[i][j] =  L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j-1], L[i-1][j])

    # read back result
    i, j = len(t1), len(t2)
    while i and j:
        if L[i][j] == L[i - 1][j]:
            i -= 1
        elif L[i][j] == L[i][j - 1]:
            j -= 1
        else:
            assert t1[i - 1] == t2[j - 1]
            seqs += t1[i - 1]
            idxs += [i-1]
            i -= 1
            j -= 1
    seqs.reverse()
    idxs.reverse()
    seqs = utils.subsets(seqs)
    idxs = utils.subsets(idxs)
    return seqs,list(idxs)


def calc_lcss(t1, t2, subseqs=None, subseqs_idx=None, conseq_lcss = False):
    '''
    :param t1: list of lonlat coordinate tuples
    :param t2: same
    :return:
    '''
    if not conseq_lcss:
        return calc_lcss_noconseq(t1, t2, subseqs, subseqs_idx)

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

    curr_len = 0
    for i, p1 in enumerate(t1):
        for j, p2 in enumerate(t2):
            # calculate the dist
            dist = qp1.calculate_lonlat_distance(p1, p2)
            equal = dist < 200
            # equal = p1 == p2
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
                    if L[i][j] > curr_len:
                        curr_len = L[i][j]
                        # append new longer sequence
                        seqs.append(t2[j-curr_len+1:j+1])
                        idxs.append(list(range(j-curr_len+1,j+1)))
            else:
                L[i][j] = 0

    return seqs, list(idxs)


if __name__ == '__main__':
    s1 = "datter"
    s2 = "dogger"
    print(s1,"vs",s2)
    seqs, idxs = calc_lcss(list(s1), list(s2), conseq_lcss=True)
    print(seqs, idxs)
    seqs, idxs = calc_lcss(list(s1), list(s2), conseq_lcss=False)
    print(seqs, idxs)
    utils.subsets([1,2,7,1,3,9])


def update_current_maxsubseq(current, new_seqs, k, elapsed, row, unique_jids = True):
    """
    Updates current sequence with longest elements of new_seq, if applicable, i.e. if there's space left or a longer subseq
    is available
    :param current:
    :param new_seqs:
    :param k:
    :return:
    """
    if unique_jids:
        new_jid = row['journeyId']
        current_jids = [r['journeyId'] for (_,_,r) in current]
        if new_jid in current_jids:
            # replace only if new sequence length is larger
            existing_item = [item for item in current if item[2]['journeyId'] == new_jid]
            if len(existing_item) > 1:
                print("Found more than one item with jid",new_jid,"impossible!")
                exit(1)
            # get the item, its position and its sequence length
            existing_item = existing_item[0]
            existing_idx = current.index(existing_item)
            old_len = len(existing_item[0])
            if len(new_seqs[0]) > old_len:
                current[existing_idx] = (new_seqs[0], elapsed, row)
                current = sorted(current, key = lambda x : len(x[0]), reverse=True)
            return current

    count = 0
    for seq in new_seqs:
        # if the same seq is, for some reason, already in the list, continue
        if seq in [x[0] for x in current]:
            continue
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
    points = [[utils.get_lonlat_tuple(test_points)]]
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
        # get the points data from the pandas dataframe
        train_points = sseq[2]["points"]
        train_points = utils.idx_to_lonlat(eval(train_points), format='tuples')
        # color matching points in red. Remaining points are drawn in blue.
        # get the points from the beginning up to the match
        b1 = train_points[0:point_idxs[0]+1]
        b1 = utils.get_lonlat_tuple(b1)
        if b1[0]:
            pts.append(b1)
            cols.append('b')
        # get the matching points
        r = train_points[point_idxs[0]:point_idxs[-1]+1]
        r = utils.get_lonlat_tuple(r)
        if r[0]:
            pts.append(r)
            cols.append('r')
        # get the points from the last matching point, to the end of the points
        b2 = train_points[point_idxs[-1]:]
        b2 = utils.get_lonlat_tuple(b2)
        if b2[0]:
            pts.append(b2)
            cols.append('b')

        # add to the list of points to draw
        points.append(pts)
        # as said above, the color sequence is blue, red, blue
        colors.append(cols)
        # print("Added pts:", pts)
        # print("Added cols:", cols)
    # send the whole parameter bundle to be drawn
    utils.visualize_point_sequences(points, colors, labels, file_name)
