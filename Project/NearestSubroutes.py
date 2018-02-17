import utils
from multiprocessing.pool import ThreadPool
import threading
import question1 as qp1
import pprint


def find_similar_subroutes_per_test_trip(test_points, train_df, k, paropts=None, conseq_lcss = True, verbosity = False, unique_trip = True):
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
            max_subseqs = exec_with_processes(train_df, numpar, test_lonlat, k, conseq_lcss, unique_trip)
        elif partype == "threads":
            max_subseqs = exec_with_threads(train_df, numpar, test_lonlat, k, conseq_lcss, unique_trip)
    else:
        max_subseqs = serial_execution(train_df, test_lonlat, k, conseq_lcss, verbosity = verbosity, unique_trip = unique_trip)
    if len(max_subseqs) != k:
        print("WARNING: Specified %d subseqs!" % k)
    print("Extracted %d nearest subsequences of a %d-long test tring in: %s" % (len(test_points), k, utils.tictoc(timestart)))
    return max_subseqs


def serial_execution(df, test_lonlat, k, conseq_lcss, verbosity = False, unique_trip = True):
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
        # keep k largest
        subseqs_idx = subseqs_idx[:k]
        if subseqs_idx:
            #print("Train trip #", 1+index,"/",len(df), " of length %d | %d largest sub-routes:" % (len(train_points),k), subseqs_idx)
            pass
        if unique_trip:
            # keep at most one (the longest) subroute for each training trip 
            subseqs_idx = subseqs_idx[0:1]
        # update the list of the longest subsequences
        if subseqs:
            max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row, unique_trip = unique_trip)
            #print("Updated max subseqs, lens now:",[len(x[0]) for x in max_subseqs])
    if verbosity:
        print("Got %d subseqs:" % len(max_subseqs), [ (x,y,z["tripId"]) for (x,y,z) in max_subseqs])
    max_subseqs = check_reverse_lcss(max_subseqs, test_lonlat)
    return max_subseqs


def check_reverse_lcss(max_subseqs, test_lonlat):
    pass

def exec_with_processes(df, process_num, test_lonlat, k, unique_trip = True):
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
        max_subseqs = update_current_maxsubseq(max_subseqs, subseqs_idx, k, elapsed, row, unique_trip = unique_trip)
    print("Got %d common subsequences" % len(max_subseqs))
    pool.close()
    pool.join()
    return max_subseqs


def exec_with_threads(df, numpar, test_lonlat, k, unique_trip = True):
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
    cache = {}
    L = [ [0 for _ in t2 + [0]] for _ in t1 + [0]]
    for i in range(1,len(t1) + 1):
        for j in range(1,len(t2) + 1):
            if (i,j) not in cache:
                p1, p2 = t1[i-1], t2[j-1]
                dist = qp1.calculate_lonlat_distance(t1[i],t2[j])
                cache[(i,j)] = dist
                cache[(j,i)] = dist
            else:
                dist = cache[(i,j)]

            equal = dist < 200
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
    # distance cache
    dist_cache = {}
    # dist matrix for debugging
    D = [ [0 for _ in t2] for _ in t1]

    curr_len = 0
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

    # if match_happened:
    #     P1 = []
    #     P2 = []
    #     if type(t1[0]) == str:
    #         P1.append(['/'] + list(t2))
    #         for i,d in enumerate(D):
    #             P1.append([t1[i]] + [str(dd) for dd in d])
    #         for p in P1:
    #             print(p)

    #         print()


    #         P2.append(['/'] + list(t2))
    #         for i,l in enumerate(L):
    #             P2.append([t1[i]] + [str(ll) for ll in l])
    #         for p in P2:
    #             print(p)
    #     else:
    #         for d in D:
    #             print(d)
    #         print()
    #         for l in L:
    #             print(l)

    return seqs, list(idxs)


if __name__ == '__main__':
    s1 = "cardouker"
    s2 = "carder"
    print(s1,"vs",s2)
    seqs, idxs = calc_lcss(list(s1), list(s2), conseq_lcss=True)
    print("Consequtive:")
    for s,i in zip(seqs,idxs):
        print(s,i)
    seqs, idxs = calc_lcss(list(s1), list(s2), conseq_lcss=False)
    print("Nonconsequtive:")
    for s,i in zip(seqs,idxs):
        print(s,i)
    utils.subsets([1,2,7,1,3,9])


def update_current_maxsubseq(current, new_seqs, k, elapsed, row, unique_trip = True):
    """
    Updates current sequence with longest elements of new_seq, if applicable, i.e. if there's space left or a longer subseq
    is available
    :param current:
    :param new_seqs:
    :param k:
    :return:
    """
    if unique_trip:
        new_trip = row['tripId']
        current_trip_ids = [r['tripId'] for (_,_,r) in current]
        if new_trip in current_trip_ids:
            # replace only if new sequence length is larger
            existing_item = [item for item in current if item[2]['tripId'] == new_trip]
            if len(existing_item) > 1:
                print("Found more than one item with tripId",new_jid,"impossible!")
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
