import FirstQuestion.CleanData as clean
import os
import datetime
import utils


def question_a2(output_folder, test_file, trips_list):
    test_list = utils.create_test_trip_list(test_file)
    k=5
    for i,test_trip in enumerate(test_list):
        print("Extracting neighbours for test trip %d/%d" % (i+1,len(test_list)))
        file_name = os.path.join(output_folder, "Question2A2", "subroutes_%d" % (i + 1))
        max_subseqs = find_similar_subroots_per_test_trip(test_trip, trips_list, k)
        preprocessing_for_visualisation(test_trip, max_subseqs, trips_list, file_name)


def find_similar_subroots_per_test_trip(test_trip, trips_list, k):
    test_lonlat = utils.create_list_tuples(test_trip)
    max_subseqs = []

    for tripidx, trip in enumerate(trips_list):
        trip_lonlat = utils.create_list_tuples(trip)
        millis_start = datetime.datetime.now()
        subseqs, pp = calc_lcss(test_lonlat, trip_lonlat)
        millis_end = datetime.datetime.now()
        elapsed = millis_end - millis_start
        # sort by decr. length
        pp = sorted(pp, key = lambda x : len(x), reverse=True)
        max_subseqs = update_lcss(max_subseqs, pp, k, elapsed, tripidx)
    return max_subseqs


def calc_lcss(t1, t2):
    L = [ [0 for _ in t2] for _ in t1]
    seqs = []
    idxs = []
    z = 0
    for i, p1 in enumerate(t1):
        for j, p2 in enumerate(t2):
            # for testing
            if type(p1) == str:
                equal = p1 == p2
            else:
                dist = clean.calculate_lonlat_distance(p1, p2)
                equal = dist < 200
            if equal:
                if i == 0 or j == 0:
                    L[i][j] =1
                    seqs.append(t2[j])
                    idxs.append([j])
                else:
                    L[i][j] =  L[i-1][j-1] + 1
                    if L[i][j] > z:
                        z = L[i][j]
                        seqs.append(t2[j-z+1:j+1])
                        idxs.append(list(range(j-z+1,j+1)))
            else:
                print(i,j,"-",p1,p2)
                for ll in L: print(ll)
                L[i][j] = 0
    return seqs, list(idxs)


def update_lcss(current, new_seqs, k, elapsed, tripidx):
    """
    Updates current seq with longest elements of new_seq, if applicable, i.e. if there's space left or a longer subseq
    is available
    :param current:
    :param new_seqs:
    :param k:
    :return:
    """
    count = 0
    for seq in new_seqs:
        do_sort = False
        count += 1
        if len(current) < k:
            # there's space, add it
            current.append((seq, elapsed, tripidx))
            do_sort = True
        else:
            # if the new guy is large enough
            if len(current[-1][0]) < len(seq):
                current[-1] = (seq, elapsed, tripidx)
                do_sort = True
        if do_sort:
            current = sorted(current, key = lambda x : len(x[0]), reverse=True)
        if count > k:
            break
    return current


def preprocessing_for_visualisation(test_trip, max_subseqs, trips_list, file_name):
    labels = ["test trip: %d" % test_trip[0]]  # test jid
    points = [[utils.idx_to_lonlat(list(range(len(test_trip[2:]))), test_trip)]]
    colors = [['b']]

    for j, m in enumerate(max_subseqs):
        trip = trips_list[m[2]]
        str = ["neighbour %d" % j, "jid: %s" % trip[1], "Matching pts: %d" % len(m[0]), "Delta-t: %s " % m[1]]
        labels.append("\n".join(str))
        mm = m[0]

        b1 = utils.idx_to_lonlat(list(range(0, mm[0])), trip)
        r = utils.idx_to_lonlat(mm, trip)
        b2 = utils.idx_to_lonlat(list(range(mm[-1], len(trip[2:]) - 1)), trip)

        pts = [b1, r, b2]

        points.append(pts)
        colors.append(['b', 'r', 'b'])
    utils.visualize_paths(points, colors, labels, file_name)