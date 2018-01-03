import utils

def question_1a(input_file, output_file):
    '''
    Runs the first subquestion
    :param input_file: The training file
    :param output_file: The processed file with the trips
    :return:
    '''
    line_objects = utils.read_train_set(input_file)
    map_per_vid = get_data_per_vehicle_id(line_objects)
    map_per_vidjid = get_data_per_vehicle_and_journey_id(map_per_vid)
    trips_list = write_trips(output_file, map_per_vidjid)
    print("Done with question 1!")
    return trips_list

def get_data_per_vehicle_id(line_objects):
    '''
    Aggregates data wrt the vehicle ID
    :param line_objects:
    :return:
    '''
    print("Aggregating per vehicle id")
    map_per_vid = {}
    for obj in line_objects:
        vid = obj["vid"]
        if vid not in map_per_vid:
            map_per_vid[vid] = []
        obj_without_vid = {key: obj[key] for key in obj if key != "vid"}
        map_per_vid[vid].append(obj_without_vid)
    return map_per_vid

def get_data_per_vehicle_and_journey_id(map_per_vid):
    '''
    Aggregates data wrt the journey ID
    :param map_per_vid:
    :return: vid1:{jid1:[pt1,pt2...], jid2:[pt1,pt2],...},vid2:{...}
    '''
    print("Aggregating per journey id")
    for vehicle_id in map_per_vid:
        # get elements for this vehicle id
        map_per_jid = {}
        vid_list = map_per_vid[vehicle_id]
        # make a map wrt to the journey id
        for item in vid_list:
            jid = item["jid"]
            if jid not in map_per_jid:
                map_per_jid[jid] = []
            # assign the points of each journeyid, minus that id
            obj_without_jid = {key: item[key] for key in item if key != "jid"}
            map_per_jid[jid].append(obj_without_jid)
        # for each jid for this vehicle
        for jid in map_per_jid:
            # get poitns
            jid_list = map_per_jid[jid]
            # sort per timestamp
            sorted_jid_list = sorted(jid_list, key=lambda k: k['ts'])
            map_per_jid[jid] = sorted_jid_list
        # assign jid-points map to this vehicle id
        map_per_vid[vehicle_id] = map_per_jid
    return map_per_vid

def write_trips(filename, map_per_vid):
    '''
    Writes trips in the output file
    :param filename: the name of the file
    :param map_per_vid: the map with all the data
    :return: the list with all the data (this list is written in the file)
    '''
    print("Writing %d trips to file %s" % (len(map_per_vid), filename))
    sorted_trips = []
    with open(filename, "w") as f:
        count = 0
        for vehicle_id in map_per_vid:
            for journey_id in map_per_vid[vehicle_id]:
                objlist = map_per_vid[vehicle_id][journey_id]
                sorted_objlist = [[float(o[k]) for k in sorted(list(o), reverse=True)] for o in objlist]
                lonlatts_str = str(sorted_objlist)
                str_to_write = "%d;%s;%s\n" % (count, journey_id, lonlatts_str)
                sorted_objlist = [count] + [journey_id] + sorted_objlist
                sorted_trips.append(sorted_objlist)
                map_per_vid[vehicle_id][journey_id] = sorted_objlist
                f.write(str_to_write)
                count += 1
    return sorted_trips
