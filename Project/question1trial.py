import pandas as pd


def read_file(input_file, output_file):
    df = pd.read_csv(input_file)
    print("Reading training file", input_file, " - will ignore null journey ids")
    print("Parsing csv...")
    print("Removing nulls and NaNs ...")
    df = df[pd.notnull(df['journeyPatternId'])]
    df = df[df['journeyPatternId'] != "null"].reindex()
    vids={}
    trips=[]
    print("Transforming data to the required format")
    for index, row in df.iterrows():
        vid = row["vehicleID"]
        jid = row["journeyPatternId"]
        data = [row["timestamp"], row["longitude"], row["latitude"]]
        if vid not in vids:
            vids[vid] = []
        if not vids[vid]:
            vids[vid] = [jid, [data]]
        else:
            if vids[vid][0] == row["journeyPatternId"]:
                vids[vid][1].append(data)
            else:
                trips.append(vids[vid])
                vids[vid] = []
                vids[vid] = [jid, [data]]
        print(vids[vid])
    points = []
    jids=[]
    tripids = []
    print("Producing new dataframe")
    for i,trip in enumerate(trips):
        points.append(trip[1])
        jids.append(trip[0])
        tripids.append(i)
    df1 = pd.DataFrame({'tripId': tripids})
    df2 = pd.DataFrame({'journeyId': jids})
    df3 = pd.DataFrame({'points': points})
    df = pd.concat([df1, df2, df3], axis=1, ignore_index=False)
    df.to_csv(output_file)
    trips_list = df.to_dict(orient='dict')
    print("Done transforming data!")
    return trips_list, df