import pandas as pd


def create_trips_file(input_file, output_file):
    print("Reading training file with pandas ignoring null jids", input_file)
    df = pd.read_csv(input_file)
    df = df[pd.notnull(df['journeyPatternId'])].reindex()
    timestamps = df.groupby(["vehicleID", "journeyPatternId"])["timestamp"].apply(list)
    lons = df.groupby(["vehicleID", "journeyPatternId"])["longitude"].apply(list)
    lats = df.groupby(["vehicleID", "journeyPatternId"])["latitude"].apply(list)
    df = pd.concat([timestamps, lons, lats], axis=1, ignore_index=False)
    print("File read")
    print("Start processing data")

    for index, row in df.iterrows():
        tslist, lonslist, latslist = [], [], []
        tslist = row["timestamp"]
        lonslist = row["longitude"]
        latslist = row["latitude"]
        data_list = []
        for i,timestamp in enumerate(tslist):
            new_list = []
            new_list.append(timestamp)
            new_list.append(lonslist[i])
            new_list.append(latslist[i])
            data_list.append(new_list)
        row["timestamp"] = data_list
    new_df = create_new_dataframe()
    new_df = pd.concat([new_df, df["timestamp"]], axis=1, ignore_index=False)
    new_df.to_csv(output_file)
    df = pd.read_csv(output_file)
    df = df.drop(["vehicleID", "tripId"], axis=1)
    df.index.name = "tripId"
    df.columns = ["journeyId", "points"]
    df = sort_timestamps(df)
    df.to_csv(output_file)
    trips_list = df.to_dict(orient='dict')
    return trips_list, df


def create_new_dataframe():
    columns = []
    columns.append("tripId")
    new_df = pd.DataFrame(columns=columns)
    return new_df


def sort_timestamps(df):
    for index, row in df.iterrows():
        points = row["points"]
        points = eval(points)
        points.sort(key=lambda x: x[0])
        df.at[index, "points"] = points
    return df