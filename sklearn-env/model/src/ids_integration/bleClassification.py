import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# load the trained model
model = joblib.load("src/bleModel.joblib")  # file path to the model

# convert BLE packet list to dataframe
def convert_to_frame(packets):
    df = []  # new list
    # iterate through list of objects and grab what the model needs
    for packet in packets:
        df.append({
            "Timestamp": packet.timestamp,
            "Channel Index": packet.channel_index,
            "Advertising Address": packet.advertising_address,
            "Packet counter": packet.packet_counter,
            "Power Level (dBm)": packet.power_level
        })
    return pd.DataFrame(df) # converts list to dataframe

# cleans the newly created frame by filling null values
def quickCleanse(df):
    df['Timestamp'] = df['Timestamp'].fillna("-1.0")
    df['Channel Index'] = df['Channel Index'].fillna("-1")
    df['Advertising Address'] = df['Advertising Address'].fillna("0")
    df['Packet counter'] = df['Packet counter'].fillna("-1")
    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna("-255")

# scale, classify, and process the input list of BLE packet objects
def classify(packets):
    df = convert_to_frame(packets)   # create frame
    quickCleanse(df)    # get rid of any null values
    
    # scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # predict and output a result
    predictions = model.predict(scaled_data)
    return ["benign" if pred == 0 else "malicious" for pred in predictions]