import json
import time

import pandas as pd
from paho.mqtt import client as mqtt_client


broker = '127.0.0.1'
port = 1883
topic = "/1dcnn"
client_id = f'1dcnn-client'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def load_data():
    names = ['TS1.txt', 'TS2.txt', 'TS3.txt', 'TS4.txt']
    df = pd.DataFrame()
    for name in names:
        data_file = f'./dataset/{name}'
        read_df = pd.read_csv(data_file, sep='\t', header=None)
        df = df.append(read_df)
    df = df.sort_index()
    df_values = df.values
    df = df_values.reshape(-1, 4, len(df.columns))
    data = df.transpose(0, 2, 1)
    return data


def publish(client):
    data = load_data()
    for x_data in data[-10:]:
        for y_data in x_data:
            t_1, t_2, t_3, t_4 = tuple(y_data)
            msg = {
                't1': round(t_1, 3),
                't2': round(t_2, 3),
                't3': round(t_3, 3),
                't4': round(t_4, 3)
            }
            time.sleep(1)
            result = client.publish(topic, json.dumps(msg))
            # result: [0, 1]
            status = result[0]
            if status == 0:
                print(f"Send `{msg}` to topic `{topic}`")
            else:
                print(f"Failed to send message to topic {topic}")


def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    run()
