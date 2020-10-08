import asyncio
import json

import numpy as np
import uvicorn
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse


app = Starlette()
queue = asyncio.Queue()
model = load_model('./1d-cnn.h5')


@app.on_event('startup')
async def on_startup():
    print('startup webhook')


@app.route('/webhook', methods=['POST'])
async def webhook(request):
    request_dict = await request.json()
    print(request_dict)
    payload = request_dict['payload']
    data = json.loads(payload)
    values = list(data.values())
    if queue.qsize() == 60:
        items = clear_queue(queue)
        task = BackgroundTask(predictive, data=items)
    else:
        task = None
    queue.put_nowait(values)
    record = {'status': 'success'}
    return JSONResponse(record, status_code=201, background=task)


async def predictive(data):
    y_label = {
        0: 3,
        1: 20,
        2: 100
    }
    y_status = {
        3: 'close to total failure',
        20: 'reduced efficiency',
        100: 'full efficiency'
    }
    x_test = np.array(data)
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    x_test = x_test.reshape(-1, x_test.shape[0], x_test.shape[1])
    results = model.predict(x_test)
    msg = "Current cooler state probability: "
    for i, probability in enumerate(results[0]):
        status = y_status[y_label[i]]
        msg += f"{probability * 100:.2f}% {status}({y_label[i]}), "
    print(msg)


def clear_queue(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8080,
        loop='uvloop',
        log_level='warning'
    )
