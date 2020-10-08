import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

label = pd.read_csv(
    './dataset/profile.txt', sep='\t',
    names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Flag'],
    header=None
)

data = ['TS1.txt', 'TS2.txt', 'TS3.txt', 'TS4.txt']
df = pd.DataFrame()

for txt in data:
    data_file = './dataset/' + txt
    read_df = pd.read_csv(data_file, sep='\t', header=None)
    df = df.append(read_df)

df = df.sort_index()
# 转换数据结构为4列
df_values = df.values
df = df_values.reshape(-1, len(data), len(df.columns))
df = df.transpose(0, 2, 1)

plt.figure(figsize=(8, 5))
plt.plot(df[0])
plt.title('Cooler Temperature Data')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.show()
np.set_printoptions(False)

label = label.Cooler

diz_label, diz_reverse_label = {}, {}
for i, lab in enumerate(label.unique()):
    diz_label[lab] = i
    diz_reverse_label[i] = lab

# 将 3 -> 0, 20 -> 1, 100 -> 2
label = label.map(diz_label)
# 将类别向量转换为二进制（只有0和1）的矩阵类型表示
y = to_categorical(label)

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42, test_size=0.2)


# 数据标准化


def con(x):
    q_test = []
    for q in x:
        scaler = StandardScaler()
        i_test = scaler.fit_transform(q.reshape(-1, q.shape[-1])).reshape(q.shape)
        q_test.append(i_test)
    return np.array(q_test)


X_train = con(X_train)
X_test = con(X_test)

# scaler = StandardScaler()
#
# X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
# X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

num_sensors = 4
TIME_PERIODS = 60
BATCH_SIZE = 16
EPOCHS = 10

model_m = Sequential()
model_m.add(Conv1D(100, 6, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 6, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 6, activation='relu'))
model_m.add(Conv1D(160, 6, activation='relu'))
model_m.add(GlobalAveragePooling1D(name='G_A_P_1D'))
model_m.add(Dropout(0.5))
model_m.add(Dense(3, activation='softmax'))

model_m.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
history = model_m.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

dc = model_m.predict(X_test)
pred_test = np.argmax(dc, axis=1)

qw = classification_report([diz_reverse_label[np.argmax(label)] for label in y_test],
                           [diz_reverse_label[label] for label in pred_test])
model_m.save('test.h5')
