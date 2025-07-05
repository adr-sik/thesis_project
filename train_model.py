import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics

n = 3
features_length = 2048

data_path = f"path{n} {features_length}.pickle"

model = Sequential()
model.add(Dense(units=1024, input_shape=(2048,)))
model.add(Dense(units=512))
model.add(Dense(units=256))
model.add(Dense(units=64))
model.add(Dense(units=1, activation='sigmoid'))        

with (open(data_path, "rb")) as f:
    data = pickle.load(f)

x = np.array([data[i][1:-1] for i in range(len(data))])
y = np.array([data[i][-1] for i in range(len(data))])
   
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x, y, epochs=20, batch_size=100, shuffle=True)

model.save(f"{n}_{features_length}_model.keras")