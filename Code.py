import json
import random
from random import shuffle


class Disease:
    def __init__(self, type, age, temperature, pressure, cholesterol, sugar, heartbeat, breath, IMB, leucocytes,
                 hemoglobin):
        self.type = type
        self.age = age
        self.temperature = temperature
        self.pressure = pressure
        self.cholesterol = cholesterol
        self.sugar = sugar
        self.heartbeat = heartbeat
        self.breath = breath
        self.IMB = IMB
        self.leucocytes = leucocytes
        self.hemoglobin = hemoglobin


def generateDisease(type):
    if type == "Autoimmune":
        age = random.randint(20, 60)
        temperature = round(random.uniform(35, 38), 1)
        pressure = round(random.uniform(90, 160), 1)
        cholesterol = round(random.uniform(150, 250), 1)
        sugar = round(random.uniform(3.9, 8.3), 1)
        heartbeat = round(random.uniform(55, 110), 1)
        breath = round(random.uniform(10, 25), 1)
        IMB = round(random.uniform(18, 35), 1)
        leucocytes = round(random.uniform(4, 14), 1)
        hemoglobin = round(random.uniform(10, 17), 1)
    elif type == "Gastrointestinal":
        age = random.randint(20, 60)
        temperature = round(random.uniform(35, 38), 1)
        pressure = round(random.uniform(90, 160), 1)
        cholesterol = round(random.uniform(120, 270), 1)
        sugar = round(random.uniform(3.9, 8.9), 1)
        heartbeat = round(random.uniform(55, 110), 1)
        breath = round(random.uniform(10, 25), 1)
        IMB = round(random.uniform(18, 35), 1)
        leucocytes = round(random.uniform(4, 12), 1)
        hemoglobin = round(random.uniform(10, 16), 1)
    elif type == "Сardiac":
        age = random.randint(20, 60)
        temperature = round(random.uniform(35, 38), 1)
        pressure = round(random.uniform(120, 200), 1)
        cholesterol = round(random.uniform(150, 300), 1)
        sugar = round(random.uniform(4.4, 11.1), 1)
        heartbeat = round(random.uniform(50, 120), 1)
        breath = round(random.uniform(10, 25), 1)
        IMB = round(random.uniform(20, 35), 1)
        leucocytes = round(random.uniform(4, 12), 1)
        hemoglobin = round(random.uniform(12, 18), 1)
    elif type == "Respiratory":
        age = random.randint(20, 60)
        temperature = round(random.uniform(35, 38), 1)
        pressure = round(random.uniform(90, 150), 1)
        cholesterol = round(random.uniform(100, 250), 1)
        sugar = round(random.uniform(3.9, 7.7), 1)
        heartbeat = round(random.uniform(60, 110), 1)
        breath = round(random.uniform(15, 30), 1)
        IMB = round(random.uniform(18.5, 30), 1)
        leucocytes = round(random.uniform(4, 12), 1)
        hemoglobin = round(random.uniform(11, 17), 1)
    elif type == "Infectious":
        age = random.randint(20, 60)
        temperature = round(random.uniform(37, 41), 1)
        pressure = round(random.uniform(90, 140), 1)
        cholesterol = round(random.uniform(100, 200), 1)
        sugar = round(random.uniform(3.9, 7.2), 1)
        heartbeat = round(random.uniform(60, 100), 1)
        breath = round(random.uniform(12, 30), 1)
        IMB = round(random.uniform(18.5, 30), 1)
        leucocytes = round(random.uniform(5, 15), 1)
        hemoglobin = round(random.uniform(11, 16), 1)
    return Disease(type, age, temperature, pressure, cholesterol, sugar, heartbeat, breath, IMB, leucocytes, hemoglobin)


def generation():
    autoimmune = [generateDisease("Autoimmune") for _ in range(30)]
    gastrointestinal = [generateDisease("Gastrointestinal") for _ in range(30)]
    cardiac = [generateDisease("Сardiac") for _ in range(30)]
    respiratory = [generateDisease("Respiratory") for _ in range(30)]
    infectious = [generateDisease("Infectious") for _ in range(30)]
    generationDisease = [vars(disease) for disease in
                         (autoimmune + gastrointestinal + cardiac + respiratory + infectious)]

    shuffle(generationDisease)

    with open("generationDisease.json", "w") as file:
        json.dump(generationDisease, file)

    x = []
    y = []
    for disease in generationDisease:
        values = list(disease.values())
        x.append(values[1::])
        type = values[0]
        if type == "Autoimmune":
            y.append(0)
        elif type == "Gastrointestinal":
            y.append(1)
        elif type == "Сardiac":
            y.append(2)
        elif type == "Respiratory":
            y.append(3)
        elif type == "Infectious":
            y.append(4)

    for disease in x:
        for s in disease:
            s = s / 500
    return x, y


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

xTrain, yTrain = generation()
xTest, yTest = generation()

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

model = keras.Sequential()
model.add(Dense(units=16, input_dim=10, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=100, batch_size=10, validation_split=0.2, verbose=0)
loss, accuracy = model.evaluate(xTest, yTest, verbose=0)
print(f'Loss: {loss}, Accuracy: {int(accuracy * 100)}%')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True)
plt.show()

print(xTest[0], yTest[0])
predictions = model.predict(np.array([xTest[0]]), verbose=0)
confidence = np.max(predictions)
prediction = np.argmax(predictions)
print("Prediction:", prediction, "Confidence:", confidence)
