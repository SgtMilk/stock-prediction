from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    input_array = []
    counter = 1
    while True:
        number = input("Dimension " + str(counter) + ": ")
        if number == "":
            break
        input_array.append(int(number))
        counter += 1

    input_shape = tuple(input_array)
    print("input shape: " + str(input_shape))
    model = build_model(tuple(input_array))
    model.summary()
