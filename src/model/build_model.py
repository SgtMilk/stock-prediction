from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Input, Concatenate


def build_model(input_shape, tech_shape):
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True,
                  input_shape=input_shape))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=6))
    return regressor


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
