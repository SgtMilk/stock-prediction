from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def build_model(input_shape):
    """
    Builds the tensorflow model
    :param input_shape: tuple, shape of x_train[0]
    :return: the model
    """
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=(5, 6)))
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

    temp_input_shape = tuple(input_array)
    print("input shape: " + str(temp_input_shape))
    temp_model = build_model(tuple(input_array))
    temp_model.summary()
