from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Input


def build_model(input_shape):
    lstm_input = Input(shape=input_shape, name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(6, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)
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
