from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model: Model, x_test, technical_indicator, y_test_unscaled, normalizer: MinMaxScaler):

    predicted_y_test = model.predict(x_test)

    unscaled_predicted = normalizer.inverse_transform(predicted_y_test)

    assert predicted_y_test.shape == unscaled_predicted.shape

    real_mse = np.mean(np.square(y_test_unscaled - unscaled_predicted))
    scaled_mse = real_mse / (np.max(y_test_unscaled) -
                             np.min(y_test_unscaled)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)

    plt.plot([value[3] for value in y_test_unscaled][::-1], label='real')
    plt.plot([value[3]
              for value in unscaled_predicted][::-1], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()
