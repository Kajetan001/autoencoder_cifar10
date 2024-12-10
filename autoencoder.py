from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# załadowanie i wstępne przetwarzanie zbiór danych CIFAR-10
def load_and_preprocess():
    (X_train, _), (X_test, _) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255  # normalizacja do zakresu [0, 1]
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(len(X_train), X_train.shape[1], X_train.shape[2], 3)  # zmiana kształtu dla conv2d
    X_test = X_test.reshape(len(X_test), X_test.shape[1], X_test.shape[2], 3)  # zmiana kształtu dla conv2d

    return X_test, X_train

# encoder
def encode(input_img):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

    return encoded

# decoder
def decode(encoded):
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return decoded

# kompilacja autoenkodera
def compile_autoencoder(input_img):
    encoded = encode(input_img)
    decoded = decode(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return autoencoder

def fit_and_predict_autoencoder(autoencoder, X_test, X_train, epochs=50):
    # trenowanie autoenkodera
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    # wizualizacja oryginalnych i zrekonstruowanych obrazów
    predicted = autoencoder.predict(X_test)

    return history, predicted