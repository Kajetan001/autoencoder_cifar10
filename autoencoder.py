from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from keras.models import Model


# załadowanie i wstępne przetwarzanie zbioru danych CIFAR-10
def load_and_preprocess():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255 

    return X_test, X_train, y_test, y_train

# encoder
# trzy warstwy konwolucyjne zmniejszające stopniowo wymiary przestrzenne obrazów
def encode(input_img):
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    # BatchNormalization normalizuje wariancję i średnią każdej warstwy dla przyśpieszenia procesu uczenia
    x = BatchNormalization()(x)
    # LeakuReLU zapobiega "śmierci neuronów" zastępując wartości zerowe bardzo niskimi wartościami
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    return encoded

# decoder
# odwrócenie procesu kodowania, trzy warstwy konwolucyjne
def decode(encoded):
    x = Conv2D(8, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return decoded

# wywoładnie encodera, decodera, kompilacja autoenkodera
def build_autoencoder(input_img):
    encoded = encode(input_img)
    decoded = decode(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def fit_and_predict_autoencoder(autoencoder, X_test, X_train, epochs=50):
    # trenowanie autoenkodera
    autoencoder = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    # predykcja obrazów dodkonana na testowym zbiorze danych
    predicted = autoencoder.predict(X_test)

    return autoencoder, predicted
