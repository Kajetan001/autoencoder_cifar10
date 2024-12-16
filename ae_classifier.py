import tensorflow as tf
import numpy as np


# trenowanie pojedynczego autoenkodera na danych konkretnej klasy
def train_autoencoder_for_class(autoencoder, class_data, epochs=50, batch_size=256):
    autoencoder.fit(class_data, class_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

# klasyfikacja obraz na podstawie błędu rekonstrukcji autoenkoderów
# reduce_retracing=True: optymalizacja dla tensorflow w celu zmniejszenia liczby ponownych śladów funkcji
# wcześniej wyskakiwał warning tego dotyczący
@tf.function(reduce_retracing=True)
def classify_image(image, autoencoders):
    errors = []
    # dodanie wymiaru batch do obrazu
    image = tf.expand_dims(image, axis=0)

    # iteracja po autoenkoderach, każdy odpowiada za jedną klasę
    for ae in autoencoders:
        reconstruction = ae(image, training=False) 
        error = tf.reduce_mean(tf.square(image - reconstruction)) # obliczenie błędu średniokwadratowego
        errors.append(error)

    # zwrócenie indeksu autoenkodera o najmniejszym błędzie (przewidywana klasa)
    return tf.argmin(errors)

# funkcja do trenowania wszystkich autoenkoderów dla poszczególnych klas
def train_all_autoencoders(X_train, y_train, autoencoders):
    # iteracja po indeksach klas oraz odpowiadających im autoenkoderach
    for i, ae in enumerate(autoencoders):
        print(f"Training autoencoder for class {i}")
        class_data = X_train[y_train.flatten() == i]
        train_autoencoder_for_class(ae, class_data)
    
    return autoencoders

# przewidywanie klas dla zbioru testowego
def make_class_predictions(X_test, autoencoders):
    y_pred = []
    for idx, img in enumerate(X_test): 
        pred_class = classify_image(img, autoencoders)
        y_pred.append(pred_class.numpy())  # konwersja tensora na numpy dla dalszego przetwarzania
        if idx % 100 == 0:
            print(f"Processed {idx} images")

    y_pred = np.array(y_pred) # konwersja listy z przewidywanymi klasami na tablicę numpy
    return y_pred
