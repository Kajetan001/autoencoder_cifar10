import tensorflow as tf
import numpy as np


def train_autoencoder_for_class(autoencoder, class_data, epochs=50, batch_size=256):
    autoencoder.fit(class_data, class_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

@tf.function(reduce_retracing=True)
def classify_image(image, autoencoders):
    errors = []
    image = tf.expand_dims(image, axis=0)  # Dodanie wymiaru batch
    for ae in autoencoders:
        reconstruction = ae(image, training=False)  # Bez predict
        error = tf.reduce_mean(tf.square(image - reconstruction))
        errors.append(error)
    return tf.argmin(errors)

def train_all_autoencoders(X_train, y_train, autoencoders):
    # Trenowanie autoenkoderów dla każdej klasy
    for i, ae in enumerate(autoencoders):
        print(f"Training autoencoder for class {i}")
        class_data = X_train[y_train.flatten() == i]
        train_autoencoder_for_class(ae, class_data)
    
    return autoencoders

# Testowanie klasyfikacji
def make_class_predictions(X_test, autoencoders):
    y_pred = []
    for idx, img in enumerate(X_test): 
        pred_class = classify_image(img, autoencoders)
        y_pred.append(pred_class.numpy())  # Konwersja tensora na numpy dla dalszego przetwarzania
        if idx % 100 == 0:
            print(f"Processed {idx} images")

    y_pred = np.array(y_pred)
    return y_pred