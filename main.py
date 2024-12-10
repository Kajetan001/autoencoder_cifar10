from keras.layers import Input

from autoencoder import *
from visualisations import *

if __name__ == "__main__" :
    X_test, X_train = load_and_preprocess()

    input_img = Input(shape=(32, 32, 3))

    autoencoder = compile_autoencoder(input_img)
    history, predicted = fit_and_predict_autoencoder(autoencoder, X_test, X_train, 50)
    
    make_training_and_validation_loss_graph(history)
    compare_original_and_reconstructed_images(X_test, predicted)