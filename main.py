from keras.layers import Input

from autoencoder import *
from ae_classifier import *
from visualisations import *


if __name__ == "__main__" :
    X_test, X_train, y_test, y_train = load_and_preprocess()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    input_img = Input(shape=(32, 32, 3))
    
    # inicjalizacja autoenkodera oraz 10 osobnych autoenkoderów do pózniejszej klasyfikacji
    autoencoder = build_autoencoder(input_img)
    class_autoencoders = [build_autoencoder(input_img) for _ in range(10)]

    # trenowanie autoenkoderów
    autoencoder, reconstructed_images = fit_and_predict_autoencoder(autoencoder, X_test, X_train)
    class_autoencoders = train_all_autoencoders(X_train, y_train, class_autoencoders)

    class_predictions = make_class_predictions(X_test, class_autoencoders)
    reconstructed_class_predictions = make_class_predictions(reconstructed_images, class_autoencoders)
    
    make_training_and_validation_loss_graph(autoencoder)
    compare_original_and_reconstructed_images(X_test, y_test, reconstructed_images, class_names)

    print("Klasyfikacja oryginalnych obrazów")
    make_confusion_matrix(y_test, class_predictions, class_names, "Klasyfikacja oryginalnych obrazów")
    make_classification_report(y_test, class_predictions)

    print("Klasyfikacja obrazów po autoencoderze")
    make_confusion_matrix(y_test, reconstructed_class_predictions, class_names, "Klasyfikacja obrazów po autoencoderze")
    make_classification_report(y_test, reconstructed_class_predictions)