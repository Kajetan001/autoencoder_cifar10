from keras.layers import Input

from autoencoder import *
from ae_classifier import *
from visualisations import *


if __name__ == "__main__" :
    # załadowanie i wstępne przetworzenie danych, dodanie listy nazw klas dla danych cifar-10
    X_test, X_train, y_test, y_train = load_and_preprocess()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


    # definicja warstwy wejściowej dla autoenkoderów
    # wejściowy obraz ma kształt 32x32 pikseli z 3 kanałami (rgb)
    input_img = Input(shape=(32, 32, 3))
    
    # inicjalizacja głównego autoenkodera
    autoencoder = build_autoencoder(input_img)
    
    # inicjalizacja listy zawierającej 10 autoenkoderów, gdzie każdy autoenkoder odpowiada za jedną klasę
    class_autoencoders = [build_autoencoder(input_img) for _ in range(10)]

    # trenowanie głównego autoenkodera na danych treningowych
    # reconstructed_images: obrazy odtworzone przez autoenkoder
    autoencoder, reconstructed_images = fit_and_predict_autoencoder(autoencoder, X_test, X_train)

    # trenowanie 10 autoenkoderów na danych dla odpowiednich klas
    class_autoencoders = train_all_autoencoders(X_train, y_train, class_autoencoders)

    # klasyfikacja oryginalnych obrazów testowych za pomocą autoenkoderów
    # zwracane są przewidywane klasy dla każdego obrazu testowego
    class_predictions = make_class_predictions(X_test, class_autoencoders)

    # klasyfikacja obrazów zrekonstruowanych przez główny autoenkoder
    # zwracane są przewidywane klasy dla obrazów po rekonstrukcji
    reconstructed_class_predictions = make_class_predictions(reconstructed_images, class_autoencoders)
    
    # wizualizacja wykresu utraty funkcji kosztu dla zbioru treningowego i walidacyjnego
    make_training_and_validation_loss_graph(autoencoder)

    # porównanie oryginalnych obrazów z obrazami odtworzonymi przez autoenkoder
    compare_original_and_reconstructed_images(X_test, y_test, reconstructed_images, class_names)

    # tworzenie i wizualizacja macierzy pomyłek dla obrazów oraz generowanie raportu klasyfikacji
    print("Klasyfikacja oryginalnych obrazów")
    make_confusion_matrix(y_test, class_predictions, class_names, "Klasyfikacja oryginalnych obrazów")
    make_classification_report(y_test, class_predictions)

    print("Klasyfikacja obrazów po autoencoderze")
    make_confusion_matrix(y_test, reconstructed_class_predictions, class_names, "Klasyfikacja obrazów po autoencoderze")
    make_classification_report(y_test, reconstructed_class_predictions)
