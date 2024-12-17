from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# wykres strat treningowych i walidacyjnych autoencodera
def make_training_and_validation_loss_graph(autoencoder):
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.history['loss'], label='Strata treningowa', color='blue', linewidth=2)
    plt.plot(autoencoder.history['val_loss'], label='Strata walidacyjna', color='orange', linestyle='--', linewidth=2)
    # tytuł i etykiety
    plt.title("Strata treningowa vs. strata walidacyjna", fontsize=16)
    plt.xlabel("Epoki", fontsize=14)
    plt.ylabel("Strata", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    plt.show()

# wykres oryginalnych i zrekonstruowanych obrazów
def compare_original_and_reconstructed_images(X_test, y_test, predicted, class_names):
    plt.figure(figsize=(20, 4))
    
    # znalezienie pierwszego indeksu pod którym występuje każda z 10 klas
    unique_classes_indices = []
    for i in range(10):
        class_indices = np.where(y_test.flatten() == i)[0]
        unique_classes_indices.append(class_indices[0])

    # iterowanie po indeksach unikalnych klas
    for i, idx in enumerate(unique_classes_indices):
        # wyświetlanie oryginalnych obrazów
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # podpisanie każdego obrazu nazwą klasy
        plt.title(f"Class {i}: {class_names[i]}", fontsize=8)
        
        # wyświetlanie obrazów zrekonstruowanych
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(predicted[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()

# rysowanie macierzy pomyłek dla
def make_confusion_matrix(y_test, y_pred, class_names, title='Macierz Pomyłek'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # etykiety i tytuł
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Faktyczna klasa')
    plt.title(title)
    # obrót etykiet dla czytelności
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() 
    plt.show()

# classification report
def make_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)])
    print("Classification Report:\n", report)
