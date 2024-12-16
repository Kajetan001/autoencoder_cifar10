from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# wykres strat treningowych i walidacyjnych
def make_training_and_validation_loss_graph(autoencoder):
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.history['loss'], label='Strata treningowa', color='blue', linewidth=2)
    plt.plot(autoencoder.history['val_loss'], label='Strata walidacyjna', color='orange', linestyle='--', linewidth=2)

    plt.title("Strata treningowa vs. strata walidacyjna", fontsize=16)
    plt.xlabel("Epoki", fontsize=14)
    plt.ylabel("Strata", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    plt.show()

# wykres oryginalnych i zrekonstruowanych obrazów
def compare_original_and_reconstructed_images(X_test, y_test, predicted, class_names):
    plt.figure(figsize=(20, 4))
    
    # Find the first image for each class
    unique_classes_indices = []
    for i in range(10):
        class_indices = np.where(y_test.flatten() == i)[0]
        unique_classes_indices.append(class_indices[0])
    
    for i, idx in enumerate(unique_classes_indices):
        # Display original image
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Add class name to the top of each original image column
        plt.title(f"Class {i}: {class_names[i]}", fontsize=8)
        
        # Display reconstructed image
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(predicted[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()

# macierz pomyłek
def make_confusion_matrix(y_test, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent cutting off labels
    plt.show()

# classification report
def make_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)])
    print("Classification Report:\n", report)