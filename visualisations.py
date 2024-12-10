import matplotlib.pyplot as plt

# wykres strat treningowych i walidacyjnych
def make_training_and_validation_loss_graph(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Strata treningowa', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Strata walidacyjna', color='orange', linestyle='--', linewidth=2)

    plt.title("Strata treningowa vs. strata walidacyjna", fontsize=16)
    plt.xlabel("Epoki", fontsize=14)
    plt.ylabel("Strata", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    plt.show()

# wykres oryginalnych i zrekonstruowanych obrazów
def compare_original_and_reconstructed_images(X_test, predicted):
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # wyświetlenie oryginalnego obrazu
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Oryginał", fontsize=12)

        # wyświetlenie zrekonstruowanego obrazu
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(predicted[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Zrekonstruowany", fontsize=12)

    plt.show()