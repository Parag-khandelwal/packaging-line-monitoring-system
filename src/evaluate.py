from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred):
    """Evaluates the model and prints the accuracy. Also plots a sns heatmap containg the confusion matrix
    Args:
        y_test: y test cases from dataset
        y_pred: predictions made on X_test"""
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

