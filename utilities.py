import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def confusion_matrix(actual_classes, predicted_classes):
    # Find unique classes in the output
    classes = [0,1]
    num_classes = len(classes)

    # Initialize the confusion matrix with zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Map each class to an index to fill the confusion matrix
    class_to_index = {label: index for index, label in enumerate(classes)}

    # Populate the confusion matrix
    for actual, predicted in zip(actual_classes, predicted_classes):
        actual_index = class_to_index[actual]
        predicted_index = class_to_index[predicted]
        conf_matrix[actual_index, predicted_index] += 1

    # Display using PrettyTable
    table = PrettyTable()
    table.field_names = ["Actual \\ Predicted"] + [f"{cls}" for cls in classes]
    for actual_class in classes:
        row = [f"{actual_class}"] + list(conf_matrix[class_to_index[actual_class], :])
        table.add_row(row)

    print("Confusion Matrix:")
    print(table)

    return conf_matrix
