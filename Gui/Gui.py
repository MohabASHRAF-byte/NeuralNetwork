import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from Data import Data
from Models.Adaline import AdalineGD
from Models.Perceptron import Perceptron
from utilities import accuracy


def run_model():
    # Get user inputs
    feature1 = feature1_var.get()
    feature2 = feature2_var.get()
    class1_selection = class_var1.get()
    class2_selection = class_var2.get()
    eta = learning_rate_var.get()
    epochs = epochs_var.get()
    mse_threshold = mse_var.get()
    bias = bias_var.get()
    algorithm = algorithm_var.get()

    # Validate inputs
    if not feature1 or not feature2 or not class1_selection or not class2_selection:
        messagebox.showerror("Input Error", "Please select features and both classes.")
        return

    if class1_selection == class2_selection:
        messagebox.showerror("Input Error", "Please select two different classes.")
        return

    try:
        eta = float(eta)
        epochs = int(epochs)
        mse_threshold = float(mse_threshold)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return

    # Generate data and run the model
    data_generator = Data()
    classes = [class1_selection, class2_selection]
    features = [feature1, feature2]

    train_input, train_output, test_input, test_output = data_generator.GenerateDataWithFeatures(classes, features)

    if algorithm == "Adaline":
        model = AdalineGD(learning_rate=eta, epochs=epochs, bias=bias, threshold=mse_threshold)
        weights, bias_value = model.fit(train_input, train_output)
        plot_decision_boundary_adaline(
            train_input, train_output, weights, bias_value, feature1, feature2, classes, model
        )
    else:
        model = Perceptron(learning_rate=eta, epochs=epochs, bias=bias)
        weights, bias_value = model.fit(train_input, train_output)
        plot_decision_boundary_perceptron(
            train_input, train_output, weights, bias_value, feature1, feature2, classes
        )

    predictions = model.predict(test_input)
    confusion = model.confusion_matrix(test_input, test_output)
    acc = accuracy(test_output, predictions)

    # Plot confusion matrix
    plot_confusion_matrix(confusion, classes)

    # Display results
    result = (
        f"Algorithm: {algorithm}\n"
        f"Features: {feature1}, {feature2}\n"
        f"Classes: {class1_selection} & {class2_selection}\n"
        f"Learning Rate: {eta}\n"
        f"Epochs: {epochs}\n"
        f"Accuracy: {acc * 100:.2f}%\n"
        f"Bias: {'Yes' if bias else 'No'}"
    )
    result_label.config(text=result)


def plot_decision_boundary_adaline(X, y, weights, bias, feature1, feature2, classes, model):
    # Transform weights and bias back to original scale
    original_weights = weights / model.training_input_std
    original_bias = bias - np.dot(model.training_input_mean / model.training_input_std, weights)

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Decision Boundary for {classes[0]} vs {classes[1]}")
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    # Plot data points
    for idx, label in enumerate(set(y)):
        ax.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {classes[idx]}")

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.array([x_min, x_max])
    y_vals = -(original_weights[0] * x_vals + original_bias) / original_weights[1]
    ax.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
    ax.legend()
    ax.grid(True)

    # Update plot
    update_plot(plot_frame, fig)


def plot_decision_boundary_perceptron(X, y, weights, bias, feature1, feature2, classes):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Decision Boundary for {classes[0]} vs {classes[1]}")
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    # Plot data points
    for idx, label in enumerate(set(y)):
        ax.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {classes[idx]}")

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.array([x_min, x_max])
    y_vals = -(weights[0] * x_vals + bias) / weights[1]
    ax.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
    ax.legend()
    ax.grid(True)

    # Update plot
    update_plot(plot_frame, fig)


def plot_confusion_matrix(confusion, classes):
    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.7)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center')

    ax.set_title("Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Update plot
    update_plot(confusion_frame, fig)


def update_plot(frame, fig):
    # Clear the previous canvas and plot the new one
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Create the main window
root = tk.Tk()
root.title("Hyperparameter Testing GUI")

# Create a canvas for scrolling
canvas = tk.Canvas(root, highlightthickness=0)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a vertical scrollbar
v_scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas to work with the scrollbar
canvas.configure(yscrollcommand=v_scrollbar.set)

# Add a frame inside the canvas
main_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=main_frame, anchor="nw")


# Function to update the canvas scroll region
def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


# Bind the configure event to update the scroll region
main_frame.bind("<Configure>", update_scroll_region)

# Variables for inputs
feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
class_var1 = tk.StringVar()
class_var2 = tk.StringVar()
learning_rate_var = tk.StringVar()
epochs_var = tk.StringVar()
mse_var = tk.StringVar()
bias_var = tk.BooleanVar()
algorithm_var = tk.StringVar(value="Perceptron")

# Layout widgets inside the main frame
ttk.Label(main_frame, text="Select Features:").grid(row=0, column=0, pady=5, padx=5)
ttk.OptionMenu(main_frame, feature1_var, "gender", "gender", "body_mass", "beak_length", "fin_length",
               "beak_depth").grid(row=0, column=1)
ttk.OptionMenu(main_frame, feature2_var, "body_mass", "gender", "body_mass", "beak_length", "fin_length",
               "beak_depth").grid(row=0, column=2)

ttk.Label(main_frame, text="Select Classes:").grid(row=1, column=0, pady=5, padx=5)
ttk.OptionMenu(main_frame, class_var1, "A", "A", "B", "C").grid(row=1, column=1)
ttk.OptionMenu(main_frame, class_var2, "B", "A", "B", "C").grid(row=1, column=2)

ttk.Label(main_frame, text="Learning Rate (eta):").grid(row=2, column=0, pady=5, padx=5)
ttk.Entry(main_frame, textvariable=learning_rate_var).grid(row=2, column=1)

ttk.Label(main_frame, text="Number of Epochs (m):").grid(row=3, column=0, pady=5, padx=5)
ttk.Entry(main_frame, textvariable=epochs_var).grid(row=3, column=1)

ttk.Label(main_frame, text="MSE Threshold:").grid(row=4, column=0, pady=5, padx=5)
ttk.Entry(main_frame, textvariable=mse_var).grid(row=4, column=1)

ttk.Label(main_frame, text="Use Bias:").grid(row=5, column=0, pady=5, padx=5)
ttk.Checkbutton(main_frame, variable=bias_var).grid(row=5, column=1)

ttk.Label(main_frame, text="Algorithm:").grid(row=6, column=0, pady=5, padx=5)
ttk.OptionMenu(main_frame, algorithm_var, "Perceptron", "Adaline","Perceptron").grid(row=6, column=1)

# Button to run the model
run_button = ttk.Button(main_frame, text="Run Model", command=run_model)
run_button.grid(row=7, column=0, columnspan=3, pady=10)

# Label to display results
result_label = ttk.Label(main_frame, text="Results will be displayed here.", justify=tk.LEFT)
result_label.grid(row=8, column=0, columnspan=3, pady=10)

# Frame for plotting decision boundary
plot_frame = ttk.Frame(main_frame)
plot_frame.grid(row=9, column=0, columnspan=3, pady=10)

# Frame for confusion matrix plot
confusion_frame = ttk.Frame(main_frame)
confusion_frame.grid(row=10, column=0, columnspan=3, pady=10)

# Run the GUI application
root.mainloop()
