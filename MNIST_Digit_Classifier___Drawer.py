
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import tkinter as tk
from tkinter import messagebox
from scipy.ndimage import zoom

# Function to load and preprocess MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Function to create CNN model for digit classification
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model with early stopping and learning rate reduction
def train_model(model, x_train, y_train, epochs=50, batch_size=64):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    return history

# Function to save the trained model to a file
def save_model(model, filename='mnist_cnn_model.h5'):
    model.save(filename)
    print(f"Model saved as {filename}")

# Function to load a saved model from a file
def load_saved_model(filename='mnist_cnn_model.h5'):
    if os.path.exists(filename):
        model = load_model(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"No saved model found at {filename}")
        return None

# Function to predict digit from a given image sample
def predict_digit(model, sample):
    sample = sample.reshape(1, 28, 28, 1)
    prediction = model.predict(sample)
    return np.argmax(prediction)

# Class for interactive drawing and digit prediction
class DrawingClassifier:
    def __init__(self, model):
        self.model = model
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.canvas = self.fig.canvas
        self.drawing_res = 280  # 10x the original MNIST resolution
        self.drawing = np.zeros((self.drawing_res, self.drawing_res), dtype=np.uint8)
        self.ax1.set_title("Draw a digit here")
        self.ax2.set_title("Model prediction")
        self.im = self.ax1.imshow(self.drawing, cmap='gray_r', vmin=0, vmax=255, extent=[0, 28, 28, 0])
        self.ax1.set_xlim(0, 28)
        self.ax1.set_ylim(28, 0)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.pressed = False
        
        # Create Tkinter window for buttons
        self.root = tk.Tk()
        self.root.title("Controls")
        self.root.geometry("200x100")  # Set a size for the Tkinter window
        
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=5)
        
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

    # Event handler for mouse button press
    def on_press(self, event):
        self.pressed = True
        self.draw(event)

    # Event handler for mouse motion
    def on_motion(self, event):
        if self.pressed:
            self.draw(event)

    # Event handler for mouse button release
    def on_release(self, event):
        self.pressed = False

    # Function to draw on the canvas
    def draw(self, event):
        if event.inaxes == self.ax1:
            x, y = int(event.xdata * 10), int(event.ydata * 10)
            brush_size = 20
            y_min, y_max = max(0, y - brush_size), min(self.drawing_res, y + brush_size)
            x_min, x_max = max(0, x - brush_size), min(self.drawing_res, x + brush_size)
            
            y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
            mask = (x_coords - x)**2 + (y_coords - y)**2 <= brush_size**2
            
            self.drawing[y_min:y_max, x_min:x_max][mask] = 255
            self.im.set_data(zoom(self.drawing, 0.1))
            self.canvas.draw_idle()

    # Function to clear the canvas
    def clear(self):
        self.drawing.fill(0)
        self.im.set_data(zoom(self.drawing, 0.1))
        self.ax2.clear()
        self.ax2.set_title("Model prediction")
        self.canvas.draw_idle()

    # Function to predict the drawn digit
    def predict(self):
        img = zoom(self.drawing, 28/self.drawing_res)
        img = img.reshape(28, 28, 1).astype('float32') / 255
        prediction = predict_digit(self.model, img)
        self.ax2.clear()
        self.ax2.text(0.5, 0.5, str(prediction), fontsize=90, ha='center', va='center')
        self.ax2.set_title("Predicted digit")
        self.canvas.draw_idle()

    # Function to run the Tkinter interface
    def run(self):
        plt.show(block=False)
        self.root.mainloop()

# Function to launch menu for training or testing the model
def launch_menu():
    root = tk.Tk()
    root.title("MNIST Classifier")
    root.geometry("300x150")  # Set a larger initial size for the window
    
    # Function to run test with existing model
    def run_test():
        root.destroy()
        model = load_saved_model()
        if model:
            classifier = DrawingClassifier(model)
            classifier.run()
        else:
            messagebox.showerror("Error", "No saved model found. Please train the model first.")
    
    # Function to start training a new model
    def start_training():
        root.destroy()
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
        model = create_model()
        history = train_model(model, x_train, y_train)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f'Test accuracy: {test_acc}')
        save_model(model)
        classifier = DrawingClassifier(model)
        classifier.run()
    
    # Create buttons in Tkinter window
    tk.Button(root, text="Run Test", command=run_test, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Start Training", command=start_training, width=20, height=2).pack(pady=10)
    
    root.mainloop()

# Main entry point of the script
if __name__ == "__main__":
    launch_menu()
