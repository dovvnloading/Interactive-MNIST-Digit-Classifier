# Interactive-MNIST-Digit-Classifier

MNIST Digit Classifier & Drawer is a versatile application designed to engage users in the world of machine 
learning and neural networks. Built around a convolutional neural network (CNN) trained on the MNIST dataset, 
the app allows users to draw handwritten digits directly onto a canvas interface. Upon completion, 
the model instantly predicts the drawn digit with high accuracy. This interactive experience not only showcases 
the capabilities of modern AI technology but also serves as an educational tool for understanding image 
classification and neural network operations. Whether for educational purposes or fun experimentation, 
MNIST Digit Classifier & Drawer provides an intuitive and insightful glimpse into the workings of machine learning.

*** I've included a trained model. To use it - simply place the file in the same location as your script. 
The Mnist dataset will automatically download to a folder within the script location. This will only occur once, 
unless the dataset is moved.
______________________________________________________________________________
______________________________________________________________________________
# Structuring 

1. Import Statements
Purpose: Imports necessary libraries and modules for machine learning (TensorFlow, Keras), data handling (NumPy),
plotting (Matplotlib), user interface (Tkinter), file system operations (os), and image manipulation (scipy.ndimage.zoom).
Modules Used:
TensorFlow and Keras: For building and training neural networks.
NumPy: For numerical operations on data arrays.
Matplotlib: For plotting graphs and images.
Tkinter: For creating the graphical user interface.
os: For file operations, like checking if a file exists.
scipy.ndimage.zoom: For resizing images (used for the drawing canvas).
__________________________________________________________
3. load_and_preprocess_data Function
Purpose: Loads and preprocesses the MNIST dataset.
Steps:
Loads the MNIST dataset using mnist.load_data().
Normalizes pixel values to the range [0, 1] by dividing by 255.
Reshapes the data to fit the Conv2D input shape (28x28 pixels, 1 channel).
Converts labels to categorical format using to_categorical.
__________________________________________________________
5. create_model Function
Purpose: Defines a Convolutional Neural Network (CNN) model for digit classification.
Layers:
Conv2D layers with ReLU activation.
BatchNormalization layers to normalize layer inputs.
MaxPooling2D layers for down-sampling.
Flatten layer to convert 2D output to 1D.
Dense layers with ReLU activation for classification.
Dropout layer for regularization.
Output Dense layer with softmax activation for multi-class classification.
Compilation: Configures model with Adam optimizer, categorical cross-entropy loss, and accuracy metric.
__________________________________________________________
7. train_model Function
Purpose: Trains the CNN model on the MNIST dataset.
Callbacks:
EarlyStopping: Stops training when validation loss stops improving.
ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
Training: Fits the model on training data, validates on a subset, and monitors validation loss and accuracy.
__________________________________________________________
9. save_model and load_saved_model Functions
save_model:
Purpose: Saves the trained model to a specified file.
Usage: Called after training to save the model for future use.
load_saved_model:
Purpose: Loads a previously saved model from a file.
Usage: Used to load a trained model for prediction without retraining.
__________________________________________________________
11. predict_digit Function
Purpose: Predicts the digit from a given image sample using the trained model.
Input: Takes a single image sample reshaped to (28, 28, 1).
Output: Returns the predicted digit as a numerical value.
__________________________________________________________
13. DrawingClassifier Class
Purpose: Implements an interactive interface for drawing digits and predicting them using the trained model.
Components:
Initialization: Sets up Matplotlib figures for drawing and prediction display, creates a Tkinter window for UI controls.
Event Handlers: Handles mouse events for drawing on the canvas.
Drawing and Prediction: Methods for drawing, clearing the canvas, and predicting the drawn digit.
Run Method: Starts the Tkinter main loop to run the interactive interface.
__________________________________________________________
15. launch_menu Function
Purpose: Provides a graphical menu interface to either start training a new model or run a test using a saved model.
Buttons: Creates buttons in a Tkinter window for 'Run Test' and 'Start Training'.
Functionality:
Run Test: Loads a saved model and runs the DrawingClassifier interface for interactive testing.
Start Training: Loads data, creates a new model, trains it, evaluates on test data, saves the model,
and then runs the DrawingClassifier interface.
__________________________________________________________
17. Main Execution Block (if __name__ == "__main__")
Purpose: Entry point of the script.
Function Call: Calls launch_menu() to start the graphical menu interface for user interaction.
______________________________________________________________________________
______________________________________________________________________________





This script is designed to build, train, and interact with a Convolutional Neural Network (CNN) 
model for handwritten digit recognition using the MNIST dataset. Here’s a summary of its key functionalities:

Data Loading and Preprocessing:
__________________________________________________________
Loads the MNIST dataset.
Preprocesses the images by normalizing pixel values and reshaping for CNN input.
Model Definition:
__________________________________________________________
Defines a CNN model architecture using TensorFlow/Keras.
Includes convolutional layers, pooling layers, batch normalization, dropout for regularization, 
and dense layers with softmax activation for classification.
Training and Evaluation:
__________________________________________________________
Trains the CNN model on the MNIST dataset.
Uses early stopping and learning rate reduction techniques for efficient training.
Evaluates the model on a test set and prints accuracy.
Model Saving and Loading:
__________________________________________________________
Saves the trained model to a file (mnist_cnn_model.h5).
Loads a saved model from file if available.
Interactive Drawing and Prediction:
__________________________________________________________
Allows users to draw digits using matplotlib.
Predicts the drawn digit in real-time using the trained CNN model.
Includes functionalities to clear the drawing canvas and display predictions.
User Interface:
__________________________________________________________
Provides a Tkinter-based menu for users to choose between testing the model with interactive drawing or training a new model.
__________________________________________________________
# To use:
__________________________________________________________
Run launch_menu() to either test an existing model or train a new one.
Follow on-screen prompts for drawing and predicting digits interactively.
