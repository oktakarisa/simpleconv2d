import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from src.data_loader import MNISTDataLoader
from src.simpleconv2d_classifier import Scratch2dCNNClassifier
from src.cnn_layers import (
    Conv2d, MaxPool2D, Flatten, FullyConnected, 
    SoftmaxCrossEntropyLoss, SGD, Adam, 
    relu, relu_derivative, calculate_output_size
)

class ReLULayer:
    """
    ReLU (Rectified Linear Unit) activation: f(x) = max(0, x)
    """
    
    def __init__(self):
        self.input = None
    
    def forward(self, x, training=True):
        # Save input for backward pass
        self.input = x
        return relu(x)
    
    def backward(self, dout):
        # Gradient is 1 where input > 0, else 0
        return relu_derivative(self.input) * dout
    
    def get_params_count(self):
        return 0  # No learnable parameters

def create_simple_cnn():
    """
    Build a basic CNN for MNIST digit classification.
    Uses two conv blocks followed by fully connected layers.
    """
    layers = [
        # First conv block: 1 -> 8 channels
        Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
        ReLULayer(),
        MaxPool2D(kernel_size=2, stride=2),  # 28x28 -> 14x14
        
        # Second conv block: 8 -> 16 channels
        Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        ReLULayer(),
        MaxPool2D(kernel_size=2, stride=2),  # 14x14 -> 7x7
        
        # Flatten 16*7*7 = 784 features
        Flatten(),
        
        # Hidden layer
        FullyConnected(in_features=16*7*7, out_features=128),
        ReLULayer(),
        
        # Output layer (10 digits)
        FullyConnected(in_features=128, out_features=10)
    ]
    
    return layers

def create_lenet():
    """
    Build the classic LeNet architecture (Yann LeCun, 1998).
    Modified to use ReLU instead of sigmoid and max pooling instead of average pooling.
    """
    layers = [
        # Conv block 1: 1 -> 6 channels, 28x28 -> 24x24 -> 12x12
        Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
        ReLULayer(),
        MaxPool2D(kernel_size=2, stride=2),
        
        # Conv block 2: 6 -> 16 channels, 12x12 -> 8x8 -> 4x4
        Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
        ReLULayer(),
        MaxPool2D(kernel_size=2, stride=2),
        
        # Flatten and dense layers
        Flatten(),  # 16*4*4 = 256 features
        
        FullyConnected(in_features=16*4*4, out_features=120),
        ReLULayer(),
        
        FullyConnected(in_features=120, out_features=84),
        ReLULayer(),
        
        # Output for 10 digit classes
        FullyConnected(in_features=84, out_features=10)
    ]
    
    return layers

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Training history dictionary
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def predict_and_visualize(model, X_test, y_test, num_samples=5):
    """
    Make predictions and visualize some results
    
    Parameters:
    -----------
    model : Scratch2dCNNClassifier
        Trained model
    X_test : numpy.ndarray
        Test images
    y_test : numpy.ndarray
        Test labels
    num_samples : int, default=5
        Number of samples to visualize
    """
    # Make predictions
    predictions = model.predict(X_test[:num_samples])
    probabilities = model.predict_proba(X_test[:num_samples])
    
    # Plot results
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        
        # Display image
        ax.imshow(X_test[i, 0], cmap='gray')
        ax.axis('off')
        
        # Add title with prediction
        pred_label = predictions[i]
        true_label = y_test[i]
        confidence = np.max(probabilities[i])
        color = 'green' if pred_label == true_label else 'red'
        
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}', 
                    color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main training and evaluation function
    """
    print("=" * 50)
    print("Simple 2D CNN Classifier - MNIST Training")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("\n1. Loading MNIST data...")
    data_loader = MNISTDataLoader(data_dir='data')
    X_train, X_test, y_train, y_test = data_loader.load_data(test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Final training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Choose architecture
    use_lenet = input("\nUse LeNet architecture? (y/n, default=n): ").strip().lower() == 'y'
    
    if use_lenet:
        print("\n2. Creating LeNet architecture...")
        layers = create_lenet()
    else:
        print("\n2. Creating simple CNN architecture...")
        layers = create_simple_cnn()
    
    # Create model
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=0.01)  # Can also use Adam(learning_rate=0.001)
    model = Scratch2dCNNClassifier(layers, loss_fn, optimizer)
    
    # Print model summary
    model.summary()
    
    # Training parameters
    epochs = int(input("\nEnter number of epochs (default=10): ").strip() or "10")
    batch_size = int(input("Enter batch size (default=32): ").strip() or "32")
    
    # Train model
    print(f"\n3. Training model for {epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history, save_path='plots/training_history.png')
    
    # Visualize some predictions
    print("\n6. Visualizing some predictions...")
    predict_and_visualize(model, X_test, y_test, num_samples=5)
    
    # Save results summary
    results = {
        'model_type': 'LeNet' if use_lenet else 'Simple CNN',
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    print("\n" + "=" * 50)
    print("Training Summary:")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    main()
