import numpy as np
from src.cnn_layers import *

class Scratch2dCNNClassifier:
    """
    A 2D CNN classifier built from scratch using only NumPy.
    Handles forward/backward passes, training, and evaluation.
    """
    
    def __init__(self, layers, loss_fn=SoftmaxCrossEntropyLoss(), optimizer=SGD(learning_rate=0.01)):
        """
        Set up the CNN with the given layers and training configuration.
        
        Parameters:
        -----------
        layers : list
            Ordered list of layers in the network
        loss_fn : object
            Loss function to use (default: softmax cross-entropy)
        optimizer : object
            Optimizer for parameter updates (default: SGD with lr=0.01)
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Give each layer a reference to the optimizer
        for layer in layers:
            if hasattr(layer, 'W'):
                layer.optimizer = optimizer
    
    def forward(self, x, training=True):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data of shape (batch_size, channels, height, width)
        training : bool, default=True
            Whether this is during training (affects some layers)
            
        Returns:
        --------
        numpy.ndarray
            Output logits of shape (batch_size, num_classes)
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, x, y):
        """
        Backward pass through the network
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data of shape (batch_size, channels, height, width)
        y : numpy.ndarray
            Target labels of shape (batch_size,)
            
        Returns:
        --------
        float
            The loss for this batch
        """
        # Forward pass
        logits = self.forward(x, training=True)
        
        # Compute loss
        loss = self.loss_fn.forward(logits, y)
        
        # Backward pass
        grad = self.loss_fn.backward(logits, y)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return loss
    
    def predict(self, x):
        """
        Make predictions
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        numpy.ndarray
            Predicted labels of shape (batch_size,)
        """
        logits = self.forward(x, training=False)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, x):
        """
        Make probability predictions
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x, training=False)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs
    
    def evaluate(self, x, y):
        """
        Evaluate the model on given data
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data of shape (num_samples, channels, height, width)
        y : numpy.ndarray
            Target labels of shape (num_samples,)
            
        Returns:
        --------
        tuple
            (loss, accuracy)
        """
        # Forward pass
        logits = self.forward(x, training=False)
        
        # Compute loss
        loss = self.loss_fn.forward(logits, y)
        
        # Compute accuracy
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y)
        
        return loss, accuracy
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, verbose=True):
        """
        Train the model
        
        Parameters:
        -----------
        x_train : numpy.ndarray
            Training data of shape (num_train_samples, channels, height, width)
        y_train : numpy.ndarray
            Training labels of shape (num_train_samples,)
        x_val : numpy.ndarray, optional
            Validation data
        y_val : numpy.ndarray, optional
            Validation labels
        epochs : int, default=10
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        dict
            Training history containing losses and accuracies
        """
        num_samples = x_train.shape[0]
        
        # Initialize history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train in mini-batches
            total_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                x_batch = x_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                
                # Backward pass (includes forward pass and weight update)
                loss = self.backward(x_batch, y_batch)
                total_loss += loss
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            train_loss, train_acc = self.evaluate(x_train, y_train)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation if provided
            if x_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(x_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
        
        return history
    
    def summary(self):
        """
        Print model summary
        """
        print("Model Summary:")
        print("=" * 50)
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = layer.get_params_count()
            total_params += params
            print(f"Layer {i+1}: {layer_name}")
            print(f"  Parameters: {params}")
        print("=" * 50)
        print(f"Total Parameters: {total_params}")
    
    def calculate_num_features(self, input_shape):
        """
        Calculate number of features after conv/pool layers
        
        Parameters:
        -----------
        input_shape : tuple
            Input shape (channels, height, width)
            
        Returns:
        --------
        int
            Number of features before first fully connected layer
        """
        x = np.zeros((1, *input_shape))
        for layer in self.layers:
            x = layer.forward(x)
            if isinstance(layer, Flatten):
                return x.shape[1]
        return None
