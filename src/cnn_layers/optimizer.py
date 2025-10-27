import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer implementation
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for updates
        """
        self.learning_rate = learning_rate
    
    def update(self, layer):
        """
        Update layer parameters using SGD
        
        Parameters:
        -----------
        layer : object
            Layer object with W, b, dW, db attributes
        """
        if hasattr(layer, 'W') and hasattr(layer, 'dW'):
            layer.W -= self.learning_rate * layer.dW
        if hasattr(layer, 'b') and hasattr(layer, 'db'):
            layer.b -= self.learning_rate * layer.db

class Adam:
    """
    Adam optimizer implementation (advanced)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Parameters:
        -----------
        learning_rate : float, default=0.001
            Learning rate for updates
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates
        epsilon : float, default=1e-8
            Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
    def update(self, layer):
        """
        Update layer parameters using Adam
        
        Parameters:
        -----------
        layer : object
            Layer object with W, b, dW, db attributes
        """
        self.t += 1
        
        # Initialize moment estimates if not exist
        if not hasattr(layer, 'mW'):
            layer.mW = np.zeros_like(layer.W)
            layer.vW = np.zeros_like(layer.W)
            layer.mb = np.zeros_like(layer.b)
            layer.vb = np.zeros_like(layer.b)
        
        # Update weights
        if hasattr(layer, 'W') and hasattr(layer, 'dW'):
            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
            
            mW_corrected = layer.mW / (1 - self.beta1 ** self.t)
            vW_corrected = layer.vW / (1 - self.beta2 ** self.t)
            
            layer.W -= self.learning_rate * mW_corrected / (np.sqrt(vW_corrected) + self.epsilon)
        
        # Update biases
        if hasattr(layer, 'b') and hasattr(layer, 'db'):
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)
            
            mb_corrected = layer.mb / (1 - self.beta1 ** self.t)
            vb_corrected = layer.vb / (1 - self.beta2 ** self.t)
            
            layer.b -= self.learning_rate * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon)
