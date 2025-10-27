import numpy as np

class FullyConnected:
    """
    Standard fully connected (dense) layer: y = Wx + b
    """
    
    def __init__(self, in_features, out_features, initializer=None, optimizer=None):
        """
        Create a fully connected layer.
        
        Parameters:
        -----------
        in_features : int
            Number of input neurons
        out_features : int
            Number of output neurons
        initializer : object, optional
            Custom weight initializer
        optimizer : object, optional
            Optimizer for weight updates
        """
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = optimizer
        
        # Initialize weights and biases
        if initializer is None:
            # Simple initialization
            self.W = np.random.randn(in_features, out_features).astype(np.float64) * 0.01
            self.b = np.zeros(out_features, dtype=np.float64)
        else:
            self.W = initializer.initialize((in_features, out_features)).astype(np.float64)
            self.b = np.zeros(out_features, dtype=np.float64)
            
        # Gradients
        self.dW = None
        self.db = None
        
        # Cache for backward pass
        self.x = None
        
    def forward(self, x):
        """
        Forward propagation
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input tensor of shape (batch_size, in_features)
            
        Returns:
        --------
        numpy.ndarray
            Output tensor of shape (batch_size, out_features)
        """
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        """
        Backward propagation
        
        Parameters:
        -----------
        dout : numpy.ndarray
            Gradient from the next layer of shape (batch_size, out_features)
            
        Returns:
        --------
        numpy.ndarray
            Gradient with respect to input of shape (batch_size, in_features)
        """
        batch_size = dout.shape[0]
        
        # Gradient w.r.t. weights and biases
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        # Gradient w.r.t. input
        dx = np.dot(dout, self.W.T)
        
        # Update parameters if optimizer is provided
        if self.optimizer is not None:
            self.optimizer.update(self)
            
        return dx
    
    def get_params_count(self):
        """
        Get number of parameters in this layer
        
        Returns:
        --------
        int
            Total number of parameters
        """
        return np.prod(self.W.shape) + np.prod(self.b.shape)
