import numpy as np

class Flatten:
    """
    Flattens multi-dimensional feature maps into a 1D vector.
    Used to transition from convolutional layers to fully connected layers.
    """
    
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        """
        Flatten the input while preserving the batch dimension.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input tensor, typically (batch_size, channels, height, width)
            
        Returns:
        --------
        numpy.ndarray
            Flattened tensor of shape (batch_size, features)
        """
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def backward(self, dout):
        """
        Backward propagation
        
        Parameters:
        -----------
        dout : numpy.ndarray
            Gradient from the next layer of shape (batch_size, flattened_size)
            
        Returns:
        --------
        numpy.ndarray
            Gradient with respect to input of original shape
        """
        return dout.reshape(self.input_shape)
    
    def get_params_count(self):
        """
        Get number of parameters in this layer (flatten has no parameters)
        
        Returns:
        --------
        int
            Total number of parameters (always 0 for flatten)
        """
        return 0
