import numpy as np

class AveragePool2D:
    """
    2D Average Pooling layer implementation from scratch
    """
    
    def __init__(self, kernel_size, stride=None):
        """
        Initialize AveragePool2D layer
        
        Parameters:
        -----------
        kernel_size : int or tuple
            Size of the pooling window (height, width)
        stride : int or tuple, optional
            Stride of the pooling. If None, defaults to kernel_size
        """
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
            
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
            
        # Cache for backward pass
        self.input_shape = None
        
    def forward(self, x):
        """
        Forward propagation
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        numpy.ndarray
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        self.input_shape = x.shape
        batch_size, channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h - self.kernel_h) // self.stride_h + 1
        out_w = (in_w - self.kernel_w) // self.stride_w + 1
        
        # Initialize output
        out = np.zeros((batch_size, channels, out_h, out_w))
        
        # Perform average pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                h_end = h_start + self.kernel_h
                w_start = j * self.stride_w
                w_end = w_start + self.kernel_w
                
                # Extract window and compute average
                window = x[:, :, h_start:h_end, w_start:w_end]  # (batch_size, channels, kernel_h, kernel_w)
                out[:, :, i, j] = np.mean(window, axis=(2, 3))
                
        return out
    
    def backward(self, dout):
        """
        Backward propagation
        
        Parameters:
        -----------
        dout : numpy.ndarray
            Gradient from the next layer of shape (batch_size, channels, out_height, out_width)
            
        Returns:
        --------
        numpy.ndarray
            Gradient with respect to input of shape (batch_size, channels, in_height, in_width)
        """
        batch_size, channels, out_h, out_w = dout.shape
        _, _, in_h, in_w = self.input_shape
        
        # Initialize gradient
        dx = np.zeros(self.input_shape)
        
        # Distribute gradient evenly over the pooling window
        gradient_per_element = 1.0 / (self.kernel_h * self.kernel_w)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                h_end = h_start + self.kernel_h
                w_start = j * self.stride_w
                w_end = w_start + self.kernel_w
                
                # Distribute gradient evenly over the window
                dx[:, :, h_start:h_end, w_start:w_end] += (
                    dout[:, :, i, j, np.newaxis, np.newaxis] * gradient_per_element
                )
                
        return dx
    
    def calculate_output_size(self, input_h, input_w):
        """
        Calculate output size after pooling
        
        Parameters:
        -----------
        input_h : int
            Input height
        input_w : int
            Input width
            
        Returns:
        --------
        tuple
            (output_height, output_width)
        """
        out_h = (input_h - self.kernel_h) // self.stride_h + 1
        out_w = (input_w - self.kernel_w) // self.stride_w + 1
        return out_h, out_w
    
    def get_params_count(self):
        """
        Get number of parameters in this layer (average pooling has no parameters)
        
        Returns:
        --------
        int
            Total number of parameters (always 0 for pooling)
        """
        return 0
