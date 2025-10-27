import numpy as np

class MaxPool2D:
    """
    Max pooling layer - reduces spatial dimensions by taking the maximum value
    in each pooling window. Commonly used to downsample feature maps.
    """
    
    def __init__(self, kernel_size, stride=None):
        """
        Create a max pooling layer.
        
        Parameters:
        -----------
        kernel_size : int or tuple
            Pooling window size (height, width)
        stride : int or tuple, optional
            How far to move the window each step (defaults to kernel_size for non-overlapping)
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
            
        # Cache for backward pass (to store indices of max elements)
        self.max_indices = None
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
        
        # Initialize output and indices storage
        out = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        
        # Perform max pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                h_end = h_start + self.kernel_h
                w_start = j * self.stride_w
                w_end = w_start + self.kernel_w
                
                # Extract window
                window = x[:, :, h_start:h_end, w_start:w_end]  # (batch_size, channels, kernel_h, kernel_w)
                
                # Reshape for finding max indices
                window_reshaped = window.reshape(batch_size, channels, -1)  # (batch_size, channels, kernel_h*kernel_w)
                
                # Find max values and their indices
                max_vals = np.max(window_reshaped, axis=2)
                max_indices_flat = np.argmax(window_reshaped, axis=2)
                
                # Store the results
                out[:, :, i, j] = max_vals
                
                # Convert flat indices back to 2D indices
                self.max_indices[:, :, i, j, 0] = (max_indices_flat // self.kernel_w) + h_start
                self.max_indices[:, :, i, j, 1] = (max_indices_flat % self.kernel_w) + w_start
                
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
        
        # Distribute gradient only to max positions
        for i in range(out_h):
            for j in range(out_w):
                for b in range(batch_size):
                    for c in range(channels):
                        h_idx = self.max_indices[b, c, i, j, 0]
                        w_idx = self.max_indices[b, c, i, j, 1]
                        dx[b, c, h_idx, w_idx] += dout[b, c, i, j]
                        
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
        Get number of parameters in this layer (max pooling has no parameters)
        
        Returns:
        --------
        int
            Total number of parameters (always 0 for pooling)
        """
        return 0
