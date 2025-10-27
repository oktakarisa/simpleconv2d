import numpy as np

class Conv2d:
    """
    2D convolutional layer. Applies learnable filters to extract spatial features.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initializer=None, optimizer=None):
        """
        Create a new convolutional layer.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        out_channels : int
            Number of filters to learn (becomes output channel count)
        kernel_size : int or tuple
            Filter dimensions (height, width)
        stride : int or tuple, default=1
            How many pixels to skip when sliding the filter
        padding : int or tuple, default=0
            How many zeros to pad around the input
        initializer : object, optional
            Custom weight initializer
        optimizer : object, optional
            Optimizer for updating weights
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
            
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
            
        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
            
        self.optimizer = optimizer
        
        # Initialize weights and biases
        if initializer is None:
            # Simple initialization
            self.W = np.random.randn(out_channels, in_channels, self.kernel_h, self.kernel_w).astype(np.float64) * 0.01
            self.b = np.zeros(out_channels, dtype=np.float64)
        else:
            self.W = initializer.initialize((out_channels, in_channels, self.kernel_h, self.kernel_w)).astype(np.float64)
            self.b = np.zeros(out_channels, dtype=np.float64)
            
        # Gradients
        self.dW = None
        self.db = None
        
        # Cache for backward pass
        self.x = None
        
    def forward(self, x):
        """
        Convolve the input with learned filters.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input, shape (batch_size, in_channels, height, width)
            
        Returns:
        --------
        numpy.ndarray
            Convolved output, shape (batch_size, out_channels, out_height, out_width)
        """
        self.x = x
        batch_size, in_channels, in_h, in_w = x.shape
        
        # Figure out output size based on padding and stride
        out_h = (in_h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (in_w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        
        # Pad the input if needed
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant')
        
        # Allocate output array
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Slide the filter across the input
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                h_end = h_start + self.kernel_h
                w_start = j * self.stride_w
                w_end = w_start + self.kernel_w
                
                # Extract the region we're convolving over
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Apply each filter
                for c_out in range(self.out_channels):
                    # Element-wise multiply and sum, then add bias
                    out[:, c_out, i, j] = np.sum(x_slice * self.W[c_out], axis=(1, 2, 3)) + self.b[c_out]
                    
        return out
    
    def backward(self, dout):
        """
        Backpropagate gradients through the layer.
        
        Parameters:
        -----------
        dout : numpy.ndarray
            Gradient from next layer, shape (batch_size, out_channels, out_h, out_w)
            
        Returns:
        --------
        numpy.ndarray
            Gradient w.r.t. input, shape (batch_size, in_channels, in_h, in_w)
        """
        batch_size, out_channels, out_h, out_w = dout.shape
        _, in_channels, in_h, in_w = self.x.shape
        
        # Apply padding to input for gradient calculation
        x_padded = np.pad(self.x, ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant')
        
        # Initialize gradients
        dx_padded = np.zeros_like(x_padded, dtype=np.float64)
        self.dW = np.zeros_like(self.W, dtype=np.float64)
        self.db = np.zeros_like(self.b, dtype=np.float64)
        
        # Calculate gradients
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                h_end = h_start + self.kernel_h
                w_start = j * self.stride_w
                w_end = w_start + self.kernel_w
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                for c_out in range(out_channels):
                    # Gradient w.r.t. bias
                    self.db[c_out] += np.sum(dout[:, c_out, i, j])
                    
                    # Gradient w.r.t. weights
                    weight_grad = np.zeros_like(self.dW[c_out])
                    for b in range(batch_size):
                        weight_grad += dout[b, c_out, i, j] * x_slice[b, 0]  # Remove channel dim
                    self.dW[c_out] += weight_grad
                    
                    # Gradient w.r.t. input
                    for b in range(batch_size):
                        dx_padded[b, :, h_start:h_end, w_start:w_end] += dout[b, c_out, i, j] * self.W[c_out]
        
        # Remove padding from dx
        if self.pad_h > 0 or self.pad_w > 0:
            dx = dx_padded[:, :, self.pad_h:-self.pad_h, self.pad_w:-self.pad_w]
        else:
            dx = dx_padded
            
        # Update parameters if optimizer is provided
        if self.optimizer is not None:
            self.optimizer.update(self)
            
        return dx
    
    def calculate_output_size(self, input_h, input_w):
        """
        Calculate output size after convolution
        
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
        out_h = (input_h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (input_w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        return out_h, out_w
    
    def get_params_count(self):
        """
        Get number of parameters in this layer
        
        Returns:
        --------
        int
            Total number of parameters
        """
        return np.prod(self.W.shape) + np.prod(self.b.shape)
