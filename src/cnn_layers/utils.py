import numpy as np

def calculate_output_size(input_size, filter_size, stride=1, padding=0):
    """
    Calculate output size after 2D convolution
    
    Parameters:
    -----------
    input_size : tuple or int
        Input size (height, width) or single value if height = width
    filter_size : tuple or int
        Filter/kernel size (height, width) or single value if height = width
    stride : tuple or int, default=1
        Stride size (height, width) or single value if height = width
    padding : tuple or int, default=0
        Padding size (height, width) or single value if height = width
    
    Returns:
    --------
    tuple
        Output size (output_height, output_width)
    """
    # Convert single values to tuples
    if isinstance(input_size, int):
        input_h = input_w = input_size
    else:
        input_h, input_w = input_size
        
    if isinstance(filter_size, int):
        filter_h = filter_w = filter_size
    else:
        filter_h, filter_w = filter_size
        
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding_h - filter_h) // stride_h + 1
    output_w = (input_w + 2 * padding_w - filter_w) // stride_w + 1
    
    return output_h, output_w

def calculate_pooling_output_size(input_size, pool_size, stride=None):
    """
    Calculate output size after pooling
    
    Parameters:
    -----------
    input_size : tuple or int
        Input size (height, width) or single value if height = width
    pool_size : tuple or int
        Pooling window size (height, width) or single value if height = width
    stride : tuple or int, optional
        Stride size. If None, defaults to pool_size
    
    Returns:
    --------
    tuple
        Output size (output_height, output_width)
    """
    if stride is None:
        stride = pool_size
    
    return calculate_output_size(input_size, pool_size, stride, padding=0)

def softmax(x):
    """
    Softmax activation function
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        Softmax output
    """
    if x.ndim == 2:
        # Batch processing
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        # Single sample
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

def relu(x):
    """
    ReLU activation function
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        ReLU output
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        ReLU derivative
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """
    Sigmoid activation function
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        Sigmoid output
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid activation function
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        Sigmoid derivative
    """
    s = sigmoid(x)
    return s * (1 - s)

def one_hot_encode(y, num_classes):
    """
    One-hot encode labels
    
    Parameters:
    -----------
    y : numpy.ndarray
        Class labels
    num_classes : int
        Number of classes
        
    Returns:
    --------
    numpy.ndarray
        One-hot encoded labels
    """
    return np.eye(num_classes)[y]

def calculate_conv_params(in_channels, out_channels, kernel_size):
    """
    Calculate number of parameters in a convolution layer
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : tuple or int
        Kernel size (height, width) or single value if height = width
    
    Returns:
    --------
    int
        Number of parameters
    """
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Weights: out_channels * in_channels * kernel_h * kernel_w
    # Biases: out_channels
    return out_channels * in_channels * kernel_h * kernel_w + out_channels
