import numpy as np

class SoftmaxCrossEntropyLoss:
    """
    Softmax with Cross-Entropy Loss implementation
    """
    
    def __init__(self):
        """
        Initialize Softmax Cross-Entropy Loss
        """
        self.probs = None
        self.grad = None
    
    def forward(self, logits, targets):
        """
        Forward pass
        
        Parameters:
        -----------
        logits : numpy.ndarray
            Output from the final layer (before softmax) of shape (batch_size, num_classes)
        targets : numpy.ndarray
            True labels as indices of shape (batch_size,) or one-hot encoded of shape (batch_size, num_classes)
            
        Returns:
        --------
        float
            Average loss over the batch
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = np.zeros((batch_size, num_classes))
            targets_one_hot[np.arange(batch_size), targets] = 1
            targets = targets_one_hot
        
        # Apply softmax for numerical stability
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(self.probs + 1e-8)) / batch_size
        
        return loss
    
    def backward(self, logits, targets):
        """
        Backward pass
        
        Parameters:
        -----------
        logits : numpy.ndarray
            Output from the final layer (before softmax) of shape (batch_size, num_classes)
        targets : numpy.ndarray
            True labels as indices of shape (batch_size,) or one-hot encoded of shape (batch_size, num_classes)
            
        Returns:
        --------
        numpy.ndarray
            Gradient with respect to logits of shape (batch_size, num_classes)
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = np.zeros((batch_size, num_classes))
            targets_one_hot[np.arange(batch_size), targets] = 1
            targets = targets_one_hot
        
        # Gradient of cross-entropy with softmax
        grad = (self.probs - targets) / batch_size
        
        return grad
