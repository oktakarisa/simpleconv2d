import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class MNISTDataLoader:
    """
    Handles loading and preprocessing MNIST handwritten digit data.
    """
    
    def __init__(self, data_dir='data', normalize=True, one_hot=False):
        """
        Set up the data loader.
        
        Parameters:
        -----------
        data_dir : str, default='data'
            Where to cache downloaded data
        normalize : bool, default=True
            If True, scales pixel values to have zero mean and unit variance
        one_hot : bool, default=False
            If True, converts labels to one-hot vectors
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.one_hot = one_hot
        self.scaler = StandardScaler()
        
    def load_data(self, test_size=0.2, random_state=42, use_cache=True):
        """
        Load and preprocess MNIST data
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Fraction of data to use for testing
        random_state : int, default=42
            Random seed for reproducibility
        use_cache : bool, default=True
            Whether to use cached data if available
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Check if cached data exists
        cache_file = os.path.join(self.data_dir, 'mnist_processed.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print("Loading cached MNIST data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        print("Loading MNIST data...")
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        
        # Reshape to (n_samples, 1, 28, 28) for CNN
        X = X.reshape(-1, 1, 28, 28).astype(np.float64)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize if requested
        if self.normalize:
            # Reshape for fitting the scaler
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            
            # Fit on training data and transform both train and test
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            
            # Reshape back
            X_train = X_train_scaled.reshape(X_train.shape)
            X_test = X_test_scaled.reshape(X_test.shape)
        
        # Convert to one-hot if requested
        if self.one_hot:
            y_train = self._one_hot_encode(y_train, 10)
            y_test = self._one_hot_encode(y_test, 10)
        
        # Cache the processed data
        if use_cache:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }, f)
        
        print(f"Data loaded and preprocessed:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _one_hot_encode(self, y, num_classes):
        """
        One-hot encode labels
        
        Parameters:
        -----------
        y : numpy.ndarray
            Labels
        num_classes : int
            Number of classes
            
        Returns:
        --------
        numpy.ndarray
            One-hot encoded labels
        """
        return np.eye(num_classes)[y]
    
    def create_mini_batches(self, X, y, batch_size=32, shuffle=True):
        """
        Create mini-batches from data
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            Labels
        batch_size : int, default=32
            Size of each batch
        shuffle : bool, default=True
            Whether to shuffle data before creating batches
            
        Returns:
        --------
        generator
            Generator yielding (X_batch, y_batch) pairs
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]
