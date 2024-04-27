import numpy as np

# Step 1: Accessing the MNIST Dataset
mnist_data = np.load('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
x_train, y_train = mnist_data['x_train'], mnist_data['y_train']

# Step 2: Filtering Data
indices = np.where((y_train == 0) | (y_train == 1) | (y_train == 2))
x_filtered = x_train[indices]
y_filtered = y_train[indices]

# Step 3: Implementing PCA
def pca(X, n_components=10):
    # Calculate covariance matrix
    cov_matrix = np.cov(X.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Choose top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    return components

# Step 4: Transform Data
x_filtered_flat = x_filtered.reshape(x_filtered.shape[0], -1)
pca_matrix = pca(x_filtered_flat)
print(pca_matrix)
