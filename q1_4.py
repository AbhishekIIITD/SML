import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pcaQ1 import PCA
from Adaboost import DecisionTree,AdaBoost
# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Filter out 0s and 1s
train_mask = (train_labels == 0) | (train_labels == 1)
test_mask = (test_labels == 0) | (test_labels == 1)

train_images = train_images[train_mask]
train_labels = train_labels[train_mask]
test_images = test_images[test_mask]
test_labels = test_labels[test_mask]

# Convert labels to -1 and 1 
train_labels = np.where(train_labels == 0, -1, 1)
test_labels = np.where(test_labels == 0, -1, 1)


class_0_indices = np.where(train_labels == -1)[0]
class_1_indices = np.where(train_labels == 1)[0]

# Randomly selecting 1000 samples from each class
val_indices_0 = np.random.choice(class_0_indices, 1000, replace=False)
val_indices_1 = np.random.choice(class_1_indices, 1000, replace=False)

# Combine the indices
val_indices = np.concatenate([val_indices_0, val_indices_1])

# Extract validation images and labels
val_images = train_images[val_indices]
val_labels = train_labels[val_indices]


train_images = np.delete(train_images, val_indices, axis=0)
train_labels = np.delete(train_labels, val_indices)

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], -1)
val_images = val_images.reshape(val_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)


#pca
pca=PCA(5)
train_images_pca = pca.fit_transform(train_images)
val_images_pca =  pca.fit_transform(val_images)
test_images_pca = pca.fit_transform(test_images)



print(train_images_pca.shape)
print(val_images_pca.shape)

clf = DecisionTree(max_depth=1)
clf.fit(train_images_pca, train_labels)
def print_tree(node, depth=0):
    if node is None:
        return

    # Indentation based on the depth of the node
    indent = '  ' * depth

    # If the node is a leaf node, print its value
    if node.is_leaf_node():
        print(indent + f"Leaf: {node.value}")
        return

    # Print the decision node
    print(indent + f"Decision: Feature {node.feature} <= {node.threshold}")

    # Recursively print the left and right subtrees
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)
print_tree(clf.root)

ada=AdaBoost()
ada.fit(train_images_pca,train_labels,val_images_pca,val_labels,test_images,test_labels)
print(ada.predict)

