from collections import Counter
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

from pcaQ1 import PCA

np.set_printoptions(threshold=sys.maxsize)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
   
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gini = float('inf')
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            #thresholds = np.unique(X_column)
            thresholds=np.median(np.sort(X_column))
            gini = self._gini_index(y, X_column, thresholds)
            if gini < best_gini:
                best_gini = gini
                split_idx = feat_idx
                split_threshold = thresholds

        return split_idx, split_threshold

    def _gini_index(self, y, X_column, threshold):
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float('inf')

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        gini_l = 1.0 - sum((np.sum(y[left_idxs] == c) / n_l) ** 2 for c in np.unique(y[left_idxs]))
        gini_r = 1.0 - sum((np.sum(y[right_idxs] == c) / n_r) ** 2 for c in np.unique(y[right_idxs]))

        gini_index = (n_l / n) * gini_l + (n_r / n) * gini_r



        return gini_index

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

        
def classwise_accuracy(y_true, y_pred):
    # Initialize a dictionary to store counts of correct and total predictions for each class
    class_counts = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Iterate through each pair of true and predicted labels
    for true_label, pred_label in zip(y_true, y_pred):
        # Increment the total count for the true label
        class_counts[true_label]['total'] += 1
        # If the true label matches the predicted label, increment the correct count for that class
        if true_label == pred_label:
            class_counts[true_label]['correct'] += 1

    # Print class-wise accuracy
    for class_label, counts in class_counts.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        print(f"Class {class_label}: Accuracy = {accuracy:.2f} ({counts['correct']} / {counts['total']})")


# Loading the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
classes=np.unique(train_labels)


indices = np.where((train_labels == 0) | (train_labels == 1) | (train_labels == 2))

x_filtered = train_images[indices]
y_filtered = train_labels[indices]



# Step 4: Transform Data
x_filtered_flat = x_filtered.reshape(x_filtered.shape[0], -1)
pca = PCA(n_components=100)

# Fit the PCA model to your data and transform it
x_transformed = pca.fit_transform(x_filtered_flat)

print(x_transformed.shape)
# print(np.mean(x_filtered_flat[0]))
# print(np.mean(x_transformed[0]))

indices=np.where((test_labels == 0) | (test_labels == 1) | (test_labels == 2))

test_images_filtered=test_images[indices]
test_images_filtered=test_images_filtered.reshape(test_images_filtered.shape[0],-1)
test_labels_filered=test_labels[indices]


clf = DecisionTree(max_depth=2)
clf.fit(x_transformed, y_filtered)
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

predictions = clf.predict(test_images_filtered)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(test_labels_filered, predictions)
classwise_accuracy(test_labels_filered,predictions)
print(acc)

