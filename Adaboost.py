from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

class AdaBoost:
    def __init__(self, n_stumps=300, max_depth=1):
        self.n_stumps = n_stumps
        self.max_depth = max_depth
        self.stumps = []
        self.stump_weights = []
        self.val_accuracies=[]
        self.best_stump=None
    
    def fit(self, X, y,X_val,y_val,X_test,y_test):
        # Initialize weights
        weights = np.ones(len(y)) / len(y)
        
        for stump_i in range(self.n_stumps):
            # Train a decision stump
            stump = DecisionTree(min_samples_split=2, max_depth=self.max_depth)
            stump.fit(X, y)
            
            # Predict with the stump
            predictions = stump.predict(X)
            
            # Calculate error
            err = np.sum(weights * (predictions != y))
            
            # Calculate stump weight
            stump_weight = 0.5 * np.log((1 - err) / err)
            
            # Update weights
            weights *= np.exp(-stump_weight * y * predictions)
            weights /= np.sum(weights)
            print("stump i : ",stump_i)
            print(weights)

            n_samples=X.shape[0]
            new_X = np.empty((n_samples, X.shape[1]))
            new_y = np.empty(n_samples)
            
            # Sample with replacement based on weights
            for i in range(n_samples):
                idx = np.random.choice(len(X), p=weights)
                new_X[i] = X[idx]
                new_y[i] = y[idx]
            X=new_X
            y=new_y
            # Save stump and weight
            self.stumps.append(stump)
            self.stump_weights.append(stump_weight)

            val_predictions = self.predict(X_val)
            val_accuracy = accuracy(y_val, val_predictions)
            self.val_accuracies.append(val_accuracy)
            
            # Check if this is the best model so far
            if self.best_stump==None or val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.best_stump = stump
        
        # Use the best model to predict on test set
        test_predictions = stump.predict(X_test)
        test_accuracy = accuracy(y_test, test_predictions)
        
        print(f"Best Validation Accuracy: {best_val_accuracy}")
        print(f"Test Accuracy with Best Model: {test_accuracy}")
        
        # Plot accuracy on validation set vs. number of trees
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.n_stumps + 1), self.val_accuracies, marker='o', linestyle='-')
        plt.title('Accuracy on Validation Set vs. Number of Trees')
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
    
    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))




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
            thresholds = (np.sort(np.unique(X_column))[:-1] + np.sort(np.unique(X_column))[1:]) / 2
            for threshold in thresholds:
                
                gini = self._gini_index(y, X_column, threshold)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _gini_index(self, y, X_column, threshold):
        # print(threshold)
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