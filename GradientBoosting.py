from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
 
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
            leaf_value = np.mean(y)
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
        best_ssr = float('inf')
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            #thresholds = np.unique(X_column)
            thresholds=np.median(np.sort(X_column))
            gini = self.find_ssr(y, X_column, thresholds)
            if gini < best_ssr:
                best_ssr = gini
                split_idx = feat_idx
                split_threshold = thresholds
                
                

        return split_idx, split_threshold

    def find_ssr(self, y, X_column, threshold):
        # print(threshold)
        left_idxs, right_idxs = self._split(X_column, threshold)


        # if len(left_idxs) == 0 or len(right_idxs) == 0:
        #     return float('inf')

        # n = len(y)
        # n_l, n_r = len(left_idxs), len(right_idxs)

        # gini_l = 1.0 - sum((np.sum(y[left_idxs] == c) / n_l) ** 2 for c in np.unique(y[left_idxs]))
        # gini_r = 1.0 - sum((np.sum(y[right_idxs] == c) / n_r) ** 2 for c in np.unique(y[right_idxs]))

        # gini_index = (n_l / n) * gini_l + (n_r / n) * gini_r
        left_ssr = y[left_idxs] - self._most_common_label(y[left_idxs])
        right_ssr = y[right_idxs] - self._most_common_label(y[right_idxs])

        # Compute SSR
        ssr = np.sum(left_ssr**2)+np.sum(right_ssr**2)
        
        return ssr


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
    
class GradientBoosting:
    def __init__(self, num_stumps=50, learning_rate=0.01):
        self.num_stumps = num_stumps
        self.learning_rate = learning_rate
        self.stumps = []
        self.alphas = []

    
    
    def fit(self, X_train, y_train, X_val, y_val):
        predictions = np.full(y_train.shape, np.mean(y_train))
        self.val_mse = []

        trees = []

        for _ in range(self.num_stumps):
            # Compute negative gradient
            negative_gradient = y_train - predictions
            
            # Fit decision stump to negative gradient
            tree = DecisionTree(max_depth=1)
            tree.fit(X_train, negative_gradient)
            
            # Compute predictions of decision stump
            stump_predictions = tree.predict(X_train)
            
            # Compute alpha (learning rate)
            alpha = self.learning_rate
            
            # Update predictions
            predictions += alpha * stump_predictions
            
            # Compute MSE on validation set
            val_predictions = tree.predict(X_val)
            mse = np.mean((y_val - val_predictions) ** 2)
            print(mse)
            self.val_mse.append(mse)
            
            trees.append(tree)
        
        # Choose the best model based on validation MSE
        best_idx = np.argmin(self.val_mse)
        self.best_model = trees[best_idx]
        
        return trees

    def evaluate_test_set(self, X_test, y_test):
        test_predictions = self.best_model.predict(X_test)
        test_mse = np.mean((y_test - test_predictions) ** 2)
        return test_mse

    def plot_val_mse(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_stumps + 1), self.val_mse, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Trees')
        plt.ylabel('Validation MSE')
        plt.title('Validation MSE vs. Number of Trees')
        plt.grid(True)
        plt.show()