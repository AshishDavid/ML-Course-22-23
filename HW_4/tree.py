import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    prob = np.sum(y, axis=0)
    prob = prob / prob.sum()
    return -np.sum(prob * np.log(prob + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    prob = np.sum(y, axis=0) 
    prob = prob / prob.sum()
    return 1 - np.sum(np.square(prob))
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.mean((y - y.mean())**2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_col, thr, pred=None):
        self.feature_col = feature_col
        self.value = thr
        self.pred = pred
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
    
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
       
        
    def make_split(self, feature_col, thr, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and thr
        
        Parameters
        ----------
        feature_col : int
            Index of feature to make split with

        thr : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < thr
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= thr
        """
        l_ind = (X_subset[:, feature_col] < thr)
        r_ind = np.logical_not(l_ind)
        return (X_subset[l_ind], y_subset[l_ind]), (X_subset[r_ind], y_subset[r_ind])
    
    def make_split_only_y(self, feature_col, thr, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and thr
        
        Parameters
        ----------
        feature_col : int
            Index of feature to make split with

        thr : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < thr

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= thr
        """
        l_ind = (X_subset[:, feature_col] < thr)
        r_ind = np.logical_not(l_ind)
        return y_subset[l_ind], y_subset[r_ind]

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best thr w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_col : int
            Index of feature to make split with

        thr : float
            Threshold value to perform split

        """
        feature_col, thr = -1, -1
        max_crit = None
        for col in range(X_subset.shape[1]):
            unicol = set(X_subset[:, col])
            unicol.add(max(unicol) + 0.01)
            for curr_thr in unicol:
                y_left, y_right = self.make_split_only_y(col, curr_thr, X_subset, y_subset)
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                curr_crit = -y_left.shape[0]*self.criterion(y_left)
                curr_crit -= y_right.shape[0]*self.criterion(y_right)
                if max_crit is None or curr_crit > max_crit:
                    max_crit = curr_crit
                    feature_col = col
                    thr = curr_thr

        return feature_col, thr
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        if self.max_depth == 0 or all([np.all(y == y_subset[0]) for y in y_subset]):
            pred = np.sum(y_subset, axis=0)/(np.sum(np.sum(y_subset, axis=0))) if self.classification else np.mean(y_subset)
            return Node(None, None, pred)
        feature_col, thr = self.choose_best_split(X_subset, y_subset)
        n_node = Node(feature_col, thr)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_col, thr, X_subset, y_subset)
        self.max_depth -= 1
        n_node.left_child = self.make_tree(X_left, y_left)
        n_node.right_child = self.make_tree(X_right, y_right)
        self.max_depth += 1
        return n_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the preds should be provided for

        Returns
        -------
        y_pred : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        y_pred = list()
        for x in X:
            curr_node = self.root
            while curr_node.pred is None:
              curr_node = curr_node.left_child if x[curr_node.feature_col] < curr_node.value else curr_node.right_child
            if self.classification:
                y_pred.append(np.argmax(curr_node.pred))
            else:
                y_pred.append(curr_node.pred)
        return np.array(y_pred).reshape((-1, 1))
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the preds should be provided for

        Returns
        -------
        y_pred_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'
        y_pred_probs = list()
        for x in X:
            curr_node = self.root
            while curr_node.pred is None:
              curr_node = curr_node.left_child if x[curr_node.feature_col] < curr_node.value else curr_node.right_child
            y_pred_probs.append(curr_node.pred)
        return np.vstack(y_pred_probs)
