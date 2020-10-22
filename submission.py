import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    # Create root A1 node
    decision_tree_root = DecisionNode(None, None, lambda a1: a1[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    a3 = DecisionNode(None, None, lambda a3: a3[2] == 1)
    decision_tree_root.right = a3

    #Create A3 left node
    a4_a3_left = DecisionNode(None, None, lambda a4: a4[3] == 1)
    a4_a3_left.left = DecisionNode(None, None, None, 1)
    a4_a3_left.right = DecisionNode(None, None, None, 0)
    a3.left = a4_a3_left

    # Create A3 right node
    a4_a3_right = DecisionNode(None, None, lambda a4: a4[3] == 1)
    a4_a3_right.left = DecisionNode(None, None, None, 0)
    a4_a3_right.right = DecisionNode(None, None, None, 1)
    a3.right = a4_a3_right

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    # Convert input into numpy arrays
    classifier_np = np.array(classifier_output)
    labels_np = np.array(true_labels)

    # Get true positive values
    tp_classifier = (classifier_np == 1)
    tp_label = (labels_np == 1)
    tp = np.count_nonzero(np.logical_and(tp_classifier, tp_label))

    # Get false positive values
    fp_classifier = (classifier_np == 1)
    fp_label = (labels_np == 0)
    fp = np.count_nonzero(np.logical_and(fp_classifier, fp_label))

    # Get false negative values
    fn_classifier = (classifier_np == 0)
    fn_label = (labels_np == 1)
    fn = np.count_nonzero(np.logical_and(fn_classifier, fn_label))

    # Get true negative values
    tn_classifier = (classifier_np == 0)
    tn_label = (labels_np == 0)
    tn = np.count_nonzero(np.logical_and(tn_classifier, tn_label))

    return np.array([[tp, fn], [fp, tn]])

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    # Convert input into numpy arrays
    classifier_np = np.array(classifier_output)
    labels_np = np.array(true_labels)

    # Get true positive values
    tp_classifier = (classifier_np == 1)
    tp_label = (labels_np == 1)
    tp = np.count_nonzero(np.logical_and(tp_classifier, tp_label))

    # Get false positive values
    fp_classifier = (classifier_np == 1)
    fp_label = (labels_np == 0)
    fp = np.count_nonzero(np.logical_and(fp_classifier, fp_label))

    return tp / (tp + fp)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    # Convert input into numpy arrays
    classifier_np = np.array(classifier_output)
    labels_np = np.array(true_labels)

    # Get true positive values
    tp_classifier = (classifier_np == 1)
    tp_label = (labels_np == 1)
    tp = np.count_nonzero(np.logical_and(tp_classifier, tp_label))

    # Get false negative values
    fn_classifier = (classifier_np == 0)
    fn_label = (labels_np == 1)
    fn = np.count_nonzero(np.logical_and(fn_classifier, fn_label))

    return tp / (tp + fn)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    cm = confusion_matrix(classifier_output, true_labels)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    return (tp + tn) / (tp + fp + fn + tn)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) == 0:
        return 0
    class_vector_np = np.array(class_vector)
    num_ones = np.count_nonzero(class_vector_np)
    num_zeros = class_vector_np.size - num_ones
    p_1 = num_ones / class_vector_np.size
    p_0 = num_zeros / class_vector_np.size
    return 1 - ((p_1 ** 2) + (p_0 ** 2))


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini_prev = gini_impurity(previous_classes)

    total_cur_elements = np.array(current_classes).size
    gini_cur = 0
    for cl in current_classes:
        gini_cur += (len(cl) / total_cur_elements) * gini_impurity(cl)
    return gini_prev - gini_cur


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=15):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(
            self,
            features,
            classes,
            attr_sample_rate=None,
            attr_considered=None
        ):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            attr_sample_rate (optional): attribute sample rate for random forest
            attr_considered (optional): attributes already considered
        """

        self.root = self.__build_tree__(
            features,
            classes,
            attr_sample_rate=attr_sample_rate,
            attr_considered=attr_considered
        )

    def __build_tree__(self, features, classes, depth=0, attr_sample_rate=None, attr_considered=None):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        features_np = np.array(features)
        classes_np = np.array(classes)

        # Check base cases
        if classes_np.size == 0:
            return None
        if np.all(classes_np == classes_np[0]):
            return DecisionNode(None, None, None, classes_np[0])
        elif depth == self.depth_limit:
            num_ones = np.count_nonzero(classes_np)
            num_zeros = classes_np.size - num_ones
            return DecisionNode(None, None, None, 1 if num_ones > num_zeros else 0)

        # Find best attribute to split on
        # First, check to see if this tree is part of a random forest
        attr_used = None
        if attr_sample_rate is not None:
            # Find attributes to split on
            num_attr = features_np.shape[1] * attr_sample_rate
            attr_avail = [attr for attr in np.arange(features_np.shape[1]) if attr not in attr_considered]
            if len(attr_avail) == 0:
                # We've used up all the attributes and should return the plurality of classes
                # as the class
                num_ones = np.count_nonzero(classes_np)
                num_zeros = classes_np.size - num_ones
                return DecisionNode(None, None, None, 1 if num_ones > num_zeros else 0)
            elif len(attr_avail) == 1:
                # Only one attribute remains
                attr_used = attr_avail
            else:
                attr_used = np.random.choice(attr_avail, size=num_attr, replace=False)

        # Keeps track of tuple of (attr_idx, best_threshold, best_gain)
        alpha_threshold = []
        for i in range(features_np.shape[1]):
            # Check to see if this is part of a random forest. If so, skip this attribute if it wasn't
            # selected to be trained on
            if attr_sample_rate is not None:
                if i not in attr_used:
                    continue

            # Create mx2 matrix. First column is the attribute 2nd column is the class label
            cur_alpha = np.zeros((features_np.shape[0], 2))
            cur_alpha[:, 0] = features_np[:, i]
            cur_alpha[:, 1] = classes_np

            # Sort cur_alpha by attribute values. Maintain class label
            cur_alpha_sorted = cur_alpha[cur_alpha[:, 0].argsort()]

            # Look for changes in class label. If there is a change, mark the first and last index
            # of the label and calculate the average, which will be the threshold to split alpha on
            # Calculate the gini gain for each split and save the best one
            start_idx = 0
            best_threshold = 0
            max_gain = float("-inf")
            for j in range(cur_alpha_sorted.shape[0]):
                if cur_alpha_sorted[j, 1] != cur_alpha_sorted[start_idx, 1]:
                    end_idx = j
                    # Calculate average of this array slice
                    threshold = cur_alpha_sorted[start_idx:end_idx].mean(axis=0)[0]
                    # Split attribute based on threshold.
                    right_split = cur_alpha_sorted[cur_alpha_sorted[:, 0] < threshold][:, 1]
                    left_split = cur_alpha_sorted[cur_alpha_sorted[:, 0] >= threshold][:, 1]
                    # Calculate Gini Gain for this threshold
                    cur_gain = gini_gain(classes_np, [left_split, right_split])
                    if cur_gain > max_gain:
                        best_threshold = threshold
                        max_gain = cur_gain
                    start_idx = j

            alpha_threshold.append((i, best_threshold, max_gain))

        # Get tuples of (attr_idx, threshold) pairs
        best_alpha_meta = np.array(alpha_threshold)
        np.argmax(best_alpha_meta[:, :, ])
        idx = np.argmax(best_alpha_meta[:, 2])
        best_alpha_threshold = best_alpha_meta[idx]

        # Create root node for subtree. Consider anything greater than the threshold to be "True"
        attr = best_alpha_threshold[0]
        thr = best_alpha_threshold[1]
        if attr_considered is not None:
            attr_considered.append(attr)
        features_classes = np.zeros((features_np.shape[0], features_np.shape[1] + 1))
        features_classes[:, 0:features_np.shape[1]] = features_np
        features_classes[:, -1] = classes_np
        root = DecisionNode(None, None, lambda ex: ex[int(attr)] >= thr)

        left_split = features_classes[features_classes[:, int(attr)] >= thr]
        right_split = features_classes[features_classes[:, int(attr)] < thr]
        # Create left node
        left_node = self.__build_tree__(
            left_split[:, 0:features_np.shape[1]],
            left_split[:, -1],
            depth+1,
            attr_sample_rate,
            attr_considered
        )
        if left_node is None:
            # Set node to return label of pluraity of current examples
            num_ones = np.count_nonzero(classes_np)
            num_zeros = classes_np.size - num_ones
            left_node = DecisionNode(None, None, None, 1 if num_ones > num_zeros else 0)

        # Create right node
        right_node = self.__build_tree__(
            right_split[:, 0:features_np.shape[1]],
            right_split[:, -1],
            depth+1,
            attr_sample_rate,
            attr_considered
        )
        if right_node is None:
            # Set node to return label of pluraity of current examples
            num_ones = np.count_nonzero(classes_np)
            num_zeros = classes_np.size - num_ones
            right_node = DecisionNode(None, None, None, 1 if num_ones > num_zeros else 0)
        root.left = left_node
        root.right = right_node
        return root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        # Each column is a feature. Send each column in to decision tree, save label
        features_np = np.array(features)
        for i in range(features_np.shape[0]):
            feature = features_np[i, :]
            class_labels.append(self.root.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    datapoints_np = np.array(dataset[0])
    labels_np = np.array(dataset[1])
    split_interval = int(datapoints_np.shape[0] / k)
    folds = []
    for i in range(k):
        # Create training dataset
        left_half = datapoints_np[:i*split_interval, :]
        right_half = datapoints_np[(i+1) * split_interval:, :]
        training_features = np.concatenate((left_half, right_half))
        left_half = labels_np[:i*split_interval]
        right_half = labels_np[(i+1) * split_interval:]
        training_classes = np.concatenate((left_half, right_half))
        train_set = (training_features, training_classes)

        # Fit tree
        dt = DecisionTree()
        dt.fit(training_features, training_classes)

        # Create test dataset
        test_features = datapoints_np[i*split_interval:(i+1)*split_interval, :]
        test_classes = dt.classify(test_features)
        test_set = (test_features, test_classes)
        folds.append((train_set, test_set))
    return folds

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        full_dataset = np.zeros((features.shape[0], features.shape[1] + 1))
        full_dataset[:, :-1] = features
        full_dataset[:, -1] = classes
        for i in range(self.num_trees):
            # Get the examples to train on
            num_samples = int(features.shape[0] * self.example_subsample_rate)
            sample_idxs = np.random.randint(0, high=features.shape[0], size=num_samples)
            examples = full_dataset[sample_idxs]

            # Train decision tree
            dt = DecisionTree()
            dt.fit(examples[:, :-1], examples[:, -1], self.attr_subsample_rate, [])
            self.trees.append(dt)


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """



class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        mult = data * data
        return mult + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        first_100 = data[0:100, :]
        summed = np.sum(first_100, axis=1)
        idx = summed.argmax()
        return summed[idx], idx

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """
        flat = data.flatten()
        unique_nums = {}
        for i in range(flat.shape[0]):
            if flat[i] > 0:
                if flat[i] in unique_nums.keys():
                    count = unique_nums[flat[i]]
                    count += 1
                    unique_nums[flat[i]] = count
                else:
                    unique_nums[flat[i]] = 1
        return unique_nums.items()

def return_your_name():
    # return your name
    return "Nikhil Badami"
