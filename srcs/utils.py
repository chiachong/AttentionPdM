import numpy as np


def one_hot_encode(labels, n_classes):
    """
    Reformat and reshape the input data and labels into TensorFlow format.

    Args:
        labels (array): input labels
        n_classes (int): Number of classes
    Returns:
        labels(ndarray): one-hot-encoded labels
    """
    # Map 0 to (1.0, 0.0, 0.0 ...), 1 to (0.0, 1.0, 0.0 ...)
    labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    return labels
