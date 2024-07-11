from datasets import DatasetDict
from datasets import load_from_disk
from sklearn.metrics import cohen_kappa_score


def qwk(y_pred, y):
    """
    Compute the Quadratic Weighted Kappa (QWK) also known as Cohen's Kappa with quadratic weights.

    The QWK measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters)
    to 1 (complete agreement between raters). In the case of QWK, the weights are squared which has the effect of giving more
    weight to larger differences.

    Parameters:
    - y_pred (array-like): Predicted labels, as returned by a classifier.
    - y (array-like): Ground truth (correct) target values.

    Returns:
    - float: The QWK score between `y_pred` and `y`.
    """
    return cohen_kappa_score(y_pred, y, weights='quadratic')


def load_dataset(path, num_of_fold = 5):
    """
    Load a dataset from a specified directory, assuming the dataset is split into 5 folds stored in separate subdirectories.

    This function loads each fold from its subdirectory (assumed to be named 'fold_{i}' where i ranges from 0 to 4), and
    formats each fold as a PyTorch dataset. It then aggregates all folds into a single DatasetDict object.

    Parameters:
    - path (str): The path to the directory containing the dataset folds.

    Returns:
    - DatasetDict: A dictionary-like object from Hugging Face's `datasets` library, where keys are fold names ('fold_{i}')
    and values are the corresponding datasets formatted as PyTorch datasets.
    """
    folds = []
    for i in range(num_of_fold):
        folds.append(load_from_disk(f'{path}/fold_{i}').with_format('torch'))

    dataset = DatasetDict({f'fold_{i}': fold for i, fold in enumerate(folds)})

    return dataset
