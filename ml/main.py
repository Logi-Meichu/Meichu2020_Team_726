import warnings

from measurements.evaluate_classifier import evaluate_dataset
from util.const import DATASET_AMOUNT
from util.settings import CURRENT_DATASET, NUM_ACTIONS, DATASET_USAGE, NUM_TRAINING_SAMPLES

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')



if __name__ == '__main__':
    evaluate_dataset(CURRENT_DATASET, DATASET_USAGE, NUM_ACTIONS, NUM_TRAINING_SAMPLES)