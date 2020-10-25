from enum import Enum

import numpy as np

# Directory where feature CSVs are stored
FEAT_DIR = 'features'

# Random states for reproducibility
RANDOM_STATE = np.random.seed(0)

# Number of users Chao Shen dataset
NUM_USERS_CHAOSHEN = 28

# Number of users DFL dataset
NUM_USERS_DFL = 21

class DATASET(Enum):
    BALABIT = 1
    CHAOSHEN = 2
    DFL = 3

# Split type
class SPLIT_TYPE(Enum):
    RANDOM = 1
    KEEP_ORDER = 2

# Amount of data to use from the dataset
class DATASET_AMOUNT(Enum):
    ALL = 1
    FIRST1000= 2

# Plot titles
CHAOSHEN_TITLE = 'Chao Shen continuous dataset'
BALABIT_TITLE = 'Balabit dataset'
DFL_TITLE ='DFL dataset'
