from util.const import DATASET, SPLIT_TYPE, DATASET_AMOUNT

# Which dataset to use?
CURRENT_DATASET = DATASET.BALABIT

# How many data to use from the dataset
DATASET_USAGE = DATASET_AMOUNT.FIRST1000

# Use Chronological or Random order
# CURRENT_SPLIT_TYPE = SPLIT_TYPE.RANDOM
CURRENT_SPLIT_TYPE = SPLIT_TYPE.KEEP_ORDER

# Number of samples used for score computations
NUM_ACTIONS = 50

# How many training samples to use - works with DATASET_USAGE = DATASET_AMOUNT.FIRST1000
# should be less or equal than  1000
NUM_TRAINING_SAMPLES = 500

# Split factor
TEST_SIZE = 0.25

# Plot details
PLOT_USER_AUC = False
