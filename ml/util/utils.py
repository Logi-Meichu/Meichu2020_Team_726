from util.const import NUM_USERS_CHAOSHEN, NUM_USERS_DFL, DATASET_AMOUNT
from util.settings import DATASET, TEST_SIZE
import numpy as np

def datasetname( datasetid, data_amount, num_training_actions ):
    if( data_amount == DATASET_AMOUNT.FIRST1000):
        switcher={
            DATASET.BALABIT: 'balabit_39feat_PC_MM_DD_'+str(num_training_actions)+'.csv',
            DATASET.CHAOSHEN: 'chaoshencont_39feat_PC_MM_DD_'+str(num_training_actions)+'.csv',
            DATASET.DFL: 'dfl_39feat_PC_MM_DD_'+str(num_training_actions)+'.csv'
        }
        return switcher.get(datasetid, 'Invalid dataset')
    else:
        # use ALL data
        switcher = {
            DATASET.BALABIT: 'balabit_39feat_PC_MM_DD.csv',
            DATASET.CHAOSHEN: 'chaoshencont_39feat_PC_MM_DD.csv',
            DATASET.DFL: 'dfl_39feat_PC_MM_DD.csv'
        }
        return switcher.get(datasetid, 'Invalid dataset')


def create_userids( datasetid ):
    switcher = {
        DATASET.BALABIT: [7, 9, 12, 15, 16, 20, 21, 23, 29, 35],
        DATASET.CHAOSHEN: range(1, NUM_USERS_CHAOSHEN + 1),
        DATASET.DFL: range(1, NUM_USERS_DFL + 1)
    }
    return switcher.get(datasetid, 'Invalid dataset')

def keeporder_split(X, y, test_size=TEST_SIZE):
    num_positive_samples = (int) (len(X) / 2)
    num_positive_test_samples = (int)(num_positive_samples * test_size)
    num_positive_train_samples = num_positive_samples - num_positive_test_samples

    # print(num_positive_samples)
    # print(num_positive_test_samples)
    # print(num_positive_train_samples)
    # X_train, X_validation, y_train, y_validation =
    X_train =  np.concatenate( (X[0:num_positive_train_samples,:], X[num_positive_samples:num_positive_samples+num_positive_train_samples,:]))
    y_train = np.concatenate((y[0:num_positive_train_samples], y[num_positive_samples:num_positive_samples+num_positive_train_samples]))
    X_validation = np.concatenate((X[num_positive_train_samples:num_positive_samples,:], X[num_positive_samples+num_positive_train_samples:len(X)+1,:]))
    y_validation = np.concatenate((y[num_positive_train_samples:num_positive_samples], y[num_positive_samples+num_positive_train_samples:len(X)+1]))
    return X_train, X_validation, y_train, y_validation