from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from measurements.preprocess import get_user_data
import sys

def my_predict(file_name) :
    with open('outmodel.pkl', 'rb') as f :
        model = pickle.load(f)

    test_data = get_user_data(file_name).values
    test_data = test_data[:, :-1]
    y_predict = model.predict(test_data)
    y_predict = y_predict.astype(int)
    yes = 0
    no = 0
    for i in range(len(y_predict)) :
        if y_predict[i] == 1 :
            yes = yes + 1
        else :
            no = no + 1

    print ("yes ", yes)
    print ("no ", no)

    if yes > no :
        print ('\n' * 10)
        print ('The result is : ')
        print ('=' * 45)
        print ("Same user !")
    else :
        print ('\n' * 10)
        print ('The result is : ')
        print ('=' * 45)
        print ("Not the same user !")


if len(sys.argv) != 2 :
    print ("usage : python3 test.py <log_file>")
    exit()
else :
    my_predict(sys.argv[1])
