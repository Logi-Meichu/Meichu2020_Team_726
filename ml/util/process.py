import csv

def writeCSVa(y_validation, y_scores):
    n = len(y_validation)
    with open('probaB.csv', mode='w', newline='') as data_file:
        data_writer = csv.writer(data_file, delimiter=',')
        for i in range(n):
            data_writer.writerow([y_validation[i], y_scores[i]])
    return