import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import model_selection, metrics
from sklearn.metrics import auc

from util.settings import NUM_ACTIONS, DATASET, CURRENT_DATASET, DATASET_USAGE, NUM_TRAINING_SAMPLES
from util.utils import datasetname


def plotROC_index(fpr, tpr, roc_auc, index):
    if( index >= len(fpr)):
        print("Wrong index "+index)
    plt.figure()
    lw = 2
    plt.plot(fpr[index], tpr[index], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[index])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plotROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def plotROCs(fpr, tpr, roc_auc, items, plot_user_auc = False):
    lw = 2
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in fpr:
        if( plot_user_auc ):
            plt.plot(fpr[i], tpr[i], lw=lw, alpha=.3, label='user %d (AUC = %0.4f)' % (i, roc_auc[i]) )
        tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    # plot mean
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # end plot mean
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    titlestr= datasetname(CURRENT_DATASET, DATASET_USAGE, NUM_TRAINING_SAMPLES)+' - '+str(NUM_ACTIONS)+' action(s)'
    plt.title(titlestr)
    plt.legend(loc="lower right")
    plt.show()
