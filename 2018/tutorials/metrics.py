"""
useful metrics of classifier performance
"""

import numpy as np
import scipy.stats as stats


def get_confusion_matrix(predicted, actual, ucats):
    """Gets confusion matrix where 
           mat[i, j] = number of instances where 
                 actual = category i and 
                 predicted = category j
    """
    cmat = []
    for a_cat in ucats:
        cvec = []
        for p_cat in ucats:
            rate = ((predicted == p_cat) & (actual == a_cat)).sum()
            cvec.append(rate)
        cmat.append(cvec)
    return np.array(cmat)


def dprime_from_rates(tpr, far, clip=5):
    """Computes the formula
          Z(true positive rate) - Z(false alarm rate)
        where Z = inverse of the CDF of the gaussian
    """
    posppf = np.clip(stats.norm.ppf(tpr), -clip, clip)
    negppf = np.clip(stats.norm.ppf(far), -clip, clip)
    return posppf - negppf


def dprime_binary(predicted, actual, clip=5):
    """Assumes predicted, actual binary (0, 1)-valued vectors -- 
       the positive class value is "1" while the negative class is "0"
    """
    total_positives = (actual == 1).sum()
    true_positives = ((predicted == 1) & (actual == 1)).sum()
    true_positive_rate = true_positives / float(total_positives)
    total_negatives = (actual == 0).sum()
    false_alarms = ((predicted == 1) & (actual == 0)).sum()
    false_alarm_rate = false_alarms / float(total_negatives)
    return dprime_from_rates(true_positive_rate, false_alarm_rate, clip=clip)


def confusion_matrix_stats(cmat):
    """get generalized statistics from confusion matrix
       arguments: confusion matrix of shape (M, M) where M = number of categories
                   rows are actual, columns are predicted
       returns:
          len-M vectors of 
            total positives (P)
            total negavites (N)
            true positives (TP)
            true negatives (TN)
            false positives (FP)
            false nevatives (FN)
    """
    M = cmat.shape[0]
    TP = []
    FN = []
    FP = []
    TN = []
    for i in range(M):
        tp = cmat[i, i]                  #true positives are the diagonal element
        fp = cmat[:, i].sum() - tp       #false positives are column sum - diagonal
        fn = cmat[i].sum() - tp          #false negatives are row sum - diagonal
        tn = cmat.sum() - fp - fn - tp   #true negatives are everything else
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)
        TN.append(tn)
    TP = np.array(TP)
    FN = np.array(FN)
    FP = np.array(FP)
    TN = np.array(TN)
    P = TP + FN    #total positives are true positives + false negatives
    N = TN + FP    #total negatives are true negatives + false positives
    return P, N, TP, TN, FP, FN
    
    
def balanced_accuracy(confmat):
    """Computes balanced accuracy (see http://mvpa.blogspot.com/2015/12/balanced-accuracy-what-and-why.html)
       from confusion matrix 
    """
    P, N, TP, TN, FP, FN = confusion_matrix_stats(confmat)
    sensitivity = TP / P.astype(float)
    specificity = TN / N.astype(float)
    balanced_acc = (sensitivity + specificity) / 2.
    return balanced_acc
    

def dprime_confmat(cmat, clip=5):
    """Computes vector of dprimes from confusion matrix
    """
    P, N, TP, TN, FP, FN = confusion_matrix_stats(cmat)
    TPR = TP / P.astype(float)
    FPR = FP / N.astype(float)
    return dprime_from_rates(TPR, FPR, clip=clip)


def accuracy_confmat(cmat):
    correct = cmat.diagonal().sum()
    total = cmat.sum()
    return correct / float(total)
    
    
    
def evaluate_results(confmats, labels):
    """Convenience function that summarize results over confusion matrices
       Arguments:
           confmats = array of shape (M, M, ns)  where M = number of categories
                      and ns = number of splits
                confmats[i, j, k] = number of times classifier predicted class j 
                                    when actual is class i, on split k 
           labels = length-M vector of category labels
           
       Returns: dictionary with useful summary metrics, including dprime,
                balanced accuracy, and percent correct (regular "accuracy")
                both for the split-mean confusion matrix, and separately across splits
                  
    """
    result = {}
    result['labels'] = labels
    result['confusion_matrices'] = confmats
    mean_confmat = confmats.mean(0)
    result['mean_dprime'] = dprime_confmat(mean_confmat)
    result['mean_balanced_accuracy'] = balanced_accuracy(mean_confmat)
    result['mean_accuracy'] = accuracy_confmat(mean_confmat)
    
    result['dprime_by_split'] = [dprime_confmat(c) for c in confmats]
    result['balanced_acc_by_split'] = [balanced_accuracy(c) for c in confmats]
    result['accuracy_by_split'] = [accuracy_confmat(c) for c in confmats]
    
    return result   
