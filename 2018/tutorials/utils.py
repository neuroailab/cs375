import numpy as np
import scipy.stats as stats


def featurewise_norm(data, fmean=None, fvar=None):
    """perform a whitening-like normalization operation on the data, feature-wise
       Assumes data = (K, M) matrix where K = number of stimuli and M = number of features
    """
    if fmean is None:
        fmean = data.mean(0)
    if fvar is None:
        fvar = data.std(0)
    data = data - fmean  #subtract the feature-wise mean of the data
    data = data / np.maximum(fvar, 1e-5)  #divide by the feature-wise std of the data
    return data, fmean, fvar


def get_off_diagonal(mat):
    n = mat.shape[0]
    i0, i1 = np.triu_indices(n, 1)
    i2, i3 = np.tril_indices(n, -1)
    return np.concatenate([mat[i0, i1], mat[i2, i3]])


def spearman_brown(uncorrected, multiple):
    numerator = multiple * uncorrected
    denominator = 1 + (multiple - 1) * uncorrected
    return numerator / denominator


def idfunc(x):
    return x


def pearsonr(a, b):
    return stats.pearsonr(a, b)[0]


def spearmanr(a, b):
    return stats.spearmanr(a, b)[0]


def split_half_correlation(datas_by_trial,
                           num_splits,
                           aggfunc=idfunc,
                           statfunc=pearsonr):

    """arguments:
              data_by_trial -- list of (numpy arrays) 
                        assumes each is a tensor with structure is (trials, stimuli)
              num_splits (nonnegative integer) how many splits of the data to make
    """
        
    random_number_generator = np.random.RandomState(seed=0)

    corrvals = []
    for split_index in range(num_splits):
        stats1 = []
        stats2 = []
        for data in datas_by_trial:
            #get total number of trials
            num_trials = data.shape[0]

            #construct a new permutation of the trial indices
            perm = random_number_generator.permutation(num_trials)

            #take the first num_trials/2 and second num_trials/2 pieces of the data
            first_half_of_trial_indices = perm[:num_trials / 2]
            second_half_of_trial_indices = perm[num_trials / 2: num_trials]

            #mean over trial dimension
            s1 = aggfunc(data[first_half_of_trial_indices].mean(axis=0))
            s2 = aggfunc(data[second_half_of_trial_indices].mean(axis=0))
            stats1.extend(s1)
            stats2.extend(s2)
        
        #compute the correlation between the means
        corrval = statfunc(np.array(stats1), 
                           np.array(stats2))
        #add to the list
        corrvals.append(corrval)
        
    return spearman_brown(np.array(corrvals), 2)
