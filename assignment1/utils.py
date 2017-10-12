from dldata.metrics import utils
import numpy as np

def post_process_neural_regression_msplit_preprocessed(result, ne_path):
    """ 
    Loads the precomputed noise estimates and normalizes the results
    """
    ne = np.load(ne_path)
    sarrays = []
    for s_ind, s in enumerate(result['split_results']):
        farray = np.asarray(
                result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = \
            (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = \
            (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = \
            np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = \
            np.median(1 - sarrays, 1).std()


def post_process_neural_regression_msplit(dataset, 
                                          result, 
                                          spec, 
                                          n_jobs=1, 
                                          splits=None, 
                                          nan=False):
    """
    Computes noise estimates and normalizes the results
    """
    name = spec[0]
    specval = spec[1]
    assert name[2] in ['IT_regression', 
                       'V4_regression', 
                       'ITc_regression', 
                       'ITt_regression'], name
    if name[2] == 'IT_regression':
        units = dataset.IT_NEURONS
    elif name[2] == 'ITc_regression':
        units = hvm.mappings.LST_IT_Chabo
    elif name[2] == 'ITt_regression':
        units = hvm.mappings.LST_IT_Tito
    else:
        units = dataset.V4_NEURONS

    units = np.array(units)
    if not splits:
        splits, validations = utils.get_splits_from_eval_config(specval, dataset)

    sarrays = []
    for s_ind, s in enumerate(splits):
        ne = dataset.noise_estimate(
                s['test'], units=units, n_jobs=n_jobs, cache=True, nan=nan)
        farray = np.asarray(
                result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = \
            (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = \
            (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = \
            np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = \
            np.median(1 - sarrays, 1).std()

