import numpy as np
import datetime
import sys
import os
from sklearn.svm import SVR
from distutils.util import strtobool

from conv_qsar_fast.utils.parse_cfg import read_config
from conv_qsar_fast.main.test import test_model, test_embeddings_demo
from conv_qsar_fast.main.data import get_data_full

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} "settings.cfg"'.format(sys.argv[0]))
        quit(1)

    # Load settings
    try:
        config = read_config(sys.argv[1])
    except:
        print('Could not read config file {}'.format(sys.argv[1]))
        quit(1)

    # Get model label
    try:
        fpath = config['IO']['model_fpath']
    except KeyError:
        print('Must specify model_fpath in IO in config')
        quit(1)

    # make directory
    try:
        os.makedirs(os.path.dirname(fpath))
    except:  # folder exists
        pass

    ###################################################################################
    # # # BUILD MODEL
    ###################################################################################

    print('...building model')
    try:
        kwargs = config['ARCHITECTURE']
        if 'kernel' not in kwargs:
            kwargs['kernel'] = 'rbf'  # default

        if kwargs['kernel'] in ['linear', 'rbf', 'poly']:
            model = SVR(kernel = kwargs['kernel'])
        elif kwargs['kernel'] in ['tanimoto']:
            def kernel(xa, xb):
                m = xa.shape[0]
                n = xb.shape[0]
                xa = xa.astype(bool)
                xb = xb.astype(bool)
                score = np.zeros((m, n))
                for i in range(m):
                    for j in range(n):
                        score[i, j] = float(np.sum(np.logical_and(xa[i], xb[j])))  / np.sum(np.logical_or(xa[i], xb[j]))
                #print(score)
                return score

            model = SVR(kernel = kernel)
        else:
            raise ValueError('Unknown kernel!')

        print('...built untrained model')
    except KeyboardInterrupt:
        print('User cancelled model building')
        quit(1)

    ###################################################################################
    # # # DEFINE DATA
    ###################################################################################

    data_kwargs = config['DATA']
    if '__name__' in data_kwargs:
        del data_kwargs['__name__'] #  from configparser

    if 'smiles_index' in data_kwargs:
        data_kwargs['smiles_index'] = int(data_kwargs['smiles_index'])

    if 'y_index' in data_kwargs:
        data_kwargs['y_index'] - int(data_kwargs['y_index'])

    if 'skipline' in data_kwargs:
        data_kwargs['skipline'] = strtobool(data_kwargs['skipline'])

    # defining data split: cross-validation or regular train-valid-test
    if strtobool(data_kwargs['cv']):
        fold_keys = [key for key in data_kwargs if "fold" in key]
        all_folds = []
        for key in fold_keys:
            all_folds.append(data_kwargs[key])
            del data_kwargs[key]

        trains = []
        vals = []
        for val_fold in all_folds:
            trains.append([fold for fold in all_folds if fold != val_fold])
            vals.append([val_fold, ])
        splits = list(zip((trains, vals)))
    else:
        splits = list(zip([[data_kwargs['train'], ], ], [[data_kwargs['val'], ], ]))
        del data_kwargs['train']
        del data_kwargs['val']

    # test set is always test set
    test_path = [data_kwargs['test'], ]
    del data_kwargs['test']
    del data_kwargs['cv']

    # Iterate through all folds
    ref_fpath = fpath
    for fold_idx, (train_paths, validation_path) in enumerate(splits):
        fpath = ref_fpath.replace('<this_fold>', str(1+fold_idx))

        ###################################################################################
        # # # LOAD DATA
        ###################################################################################

        print(f"Using CV fold {1+fold_idx}/{len(splits)}")
        data = get_data_full(train_paths=train_paths, validation_path=validation_path, test_path=test_path,
                             **data_kwargs)

        # TRAIN MODEL
        try:
            print('...training model')
            model.fit([x[0] for x in data[0]['mols']], data[0]['y'])
            print('...trained model')
        except KeyboardInterrupt:
            pass

        ###################################################################################
        # # # TEST MODEL
        ###################################################################################

        print('...testing model')
        tstamp = datetime.datetime.utcnow().strftime('%m-%d-%Y_%H-%M')
        # Need to define predict_on_batch to be compatible
        model.predict_on_batch = model.predict
        _ = test_model(model, data, fpath, tstamp=tstamp, batch_size=1)
        print('...tested model')
