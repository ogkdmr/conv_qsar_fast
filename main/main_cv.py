import os
import sys
import datetime
import numpy as np
from distutils.util import strtobool

from conv_qsar_fast.main.core import build_model, train_model, save_model
from conv_qsar_fast.main.test import test_model, test_embeddings_demo
from conv_qsar_fast.main.data import get_data_full
from conv_qsar_fast.utils.parse_cfg import read_config

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} "settings.cfg"'.format(sys.argv[0]))
        quit(1)

    print(f"Running main_cv for config: {sys.argv[1]}")
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
    # # # DEFINE DATA
    ###################################################################################

    data_kwargs = config['DATA']
    if '__name__' in data_kwargs:
        del data_kwargs['__name__']

    if 'molecular_attributes' in data_kwargs: 
        data_kwargs['molecular_attributes'] = strtobool(data_kwargs['molecular_attributes'])

    if 'smiles_index' in data_kwargs:
        data_kwargs['smiles_index'] = int(data_kwargs['smiles_index'])

    if 'y_index' in data_kwargs:
        data_kwargs['y_index'] = int(data_kwargs['y_index'])

    if 'skip_line' in data_kwargs:
        data_kwargs['skip_line'] = strtobool(data_kwargs['skip_line'])

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
        splits = list(zip(trains, vals))
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
        # # # BUILD MODEL
        ###################################################################################

        print('...building model')
        try:
            kwargs = config['ARCHITECTURE']
            if '__name__' in kwargs: del kwargs['__name__']
            if 'embedding_size' in kwargs: 
                kwargs['embedding_size'] = int(kwargs['embedding_size'])
            if 'hidden' in kwargs: 
                kwargs['hidden'] = int(kwargs['hidden'])
            if 'hidden2' in kwargs:
                kwargs['hidden2'] = int(kwargs['hidden2'])
            if 'depth' in kwargs: 
                kwargs['depth'] = int(kwargs['depth'])
            if 'dr1' in kwargs:
                kwargs['dr1'] = float(kwargs['dr1'])
            if 'dr2' in kwargs:
                kwargs['dr2'] = float(kwargs['dr2'])
            if 'output_size' in kwargs:
                kwargs['output_size'] = int(kwargs['output_size'])
            if 'optimizer' in kwargs:
                kwargs['optimizer'] = kwargs['optimizer']
            if 'lr' in kwargs:
                kwargs['lr'] = float(kwargs['lr'])

            if 'molecular_attributes' in config['DATA']:
                kwargs['molecular_attributes'] = config['DATA']['molecular_attributes']

            #print(kwargs) {'embedding_size': 512, 'depth': 5, 'hidden': 50, 'dr1': 0.5, 'dr2': 0.5, 'molecular_attributes': 1}

            model = build_model(**kwargs)
            print('...built untrained model')
        except KeyboardInterrupt:
            print('User cancelled model building')
            quit(1)

        ###################################################################################
        # # # LOAD DATA
        ###################################################################################

        print(f"Using CV fold {1+fold_idx}/{len(splits)}")
        data = get_data_full(train_paths=train_paths, validation_path=validation_path, test_path=test_path, **data_kwargs)

        ###################################################################################
        # # # LOAD WEIGHTS?
        ###################################################################################

        if 'weights_fpath' in config['IO']:
            weights_fpath = config['IO']['weights_fpath']
        else:
            weights_fpath = fpath + '.h5'

        try:
            use_old_weights = strtobool(config['IO']['use_existing_weights'])
        except KeyError:
            print('Must specify whether or not to use existing model weights')
            quit(1)

        if use_old_weights and os.path.isfile(weights_fpath):
            model.load_weights(weights_fpath)
            print('...loaded weight information')

            # Reset final dense?
            if 'reset_final' in config['IO']:
                if config['IO']['reset_final'] in ['true', 'y', 'Yes', 'True', '1']:
                    layer = model.layers[-1]
                    layer.W.set_value((layer.init(layer.W.shape.eval()).eval()).astype(np.float32))
                    layer.b.set_value(np.zeros(layer.b.shape.eval(), dtype=np.float32))

        elif use_old_weights and not os.path.isfile(weights_fpath):
            print('Weights not found at specified path {}'.format(weights_fpath))
            quit(1)
        else:
            pass

        ###################################################################################
        # # # TRAIN THE MODEL
        ###################################################################################

        # Train model
        try:
            print('...training model')
            kwargs = config['TRAINING']
            if '__name__' in kwargs:
                del kwargs['__name__']  # from configparser
            if 'nb_epoch' in kwargs:
                kwargs['nb_epoch'] = int(kwargs['nb_epoch'])
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = int(kwargs['batch_size'])
            if 'patience' in kwargs:
                kwargs['patience'] = int(kwargs['patience'])
            (model, loss, val_loss) = train_model(model, data, **kwargs)
            print('...trained model')
        except KeyboardInterrupt:
            pass

        ###################################################################################
        # # # SAVE MODEL
        ###################################################################################

        # Get the current time
        tstamp = datetime.datetime.utcnow().strftime('%m-%d-%Y_%H-%M')
        print('...saving model')
        save_model(model, loss, val_loss, fpath=fpath, config=config, tstamp=tstamp)
        print('...saved model')

        ###################################################################################
        # # # TEST MODEL
        ###################################################################################

        print('...testing model')
        test_kwargs = config['TEST']

        calculate_parity = False
        if 'calculate_parity' in test_kwargs:
            calculate_parity = strtobool(test_kwargs['calculate_parity'])

        calculate_rocauc = False
        if 'calculate_rocauc' in test_kwargs:
            calculate_rocauc = strtobool(test_kwargs['calculate_rocauc'])

        _ = test_model(model, data, fpath, tstamp=tstamp, batch_size=int(config['TRAINING']['batch_size']),
                       calculate_parity=calculate_parity, calculate_rocauc=calculate_rocauc)
        print('...tested model')

        ###################################################################################
        # # # TEST EMBEDDINGS?
        ###################################################################################

        # Testing embeddings?
        try:
            if strtobool(config['TESTING']['test_embedding']):
                test_embeddings_demo(model, fpath)
        except KeyError:
            pass