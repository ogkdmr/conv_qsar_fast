from __future__ import print_function
import rdkit.Chem as Chem
from conv_qsar_fast.utils.neural_fp import molToGraph
import conv_qsar_fast.utils.stats as stats
import keras.backend as K 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
import json


def rocauc_plot(true, pred, set_label, test_fpath, verbose=False):
    if len(true) == 0:
        print('skipping parity plot for empty dataset')
        return

    try:
        print('{}:'.format(set_label))

        # Trim it to recorded values (not NaN)
        true = np.array(true).flatten()
        if verbose: print(true)
        if verbose: print(true.shape)

        pred = np.array(pred).flatten()
        if verbose: print(pred)
        if verbose: print(pred.shape)

        pred = pred[~np.isnan(true)]
        true = true[~np.isnan(true)]

        print('{}:'.format(set_label))

        # For TOX21
        roc_x, roc_y, _ = roc_curve(true, pred)
        rocauc_score = roc_auc_score(true, pred)
        print('AUC = {}'.format(rocauc_score))

        plt.figure()
        lw = 2
        plt.plot(roc_x, roc_y, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % rocauc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for {}'.format(set_label))
        plt.legend(loc="lower right")
        plt.savefig(test_fpath + '_{}_rocauc.png'.format(set_label), bbox_inches='tight')
        plt.clf()

        return rocauc_score

    except Exception as e:
        print(e)
        return 99999


def parity_plot(true, pred, set_label, test_fpath, verbose=False):
    if len(true) == 0:
        print('skipping parity plot for empty dataset')
        return

    try:
        print('{}:'.format(set_label))

        # Trim it to recorded values (not NaN)
        true = np.array(true).flatten()
        pred = np.array(pred).flatten()

        if verbose:
            print(true)
            print(true.shape)
            print(pred)
            print(pred.shape)

        pred = pred[~np.isnan(true)]
        true = true[~np.isnan(true)]

        # calculate values
        min_y = np.min((true, pred))
        max_y = np.max((true, pred))

        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        q = stats.q(true, pred)
        (r2, a) = stats.linreg(true, pred)  # predicted v observed
        (r2p, ap) = stats.linreg(pred, true)  # observed v predicted

        # printing
        print('mse = {}, mae = {}'.format(mse, mae))
        if verbose:
            print('  q = {}'.format(q))
            print('  r2 through origin = {} (pred v. true), {} (true v. pred)'.format(r2, r2p))
            print('  slope through origin = {} (pred v. true), {} (true v. pred)'.format(a[0], ap[0]))

        # Create parity plot
        plt.scatter(true, pred, alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Parity plot for {} ({} set, N = {})'.format(y_label, set_label, len(true)) +
                  '\nMSE = {}, MAE = {}, q = {}'.format(np.round(mse, 3), np.round(mae, 3), np.round(q, 3)) +
                  '\na = {}, r^2 = {}'.format(np.round(a, 3), np.round(r2, 3)) +
                  '\na` = {}, r^2` = {}'.format(np.round(ap, 3), np.round(r2p, 3)))
        plt.grid(True)
        plt.plot(true, true * a, 'r--')
        plt.axis([min_y, max_y, min_y, max_y])

        plt.savefig(test_fpath + '_{}_parity.png'.format(set_label), bbox_inches='tight')
        plt.clf()

        return mse, mae

    except Exception as e:
        print(e)
        return 99999, 99999


def test_model(model, data, fpath, calculate_parity=True, calculate_rocauc=True,
               tstamp='no_time', batch_size=128, return_test_MSE=False, verbose=False):
    """This function evaluates model performance using test data.

    inputs:
        model - the trained Keras model
        data - three dictionaries for training,
                    validation, and testing data. Each dictionary should have
                    keys of 'mol', a molecular tensor, 'y', the target output,
                    and 'smiles', the SMILES string of that molecule
        fpath - folderpath to save test data to, will be appended with '/tstamp.test'
        calculate_parity - should parity plot be calculated and saved
        calculate_rocauc - should rocauc plot be calculated and saved
        tstamp - timestamp to add to the testing
        batch_size - batch_size to use while testing
        return_test_MSE - will it return a float or not, may be used for looking for hparams """

    # Create folder to dump testing info to
    try:
        os.makedirs(fpath)
    except:  # file exists
        pass
    test_fpath = os.path.join(fpath, tstamp)

    # Unpack data
    (train, val, test) = data

    mols_train = train['mols']
    y_train = train['y']

    mols_val = val['mols']
    y_val = val['y']

    mols_test = test['mols']
    y_test = test['y']
    smiles_test = test['smiles']

    print('mols train {}'.format(len(mols_train)))
    print('mols val {}'.format(len(mols_val)))
    print('mols test {}'.format(len(mols_test)))

    # make predictions
    y_train_pred = []
    y_val_pred = []
    y_test_pred = []

    if batch_size == 1:  # UNEVEN TENSORS, ONE AT A TIME PREDICTION
        # Run through training set
        for j in tqdm(range(len(mols_train))):
            single_mol = mols_train[j]
            spred = model.predict_on_batch(
                [np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])
            y_train_pred.append(spred)

        # Run through validation set
        for j in tqdm(range(len(mols_val))):
            single_mol = mols_val[j]
            spred = model.predict_on_batch(
                [np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])
            y_val_pred.append(spred)

        # Run through testing set
        for j in tqdm(range(len(mols_test))):
            single_mol = mols_test[j]
            spred = model.predict_on_batch(
                [np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])
            y_test_pred.append(spred)

    else:  # PADDED
        y_train_pred = np.array([])
        y_val_pred = np.array([])
        y_test_pred = np.array([])
        if mols_train: y_train_pred = model.predict(np.array(mols_train), batch_size=batch_size, verbose=1)
        if mols_val: y_val_pred = model.predict(np.array(mols_val), batch_size=batch_size, verbose=1)
        if mols_test: y_test_pred = model.predict(np.array(mols_test), batch_size=batch_size, verbose=1)


    # Save predictions for test set
    with open(test_fpath + '.test', 'w') as fid:
        fid.write('{} tested {}, predicting {}\n\n'.format(fpath, tstamp, y_label))
        fid.write('test entry\tsmiles\tactual\tpredicted\tactual - predicted\n')
        for i in range(len(smiles_test)):
            fid.write('{}\t{}\t{}\t{}\t{}\n'.format(i,
                                                    smiles_test[i],
                                                    y_test[i],
                                                    y_test_pred[i],
                                                    y_test[i] - y_test_pred[i]))

    #  Plots
    mse_mae = {}
    if calculate_parity:
        if len(y_train) > 0:
            mse_mae['train'] = parity_plot(y_train, y_train_pred, 'train', test_fpath=test_fpath, verbose=verbose)

        if len(y_val):
            mse_mae['val'] = parity_plot(y_val, y_val_pred, 'val', test_fpath=test_fpath, verbose=verbose)

        if len(y_test):
            mse_mae['test']= parity_plot(y_test, y_test_pred, 'test', test_fpath=test_fpath, verbose=verbose)

        # save mse_mae
        with open(os.path.join(fpath, "mse-mae.json"), "w") as f:
            json.dump(mse_mae, f)

    rocauc = {}
    if calculate_rocauc:
        if len(y_train) > 0:
            rocauc['train'] = rocauc_plot(y_train, y_train_pred, 'train', test_fpath=test_fpath, verbose=verbose)

        if len(y_val) > 0:
            rocauc['val'] = rocauc_plot(y_val, y_val_pred, 'val', test_fpath=test_fpath, verbose=verbose)

        if len(y_test) > 0:
            rocauc['test'] = rocauc_plot(y_test, y_test_pred, 'test', test_fpath=test_fpath, verbose=verbose)

        # save rocauc
        with open(os.path.join(fpath, "rocauc.json"), "w") as f:
            json.dump(rocauc, f)


    return (train, val, test), mse_mae, rocauc

def test_embeddings_demo(model, fpath):
	'''This function tests molecular representations by creating visualizations
	of fingerprints given a SMILES string. Molecular attributes are used, so the
	model to load should have been trained using molecular attributes.

	inputs:
		model - the trained Keras model
		fpath - folderpath to save test data to, will be appended with '/embeddings/'
	'''
	print('Building images of fingerprint examples')

	# Create folder to dump testing info to
	try:
		os.makedirs(fpath)
	except: # folder exists
		pass
	try:
		fpath = os.path.join(fpath, 'embeddings')
		os.makedirs(fpath)
	except: # folder exists
		pass

	# Define function to test embedding
	x = K.placeholder(ndim = 4)
	tf = K.function([x], 
		model.layers[0].call(x))

	# Define function to save image
	def embedding_to_png(embedding, label, fpath):
		print(embedding)
		print(embedding.shape)
		fig = plt.figure(figsize=(20,0.5))
		plt.pcolor(embedding, vmin = 0, vmax = 1, cmap = plt.get_cmap('Greens'))
		plt.title('{}'.format(label))
		# cbar = plt.colorbar()
		plt.gca().yaxis.set_visible(False)
		plt.gca().xaxis.set_visible(False)
		plt.xlim([0, embedding.shape[1]])
		plt.subplots_adjust(left = 0, right = 1, top = 0.4, bottom = 0)
		plt.savefig(os.path.join(fpath, label) + '.png', bbox_inches = 'tight')
		with open(os.path.join(fpath, label) + '.txt', 'w') as fid:
			fid.write(str(embedding))
		plt.close(fig)
		plt.clf()
		return

	smiles = ''
	print('**using molecular attributes**')
	while True:
		smiles = raw_input('Enter smiles: ').strip()
		if smiles == 'done':
			break
		try:
			mol = Chem.MolFromSmiles(smiles)
			mol_graph = molToGraph(mol, molecular_attributes = True).dump_as_tensor()
			single_mol_as_array = np.array([mol_graph])
			embedding = tf([single_mol_as_array])
			with open(os.path.join(fpath, smiles) + '.embedding', 'w') as fid:
				for num in embedding.flatten():
					fid.write(str(num) + '\n')
			embedding_to_png(embedding, smiles, fpath)
		except:
			print('error saving embedding - was that a SMILES string?')

	return
