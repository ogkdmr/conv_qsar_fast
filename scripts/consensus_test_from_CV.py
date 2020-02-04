import rdkit.Chem as Chem
import numpy as np
import datetime
import json
import sys
import os
import time
from tqdm import tqdm
from distutils.util import strtobool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conv_qsar_fast.main.core import build_model, train_model, save_model
from conv_qsar_fast.main.test import test_model, test_embeddings_demo, rocauc_plot, parity_plot
from conv_qsar_fast.main.data import get_data_full
from conv_qsar_fast.utils.parse_cfg import read_config
import conv_qsar_fast.utils.reset_layers as reset_layers

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

	###################################################################################
	### BUILD MODEL
	###################################################################################

	print('...building model')
	try:
		kwargs = config['ARCHITECTURE']
		if '__name__' in kwargs:  # often it does not exist and raises error
			del kwargs['__name__'] #  from configparser
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

		if 'molecular_attributes' in config['DATA']:
			kwargs['molecular_attributes'] = config['DATA']['molecular_attributes']

		model = build_model(**kwargs)
		print('...built untrained model')
	except KeyboardInterrupt:
		print('User cancelled model building')
		quit(1)

	###################################################################################
	### DEFINE DATA 
	###################################################################################

	data_kwargs = config['DATA']
	if '__name__' in data_kwargs:
		del data_kwargs['__name__'] #  from configparser

	if 'molecular_attributes' in data_kwargs:
		data_kwargs['molecular_attributes'] = strtobool(data_kwargs['molecular_attributes'])

	if 'smiles_index' in data_kwargs:
		data_kwargs['smiles_index'] = int(data_kwargs['smiles_index'])

	if 'y_index' in data_kwargs:
		data_kwargs['y_index'] = int(data_kwargs['y_index'])

	if 'skipline' in data_kwargs:
		data_kwargs['skipline'] = strtobool(data_kwargs['skipline'])

	# In the original code the whole dataset was used for testing consensus
	# for now I'll leave it, TODO: change it in the future
	if strtobool(data_kwargs['cv']):
		fold_keys = [key for key in data_kwargs if "fold" in key]
		one_data = []
		for key in fold_keys:
			one_data.append(data_kwargs[key])
			del data_kwargs[key]
	else:
		one_data = [data_kwargs['train'], data_kwargs['val']]
		del data_kwargs['train']
		del data_kwargs['val']

	# test set is always test set
	one_data.append(data_kwargs['test'])
	del data_kwargs['test']
	del data_kwargs['cv']

	data = get_data_full(train_paths=one_data, validation_path=[], test_path=[], **data_kwargs)

	# Unpack
	(train, val, test) = data
	# Unpack
	mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']
	# mols_val   = val['mols'];   y_val   = val['y'];   smiles_val   = val['smiles']
	# mols_test  = test['mols'];  y_test  = test['y'];  smiles_test  = test['smiles']
	y_label = train['y_label']

	y_train_pred = np.array([np.array([0.0 for t in range(1)]) for z in mols_train])
	print(y_train_pred.shape)
	# y_val_pred = np.array([0 for z in mols_val])
	# y_test_pred = np.array([0 for z in mols_test])

	ref_fpath = fpath
	experiment_dir = os.path.dirname(fpath)
	fold_dirs = [os.path.join(experiment_dir, dir) for dir in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, dir))]

	for cv_fold in fold_dirs:
		print('Using weights from CV fold {}'.format(os.path.dirname(cv_fold)))
		fpath = cv_fold

		# Load weights
		weights_fpath = fpath + '.h5'
		model.load_weights(weights_fpath)

		for j in tqdm(range(len(mols_train))):
			# single_mol_as_array = np.array(mols_train[j:j+1])  # old
			atom_f, adj_mat, bond_f = mols_train[j:j + 1][0]  # tuple of len 3 (atom_f, adj_mat, bond_f)
			single_mol_as_array = [np.reshape(atom_f, newshape=(1,)+atom_f.shape),
								   np.reshape(adj_mat, newshape=(1,)+adj_mat.shape),
								   np.reshape(bond_f, newshape=(1,)+bond_f.shape)]  # now it's a list, and dimensions match
			single_y_as_array = np.array(y_train[j:j+1])
			spred = model.predict_on_batch(single_mol_as_array)
			y_train_pred[j] += spred[0]  # because we get a batch of one example
			# y_train_pred[j,:] += spred.flatten()

	# Now divide by the number of folds to average predictions
	y_train_pred = y_train_pred / float(len(fold_dirs))

	# Create plots for datasets
	if strtobool(config['TEST']['calculate_parity']):
		parity_plot(y_train, y_train_pred, '(consensus)', os.path.join(experiment_dir, "consensus"), y_label)
	if strtobool(config['TEST']['calculate_rocauc']):
		rocauc_plot(y_train, y_train_pred, '(consensus)', os.path.join(experiment_dir, "consensus"))
