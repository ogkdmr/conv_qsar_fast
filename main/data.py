from conv_qsar_fast.utils.neural_fp import *
import rdkit.Chem as Chem
import numpy as np
import os
import csv


def get_data_full(train_path=None, validation_path=None, test_path=None, **kwargs):
	'''Wrapper for get_data_full, which allows for multiple datasets to
	be concatenated as a multi-target problem'''

	train = get_data_one(data_fpath=train_path, data_label='', **kwargs)
	validation = get_data_one(data_fpath=validation_path, data_label='', **kwargs)
	test = get_data_one(data_fpath=test_path, data_label='', **kwargs)

	print('AFTER MERGING DATASETS...')
	print('# training: {}'.format(len(train['y'])))
	print('# validation: {}'.format(len(validation['y'])))
	print('# testing: {}'.format(len(test['y'])))

	return train, validation, test


def get_data_one(data_fpath, delimeter, smiles_index, y_index, y_label,
				 data_label='', shuffle_seed=None,  molecular_attributes=False):
	"""This is a helper script to read the data file and return
	data sets. Train, validation and test sets should be kept in
	separate files."""

	# Roots
	data_label = data_label.lower()

	###################################################################################
	### WHICH DATASET ARE WE TRYING TO USE?
	###################################################################################

	# Delaney solubility
	if data_label in ['delaney', 'delaney sol']:
		dset = 'delaney'

	# Abraham octanol set
	elif data_label in ['abraham', 'abraham sol', 'abraham_oct']:
		dset = 'abraham'

	elif data_label in ['bradley_good', 'bradley']:
		dset = 'bradley_good'

	elif 'nr' in data_label or 'sr' in data_label:
		print('Assuming TOX21 data {}'.format(data_label))
		dset = data_label

	# ten nie bedzie dzialac
	elif 'tox21' == data_label:
		print('Assuming ALL TOX21 data')
		dset = 'tox21'
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']

	# ani ten
	elif 'tox21-test' == data_label:
		print('Assuming ALL TOX21 data, leaderboard test set')
		dset = 'tox21-test'
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']

	elif 'tox21-eval' == data_label:
		print('Assuming ALL TOX21 data, eval set')
		dset = 'tox21-test'
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']

	elif 'tox21-traintest' == data_label:
		print('Assuming traintest TOX21 data')
		dset = 'tox21-traintest'
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']


	# Other?
	else:
		print('Unrecognized data_label {}'.format(data_label))
		quit(1)


	###################################################################################
	### READ AND TRUNCATE DATA
	###################################################################################

	print('reading data...')
	data = []
	with open(data_fpath, 'r') as data_fid:
		reader = csv.reader(data_fid, delimiter = delimeter, quotechar = '"')
		# Abraham, Bradley and Delaney start with column names
		if data_label in ['delaney', 'delaney sol', 'abraham', 'abraham sol', 'abraham_oct', 'bradley_good', 'bradley']:
			next(reader)
		for row in reader:
			data.append(row)
	print('done')

	# set shuffle seed if possible
	if shuffle_seed is not None:
		np.random.seed(shuffle_seed)
	
	###################################################################################
	### ITERATE THROUGH DATASET AND CREATE NECESSARY DATA LISTS
	###################################################################################

	smiles = []
	mols = []
	y = []
	print('processing data...')
	# Randomize
	np.random.shuffle(data)
	for i, row in enumerate(data):
		try:
			# Molecule first (most likely to fail)
			mol = Chem.MolFromSmiles(row[smiles_index], sanitize = False)
			Chem.SanitizeMol(mol)
			
			(mat_features, mat_adjacency, mat_specialbondtypes) = molToGraph(mol, molecular_attributes=molecular_attributes).dump_as_matrices()


			if 'tox21' not in dset:
				this_y = float(row[y_index])
			else: # get full TOX21 data
				this_y = np.array([float(x) for x in row[y_index:]])

			mols.append((mat_features, mat_adjacency, mat_specialbondtypes))

			y.append(this_y) # Measured log(solubility M/L)
			smiles.append(Chem.MolToSmiles(mol, isomericSmiles = True)) # Smiles

			# Check for redundancies and average
			if 'nr-' in dset or 'sr-' in dset or 'tox21' in dset:
				continue
				# Don't worry about duplicates in TOX21 dataset

			elif smiles.count(smiles[-1]) > 1:
				print('**** DUPLICATE ENTRY ****')
				print(smiles[-1])

				indices = [x for x in range(len(smiles)) if smiles[x] == smiles[-1]]
				y[indices[0]] = (y[indices[0]] + this_y) / 2.
				
				del y[-1]
				del smiles[-1]
				del mols[-1]


		except Exception as e:
			print('Failed to generate graph for {}, y: {}'.format(row[smiles_index], row[y_index]))
			print(e)

	###################################################################################
	### DIVIDE DATA VIA RATIO OR CV
	# but everything is a set, sets are kept separately
	###################################################################################



	if 'all_train' in data_split: # put everything in train
		print('Using ALL as training')
		# Create training/development split
		mols_train = mols
		y_train = y
		smiles_train = smiles

		print('Training size: {}'.format(len(mols_train)))

	###################################################################################
	### REPACKAGE AS DICTIONARIES
	###################################################################################
	if 'cv_full' in data_split: # cross-validation, but use 'test' as validation
		train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
		val   = {}; val['mols']   = mols_test;   val['y']   = y_test;   val['smiles']   = smiles_test;   val['y_label']   = y_label
		test  = {}; test['mols']  = [];  test['y']  = [];  test['smiles']  = []; test['y_label']  = []

	else:

		train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
		val   = {}; val['mols']   = mols_val;   val['y']   = y_val;   val['smiles']   = smiles_val;   val['y_label']   = y_label
		test  = {}; test['mols']  = mols_test;  test['y']  = y_test;  test['smiles']  = smiles_test; test['y_label']  = y_label

	return (train, val, test)
