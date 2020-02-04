from conv_qsar_fast.utils.neural_fp import molToGraph
import rdkit.Chem as Chem
import csv


def get_data_full(train_paths=None, validation_path=None, test_path=None, **kwargs):
    """
    Load all datastes, wrapper for get_data_one.
    :param train_paths: list of paths to training folds
    :param validation_path: list of paths to validation folds
    :param test_path: list of paths to test folds
    :param kwargs: kwargs passed to get_data_one
    :return: train, validation, test
    """

    train = get_data_one(data_fpath=train_paths, **kwargs)
    validation = get_data_one(data_fpath=validation_path, **kwargs)
    test = get_data_one(data_fpath=test_path, **kwargs)

    print('# training: {}'.format(len(train['y'])))
    print('# validation: {}'.format(len(validation['y'])))
    print('# testing: {}'.format(len(test['y'])))

    return train, validation, test


def get_data_one(data_fpath, smiles_index, y_index, delimiter,
                 y_label='', skip_line=False, molecular_attributes=False, averaging='max'):
    """
    This is a helper script to read the data files and return data sets.
    :param averaging: if same samples appears mutiple times in the dataset what function should be used to average labels, 'mean' to take mean of all values, 'max' to leave the maximal value
    :param data_fpath: list of paths to all files (folds) which should be loaded
    :param smiles_index: index of column with smiles
    :param y_index: index of column with y
    :param delimiter: delimiter used in csv
    :param y_label:  what is y? activity? solubility?
    :param skip_line: should the first line be skipped (contains column names?)
    :param molecular_attributes: use molecular attributes?
    :return: {'mols': mols, 'y': y, 'smiles': smiles, 'y_label': y_label}
    """

    ###################################################################################
    # # # READ DATA
    ###################################################################################

    print('reading data...')

    # configparser has problems with \t so we solve it brutally
    if delimiter == '\\t' or delimiter == 'tab':
        delimiter = '\t'

    data = []
    for single_fpath in data_fpath:
        with open(single_fpath, 'r') as data_fid:
            reader = csv.reader(data_fid, delimiter=delimiter, quotechar='"')
            if skip_line:
                next(reader)
            for row in reader:
                data.append(row)
    print('done')

    ###################################################################################
    # # # ITERATE THROUGH DATASET AND CREATE NECESSARY DATA LISTS
    ###################################################################################

    print('processing data...')
    smiles = []
    mols = []
    y = []
    smiles_counter = {}  # keep counting for clever averaging

    for row in data:
        try:
            # Molecule first (most likely to fail)
            mol = Chem.MolFromSmiles(row[smiles_index], sanitize=False)
            Chem.SanitizeMol(mol)
            this_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            this_y = float(row[y_index])

            if this_smiles in smiles_counter:
                # important: Coley did not average TOX21, we do
                print(f"Averaging duplicate entry for: {this_smiles}")
                index = smiles.index(this_smiles)
                old_value = y[index]
                if averaging == 'mean':
                    new_value = (this_y + old_value*smiles_counter[this_smiles]) / (1.+smiles_counter[this_smiles])
                    smiles_counter[this_smiles] += 1
                elif averaging == 'max':
                    new_value = max(old_value, this_y)
                    # updating smiles counter is not needed in this case
                else:
                    raise ValueError("averaging must be either 'mean' or 'max'")
                y[index] = new_value

            else:
                # calculate representation and update lists
                mat_features, mat_adjacency, mat_specialbondtypes = \
                    molToGraph(mol, molecular_attributes=molecular_attributes).dump_as_matrices()

                mols.append((mat_features, mat_adjacency, mat_specialbondtypes))
                y.append(this_y)  # Measured log(solubility M/L)
                smiles.append(this_smiles)  # Smiles
                smiles_counter[this_smiles] = 1

        except Exception as e:
            print('Failed to generate graph for {}, y: {}'.format(row[smiles_index], row[y_index]))
            print(e)

    ###################################################################################
    # # # REPACKAGE AS DICTIONARY
    ###################################################################################
    dataset = {'mols': mols, 'y': y, 'smiles': smiles, 'y_label': y_label}

    return dataset
