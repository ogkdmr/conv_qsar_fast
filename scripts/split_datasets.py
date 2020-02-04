import os
import rdkit.Chem as Chem
import csv
import numpy as np
from operator import itemgetter

"""This script randomly splits Abraham, Bradley and Delaney into 5 folds + test and saves as separate files."""

RANDOM_SEED = 123


def load_original_file(filename, delimiter, skipline):
    """
    File loader
    :param filename: str: path
    :param delimiter: str: delimiter used, passed to csv.reader
    :param skipline: bool: does the frst line contains column names (true) or data (then false)
    :return: data, firstline
    """
    firstline = None
    data = []
    with open(filename, 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar='"')
        if skipline:
            firstline = next(reader)
        for row in reader:
            data.append(row)
    return data, firstline


def get_unique_smiles(data, smiles_index):
    """
    Some molecules are listed multiple times. Checking on canonical SMILES.
    :param data: list of lists: data returned by load_original_file
    :param smiles_index: int: which entry is SMILES
    :return: list of unique canonical SMILES
    """
    unique_smiles = set()
    for row in data:
        try:
            mol = Chem.MolFromSmiles(row[smiles_index], sanitize=False)
            Chem.SanitizeMol(mol)
            this_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            unique_smiles.add(this_smiles)
        except Exception as e:
            pass
    # sometimes there are empty smiles
    if '' in unique_smiles:
        unique_smiles.remove('')
    return unique_smiles


def define_split_on_smiles(unique, fold_sizes, test_size, seed=123):
    """
    Defines split.
    :param unique: list of unique canonical SMILES
    :param fold_sizes: list of ints: sizes of folds
    :param test_size: int: size of test
    :param seed: int: seed
    :return: smiles_folds, smiles_test
    """
    assert test_size + sum(fold_sizes) == len(unique)

    smiles_folds_indices = []
    smiles_test_indices = []

    all_smiles = list(unique)

    np.random.seed(seed)
    randomised_indices = list(range(len(all_smiles)))
    np.random.shuffle(randomised_indices)

    current_index = 0
    # folds
    for f_size in fold_sizes:
        smiles_folds_indices.append(randomised_indices[current_index:current_index + f_size])
        current_index += f_size

    # test
    smiles_test_indices = randomised_indices[current_index:]

    # good sizes?
    assert len(smiles_test_indices) == test_size
    for f_si, smi_fo in zip(fold_sizes, smiles_folds_indices):
        assert f_si == len(smi_fo)

    # all indices present exactly once?
    temp = []
    temp.extend(smiles_test_indices)
    for smi_fo in smiles_folds_indices:
        temp.extend(smi_fo)
    assert sorted(temp) == list(range(len(unique)))

    smiles_test = itemgetter(*smiles_test_indices)(all_smiles)
    smiles_folds = []
    for indices in smiles_folds_indices:
        smiles_folds.append(itemgetter(*indices)(all_smiles))

    return smiles_folds, smiles_test


def save_each_part_in_seperate_file(original_filename, original_data, leading_line, smiles_folds, smiles_test,
                                    smiles_index, saving_path, data_name):
    """
    For each subset find lines that belong to it and save them to the corresponding file.
    :param original_filename: str: path to the original dataset
    :param original_data: data returned by load_original_file
    :param leading_line: str or None: firstline returned by load_original_file
    :param smiles_folds: returned by define_split_on_smiles
    :param smiles_test: returned by define_split_on_smiles
    :param smiles_index: int: which entry is SMILES
    :param saving_path: where should the data be stored
    :param data_name: under what label should the data be stored
    :return: None
    """
    with open(original_filename, 'r') as f:
        lines = f.readlines()

    if leading_line is not None:
        leading_line = lines[0]
        lines = lines[1:]

    # test set
    with open(os.path.join(saving_path, data_name + "-test.csv"), 'w') as f:
        if leading_line is not None:
            f.write(leading_line)

        for line, row in zip(lines, original_data):
            try:
                mol = Chem.MolFromSmiles(row[smiles_index], sanitize=False)
                Chem.SanitizeMol(mol)
                this_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                if this_smiles in smiles_test:
                    f.write(line)
            except Exception as e:
                pass

    # folds
    for idx, current_smiles in enumerate(smiles_folds):
        with open(os.path.join(saving_path, f"{data_name}-fold{idx+1}.csv"), 'w') as f:
            if leading_line is not None:
                f.write(leading_line)

            for line, row in zip(lines, original_data):
                try:
                    mol = Chem.MolFromSmiles(row[smiles_index], sanitize=False)
                    Chem.SanitizeMol(mol)
                    this_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if this_smiles in current_smiles:
                        f.write(line)
                except Exception as e:
                    pass


if __name__ == '__main__':
    # Abraham
    filename = "conv_qsar_fast/data/AbrahamAcree2014_Octsol_partialSmiles.csv"
    smiles_index = 1
    delimiter = ','
    skipline = True

    fold_sizes = [39, 39, 39, 39, 40]
    test_size = 49
    saving_path = "conv_qsar_fast/data/"
    data_name = "AbrahamAcree2014_Octsol_partialSmiles"

    da, first = load_original_file(filename, delimiter, skipline)
    unique = get_unique_smiles(da, smiles_index)
    smiles_folds, smiles_test = define_split_on_smiles(unique, fold_sizes, test_size, RANDOM_SEED)
    save_each_part_in_seperate_file(filename, da, first, smiles_folds, smiles_test, smiles_index, saving_path, data_name)

    # Bradley
    filename = "conv_qsar_fast/data/BradleyDoublePlusGoodMeltingPointDataset.csv"
    smiles_index = 2
    delimiter = ','
    skipline = True

    fold_sizes = [483, 483, 484, 484, 484]
    test_size = 604
    saving_path = "conv_qsar_fast/data/"
    data_name = "BradleyDoublePlusGoodMeltingPointDataset"

    da, first = load_original_file(filename, delimiter, skipline)
    unique = get_unique_smiles(da, smiles_index)
    smiles_folds, smiles_test = define_split_on_smiles(unique, fold_sizes, test_size, RANDOM_SEED)
    save_each_part_in_seperate_file(filename, da, first, smiles_folds, smiles_test, smiles_index, saving_path, data_name)

    # Delaney
    filename = "conv_qsar_fast/data/Delaney2004.txt"
    smiles_index = 3
    delimiter = ','
    skipline = True

    fold_sizes = [178, 178, 179, 179, 179]
    test_size = 224
    saving_path = "conv_qsar_fast/data/"
    data_name = "Delaney2004"

    da, first = load_original_file(filename, delimiter, skipline)
    unique = get_unique_smiles(da, smiles_index)
    smiles_folds, smiles_test = define_split_on_smiles(unique, fold_sizes, test_size, RANDOM_SEED)
    save_each_part_in_seperate_file(filename, da, first, smiles_folds, smiles_test, smiles_index, saving_path, data_name)

    print("done")
