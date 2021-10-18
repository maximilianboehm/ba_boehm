from src.data_loader import create_data_da
from src.data_loader import load_csv_as_df
from src.data_augmentation import augment_sets_numerical_rand_percentage
from src.data_evaluation import eval_with_linear_reg
import simbench as sb
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--times_to_aug', nargs='?', help='number of times to augment the dataset')
parser.add_argument('--max_percentage', nargs='?', help='max possible percentage for transformations')
args=parser.parse_args()

if args.__getattribute__('times_to_aug') == None:
    TIMES_TO_AUG = 5

else:
    TIMES_TO_AUG = int(args.__getattribute__('times_to_aug'))

if args.__getattribute__('max_percentage') == None:
    MAX_PERCENTAGE = 0.1
else:
    MAX_PERCENTAGE = float(args.__getattribute__('max_percentage'))

TRAIN_TEST_SPLIT = 0.7
SB_CODE = '1-MV-rural--0-sw'
DATA_FOLDER = 'data/'

PERCENTAGE_TRANSFORMATIONS_PKL_PATH = 'data/percentage_transformations/'
PERCENTAGE_TRANSFORMATIONS_TRAIN_PATH = 'data/percentage_transformations/train/'
RESULTS_PERCENTAGE_TRANSFORMATIONS = 'results/result_percentage_transformations.txt'

WITHOUT_DA_TRAIN_PATH = 'data/without_da/train/'

def main():

    # simbench net
    net = sb.get_simbench_net(SB_CODE)

    # load profiles
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    def augment_dataset(path, times_to_aug):

        if not os.path.exists(path):
            os.makedirs(path)

        # load profiles
        load_p_mw = profiles.get(('load', 'p_mw'))
        load_q_mvar = profiles.get(('load', 'q_mvar'))
        sgen_p_mw = profiles.get(('sgen', 'p_mw'))

        train_index = round(len(load_p_mw) * TRAIN_TEST_SPLIT)

        load_p_mw_train = load_p_mw.iloc[:train_index,:]
        load_q_mvar_train = load_q_mvar.iloc[:train_index,:]
        sgen_p_mw_train = sgen_p_mw.iloc[:train_index,:]

        # rand numerical transformations for every value
        load_p_mw_da = augment_sets_numerical_rand_percentage(dfs=load_p_mw_train, data_name='load_p_mw', \
                                                              times_to_aug=times_to_aug, percentage=MAX_PERCENTAGE)
        load_p_mw_da = load_p_mw_da.reset_index(drop=True)
        load_p_mw_da.to_pickle('./'+path+'/load_p_mw_da.pkl')
        load_q_mvar_da = augment_sets_numerical_rand_percentage(dfs=load_q_mvar_train, data_name='load_q_mvar', \
                                                                times_to_aug=times_to_aug, percentage=MAX_PERCENTAGE)
        load_q_mvar_da = load_q_mvar_da.reset_index(drop=True)
        load_q_mvar_da.to_pickle('./'+path+'/load_q_mvar_da.pkl')
        sgen_p_mw_da = augment_sets_numerical_rand_percentage(dfs=sgen_p_mw_train, data_name='sgen_p_mw', \
                                                              times_to_aug=times_to_aug, percentage=MAX_PERCENTAGE)
        sgen_p_mw_da = sgen_p_mw_da.reset_index(drop=True)
        sgen_p_mw_da.to_pickle('./'+path+'/sgen_p_mw_da.pkl')

    def run_with_da(x_path, y_train_path, y_test_path, times_to_aug, additional_info, res_file):
        # run with Data Augmentation
        load_p_mw = profiles.get(('load', 'p_mw'))
        load_q_mvar = profiles.get(('load', 'q_mvar'))
        sgen_p_mw = profiles.get(('sgen', 'p_mw'))

        load_p_mw_da = pd.read_pickle(x_path + '/load_p_mw_da.pkl')
        load_q_mvar_da = pd.read_pickle(x_path + '/load_q_mvar_da.pkl')
        sgen_p_mw_da = pd.read_pickle(x_path + '/sgen_p_mw_da.pkl')
        x_data_train = pd.concat([load_p_mw_da, load_q_mvar_da, sgen_p_mw_da], axis=1)

        # split profiles into train and test sets
        train_index = round(len(load_p_mw) * TRAIN_TEST_SPLIT)

        # get real test data
        load_p_mw_test = load_p_mw.iloc[train_index + 1:, :]
        load_q_mvar_test = load_q_mvar.iloc[train_index + 1:, :]
        sgen_p_mw_test = sgen_p_mw.iloc[train_index + 1:, :]

        # get real not augmented data for test set
        x_data_test = pd.concat([load_p_mw_test, load_q_mvar_test, sgen_p_mw_test], axis=1)
        x_data_test = x_data_test.T.reset_index(drop=True).T
        x_data_test = x_data_test.reset_index(drop=True)

        y_data_train_res_bus_vm_pu = load_csv_as_df(y_train_path + '/res_bus/vm_pu.csv')
        y_data_train_res_load_p_mw = load_csv_as_df(y_train_path + '/res_load/p_mw.csv')
        y_data_train_res_line_lp = load_csv_as_df(y_train_path + '/res_line/loading_percent.csv')
        y_data_train = pd.concat([y_data_train_res_bus_vm_pu, y_data_train_res_load_p_mw, y_data_train_res_line_lp], \
                                 axis=1)
        y_data_train = y_data_train.T.reset_index(drop=True).T
        y_data_train = y_data_train.reset_index(drop=True)

        # load real not augmented data for test set
        y_data_test_res_bus_vm_pu = load_csv_as_df(y_test_path + '/res_bus/vm_pu.csv')
        y_data_test_res_load_p_mw = load_csv_as_df(y_test_path + '/res_load/p_mw.csv')
        y_data_test_res_line_lp = load_csv_as_df(y_test_path + '/res_line/loading_percent.csv')
        y_data_test = pd.concat([y_data_test_res_bus_vm_pu, y_data_test_res_load_p_mw, y_data_test_res_line_lp], \
                                axis=1)
        y_data_test = y_data_test.iloc[train_index + 1:, :]
        y_data_train = y_data_train.T.reset_index(drop=True).T
        y_data_train = y_data_train.reset_index(drop=True)

        eval_with_linear_reg(X_train=x_data_train, X_test=x_data_test, y_train=y_data_train, y_test=y_data_test, \
                             file_name=res_file, times_to_aug=times_to_aug, additional_info=additional_info)

    def generate_y_data(train_path, pkl_path):
        # load data with Data Augmentation
        load_p_mw_da_sv = pd.read_pickle(pkl_path + '/load_p_mw_da.pkl')
        load_q_mvar_da_sv = pd.read_pickle(pkl_path + '/load_q_mvar_da.pkl')
        sgen_p_mw_da_sv = pd.read_pickle(pkl_path + '/sgen_p_mw_da.pkl')

        create_data_da(net=net, path=train_path, timesteps=range(0, len(load_p_mw_da_sv)),
                        load_p_mw=load_p_mw_da_sv, load_q_mvar=load_q_mvar_da_sv,
                        sgen_p_mw=sgen_p_mw_da_sv)

    # da with percentage transformations
    augment_dataset(path=PERCENTAGE_TRANSFORMATIONS_PKL_PATH, times_to_aug=TIMES_TO_AUG)
    generate_y_data(train_path=PERCENTAGE_TRANSFORMATIONS_TRAIN_PATH, pkl_path=PERCENTAGE_TRANSFORMATIONS_PKL_PATH)
    run_with_da(x_path=PERCENTAGE_TRANSFORMATIONS_PKL_PATH, \
                y_train_path=PERCENTAGE_TRANSFORMATIONS_TRAIN_PATH, \
                y_test_path=WITHOUT_DA_TRAIN_PATH, times_to_aug=TIMES_TO_AUG, \
                additional_info='transform test 10%', res_file=RESULTS_PERCENTAGE_TRANSFORMATIONS)

if __name__ == "__main__":
    main()
