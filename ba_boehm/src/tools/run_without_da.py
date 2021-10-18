from src.data_loader import load_csv_as_df
from src.data_evaluation import eval_with_linear_reg
import simbench as sb
import pandas as pd

TRAIN_TEST_SPLIT = 0.7
SB_CODE = '1-MV-rural--0-sw'
DATA_FOLDER = 'data/'
TIMES_TO_AUG = 5

WITHOUT_DA_TRAIN_PATH = 'data/without_da/train/'
WITHOUT_DA_PKL_PATH = 'data/without_da/'
RESULTS_WITHOUT_DA = 'results/result_without_da.txt'

def main():

    # simbench net
    net = sb.get_simbench_net(SB_CODE)

    # load profiles
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    def run_without_da(res_file):

        # run without Data Augmentation

        load_p_mw = profiles.get(('load', 'p_mw'))
        load_q_mvar = profiles.get(('load', 'q_mvar'))
        sgen_p_mw = profiles.get(('sgen', 'p_mw'))

        x_data = pd.concat([load_p_mw, load_q_mvar, sgen_p_mw], axis=1)
        x_data = x_data.T.reset_index(drop=True).T

        train_index = round(len(load_p_mw) * TRAIN_TEST_SPLIT)

        # split into train and test set
        x_data_train = x_data.iloc[:train_index,:]
        x_data_test = x_data.iloc[train_index+1:,:]

        y_data_res_bus_vm_pu = load_csv_as_df(WITHOUT_DA_TRAIN_PATH + 'res_bus/vm_pu.csv')
        y_data_res_bus_p_mw = load_csv_as_df(WITHOUT_DA_TRAIN_PATH + 'res_load/p_mw.csv')
        y_data_res_line_lp = load_csv_as_df(WITHOUT_DA_TRAIN_PATH + 'res_line/loading_percent.csv')

        y_data = pd.concat([y_data_res_bus_vm_pu, y_data_res_bus_p_mw, y_data_res_line_lp], axis=1)
        y_data = y_data.T.reset_index(drop=True).T
        y_data_train = y_data.iloc[:train_index,:]
        y_data_test = y_data.iloc[train_index+1:,:]
        eval_with_linear_reg(X_train=x_data_train, X_test=x_data_test, y_train=y_data_train, y_test=y_data_test, \
                             file_name=res_file, times_to_aug=0, additional_info='For reference')

    # without Data Augmentation
    run_without_da(res_file=RESULTS_WITHOUT_DA)

if __name__ == "__main__":
    main()