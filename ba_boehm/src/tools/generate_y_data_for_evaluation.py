from src.data_loader import create_data
import simbench as sb

SB_CODE = '1-MV-rural--0-sw'

WITHOUT_DA_TRAIN_PATH = 'data/without_da/train/'

def main():

    # simbench net
    net = sb.get_simbench_net(SB_CODE)

    # load profiles
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    # generate y_data without data augmentation.
    load_p_mw = profiles.get(('load', 'p_mw'))
    create_data(net=net, path=WITHOUT_DA_TRAIN_PATH, timesteps=range(0, len(load_p_mw)))

if __name__ == "__main__":
    main()