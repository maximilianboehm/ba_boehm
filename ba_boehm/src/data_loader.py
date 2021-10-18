from src.timeseries_simulation import run_timeseries_simulation
from src.timeseries_simulation import run_timeseries_simulation_with_da
import pandas as pd

def create_data(net, path, timesteps):
    run_timeseries_simulation(net=net, path=path, timesteps=timesteps)

def create_data_da(net, path, timesteps, load_p_mw, load_q_mvar, sgen_p_mw):
    run_timeseries_simulation_with_da(net=net, path=path, timesteps=timesteps, load_p_mw=load_p_mw, \
                                      load_q_mvar=load_q_mvar, sgen_p_mw=sgen_p_mw)

def load_csv_as_df(path):
    df = pd.read_csv(path, sep=';', index_col=[0])
    return df
