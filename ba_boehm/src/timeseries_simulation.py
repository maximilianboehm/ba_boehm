import pandapower.control as control
import pandapower.timeseries as timeseries
import simbench as sb

def run_timeseries_simulation(net, path, timesteps):

    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_p_mw = timeseries.DFData(profiles.get(('load', 'p_mw')))
    load_q_mvar = timeseries.DFData(profiles.get(('load', 'q_mvar')))
    sgen_p_mw = timeseries.DFData(profiles.get(('sgen', 'p_mw')))

    # controllers for power flow simulation
    const_load_p_mw = control.ConstControl(net, element='load', element_index=net.load.index,
                                      variable='p_mw', data_source=load_p_mw, profile_name=net.load.index)
    const_load_q_mvar = control.ConstControl(net, element='load', element_index=net.load.index,
                                             variable='q_mvar', data_source=load_q_mvar, profile_name=net.load.index)
    const_sgen_p_mw = control.ConstControl(net, element='sgen', element_index=net.sgen.index,
                                           variable='p_mw', data_source=sgen_p_mw, profile_name=net.sgen.index)

    # wirte output to desired directory as .csv
    ow = timeseries.OutputWriter(net, output_path=path, output_file_type=".csv")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_load', 'p_mw')

    timeseries.run_timeseries(net, time_steps=timesteps)

def run_timeseries_simulation_with_da(net, path, timesteps, load_p_mw, load_q_mvar, sgen_p_mw):

    load_p_mw = timeseries.DFData(load_p_mw)
    load_q_mvar = timeseries.DFData(load_q_mvar)
    sgen_p_mw = timeseries.DFData(sgen_p_mw)

    # controllers for power flow simulation
    const_load_p_mw = control.ConstControl(net, element='load', element_index=net.load.index,
                                      variable='p_mw', data_source=load_p_mw, profile_name=net.load.index)
    const_load_q_mvar = control.ConstControl(net, element='load', element_index=net.load.index,
                                             variable='q_mvar', data_source=load_q_mvar, profile_name=net.load.index)
    const_sgen_p_mw = control.ConstControl(net, element='sgen', element_index=net.sgen.index,
                                           variable='p_mw', data_source=sgen_p_mw, profile_name=net.sgen.index)

    # wirte output to desired directory as .csv
    ow = timeseries.OutputWriter(net, output_path=path, output_file_type=".csv")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_load', 'p_mw')

    timeseries.run_timeseries(net, time_steps=timesteps)
