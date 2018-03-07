import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from simulate.simulate import Simulation

from battery_controller_trainable import BatteryContollerTrainable
from battery import Battery


class Train(Simulation):
    """ Handles running a simulation.
    """
    def __init__(self,
                 data,
                 battery,
                 site_id):
        """ Creates initial simulation state based on data passed in.

            :param data: contains all the time series needed over the considered period
            :param battery: is a battery instantiated with 0 charge and the relevant properties
            :param site_id: the id for the site (building)
        """
        super().__init__(self, data, battery, site_id)

    def simulate_timestep(self, battery_controller, current_time, timestep):
        """ Executes a single timestep using `battery_controller` to get
            a proposed state of charge and then calculating the cost of
            making those changes.

            :param battery_controller: The battery controller
            :param current_time: the timestamp of the current time step
            :param timestep: the data available at this timestep
        """
        # get proposed state of charge from the battery controller
        proposed_state_of_charge = battery_controller.propose_state_of_charge(
            self.site_id,
            current_time,
            self.battery,
            self.actual_previous_load,
            self.actual_previous_pv,
            timestep[self.price_buy_columns],
            timestep[self.price_sell_columns],
            timestep[self.load_columns],
            timestep[self.pv_columns]
        )

        # get energy required to achieve the proposed state of charge
        grid_energy, battery_energy_change = self.simulate_battery_charge(self.battery.current_charge,
                                                                          proposed_state_of_charge,
                                                                          timestep.actual_consumption,
                                                                          timestep.actual_pv)

        grid_energy_without_battery = timestep.actual_consumption - timestep.actual_pv

        # buy or sell energy depending on needs
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00
        price_without_battery = timestep.price_buy_00 if grid_energy_without_battery >= 0 else timestep.price_sell_00

        # calculate spending based on price per kWh and energy per Wh
        money_spent_this_period = grid_energy * (price / 1000.)
        money_spent_this_period_without_battery = grid_energy_without_battery * (price_without_battery / 1000.)
        
        reward_this_period = money_spent_this_period_without_battery - money_spent_this_period
        battery_controller.get_feedback(reward_this_period)

        self.money_spent += money_spent_this_period
        self.money_spent_without_battery += money_spent_this_period_without_battery

        # update current state of charge
        self.battery.current_charge += battery_energy_change / self.battery.capacity
        self.actual_previous_load = timestep.actual_consumption
        self.actual_previous_pv = timestep.actual_pv


if __name__ == '__main__':
    simulation_dir = (Path(__file__)/os.pardir/os.pardir).resolve()
    data_dir = simulation_dir/'data'
    output_dir = simulation_dir/'output'

    # load available metadata to determine the runs
    metadata_path = data_dir/'metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0)

    # store results of each run
    results = []

    # # execute two runs with each battery for every row in the metadata file:
    for site_id, parameters in tqdm(metadata.iterrows(), desc='sites\t\t\t', total=metadata.shape[0]):
        site_data_path = data_dir/"submit"/f"{site_id}.csv"

        if site_data_path.exists():
            site_data = pd.read_csv(site_data_path,
                                    parse_dates=['timestamp'],
                                    index_col='timestamp')

            for batt_id in tqdm([1, 2], desc=' > batteries \t\t'):
                # create the battery for this run
                # (Note: Quantities in kW are converted to watts here)
                batt = Battery(capacity=parameters[f"Battery_{batt_id}_Capacity"] * 1000,
                               charging_power_limit=parameters[f"Battery_{batt_id}_Power"] * 1000,
                               discharging_power_limit=-parameters[f"Battery_{batt_id}_Power"] * 1000,
                               charging_efficiency=parameters[f"Battery_{batt_id}_Charge_Efficiency"],
                               discharging_efficiency=parameters[f"Battery_{batt_id}_Discharge_Efficiency"])

                # execute the simulation for each simulation period in the data
                n_periods = site_data.period_id.nunique()
                for g_id, g_df in tqdm(site_data.groupby('period_id'), total=n_periods, desc=' > > periods\t\t'):
                    sim = Train(g_df, batt, site_id)
                    money_spent, money_no_batt = sim.run()

                    # store the results
                    results.append({
                        'run_id': f"{site_id}_{batt_id}_{g_id}",
                        'site_id': site_id,
                        'battery_id': batt_id,
                        'period_id': g_id,
                        'money_spent': money_spent,
                        'money_no_batt': money_no_batt,
                        'score': money_spent / money_no_batt,
                    })

    # write all results out to a file
    results_df = pd.DataFrame(results).set_index('run_id')
    results_df = results_df[['site_id', 'battery_id', 'period_id', 'money_spent', 'money_no_batt', 'score']]
    results_df.to_csv(output_dir/'results.csv')
