# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from optimization import Optimization
from backtest import Backtest
from typing import List

class BacktestConfig:
    def __init__(self, name, optimization: Optimization, n_days: int = 21, lookback: int = 252):
        self.name = name
        self.n_days = n_days
        self.lookback = lookback
        self.optimization = optimization


class BacktestMutator:
    def __init__(self, data, start_date: str):
        self.data = data
        self.start_date = start_date
        dates = data['return_series'].index
        self.full_timeline = dates[dates > start_date]

    def run(self, configs: List[BacktestConfig]):
        results = {}
        for i, config in enumerate(configs):
            n_days = config.n_days
            rebdates = self.full_timeline[::n_days].strftime('%Y-%m-%d').tolist()

            # Initialize backtest object
            bt = Backtest(rebdates = rebdates, width = config.lookback)
            bt.data = self.data
            bt.optimization = config.optimization

            bt.run()

            # Simulation
            sim_bt = bt.strategy.simulate(return_series = bt.data['return_series'], fc = 0, vc = 0)

            # Analyze weights
            bt.strategy.get_weights_df()

            # Analyze simulation
            sim = pd.concat({'sim': sim_bt, 'index': bt.data['return_series_index']}, axis = 1).dropna()
            sim.columns = sim.columns.get_level_values(0)

            sim = np.log(1 + sim).cumsum()

            results[config] = [bt.strategy.number_of_assets(), sim]
        return results

    def plot_results(results):
        plt.xticks([])
        for config, result in results.items():
            plt.plot(result[0], label=config.name)
        plt.legend()
        plt.show()

        for config, result in results.items():
            plt.plot(result[1]['sim'] - result[1]['index'], label=config.name)
        plt.legend()
        plt.show()


        return None
