# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from optimization import Optimization
from backtest import Backtest, BacktestConstraintProvider
from typing import List

class BacktestConfig:
    def __init__(self, name, optimization: Optimization, constraint_provider: BacktestConstraintProvider = None, n_days: int = 21, lookback: int = 252):
        self.name = name
        self.n_days = n_days
        self.lookback = lookback
        self.optimization = optimization
        self.constraint_provider = constraint_provider


class BacktestMutator:
    def __init__(self, data, start_date: str, **kargs):
        self.data = data
        self.start_date = start_date
        dates = data['return_series'].index.intersection(data['return_series_index'].index)
        self.full_timeline = dates[dates > start_date]
        self.settings = kargs

    def run(self, configs: List[BacktestConfig]):
        results = []
        for i, config in enumerate(configs):
            if not self.settings.get('quiet'):
                print(f'Running backtest {config.name}: {i}/{len(configs)}...')

            n_days = config.n_days
            rebdates = self.full_timeline[::n_days].strftime('%Y-%m-%d').tolist()

            # Initialize backtest object
            bt = Backtest(rebdates = rebdates, width = config.lookback, **self.settings)
            bt.data = self.data
            bt.optimization = config.optimization
            bt.constraint_provider = config.constraint_provider

            start_time = time.time()
            bt.run()
            elapsed_time = time.time() - start_time

            results.append({'config' : config,
                                'elapsed_time' : elapsed_time,
                                'backtest' : bt})
        return results

    def plot_results(results):
        summaries = [result['backtest'].compute_summary() for result in results]

        fig1, ax1 = plt.subplots()
        ax1.set_title('Number of assets')
        fig2, ax2 = plt.subplots()
        ax2.set_title('Difference with benchmark')
        fig3, ax3 = plt.subplots()
        ax3.set_title('Turnover')

        for result, summary in zip(results, summaries):
            ax1.plot(summary['number_of_assets'], label=result['config'].name)
            ax2.plot(summary['returns']['sim'] - summary['returns']['index'], label=result['config'].name)
            ax3.plot(summary['turnover'], label=result['config'].name)

        plt.legend(loc='upper center')

        return summaries
