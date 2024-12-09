import sys
import unittest
import pandas as pd
sys.path.insert(1, 'src')

from data_loader import load_data_msci
from portfolio import Portfolio, Strategy



class TestPortfolio(unittest.TestCase):

    def __init__(self, testname):
        super().__init__(testname)
        data = load_data_msci(path=None, n=3)
        self.return_series = data['return_series']

    def test_portfolio(self):
        tickers = self.return_series.columns

        timeline = ['2022-08-30', '2022-09-30', '2022-10-31']
        weights = [{tickers[0]: 0.3, tickers[1]: 0.7},
                   {tickers[0]: 0.2, tickers[1]: 0.4, tickers[2]: 0.4},
                   {tickers[1]: 0.5, tickers[2]: 0.5}]

        # create portfolios
        portfolios = [Portfolio(date, weight) for date, weight in zip(timeline, weights)]

        # test float weight
        w_float = portfolios[0].float_weights(self.return_series, '2022-10-01')

        # create a strategy
        strategy = Strategy(portfolios)

        weights_df = strategy.get_weights_df()
        weights_at_date = strategy.get_weights('2022-09-30')
        turnover = strategy.turnover(self.return_series, True)
        portf_returns = strategy.simulate(self.return_series)
        self.assertEqual(turnover.iloc[0], 0.8)



if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(TestPortfolio('test_portfolio'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
