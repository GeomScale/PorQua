'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''



import pandas as pd
import numpy as np





class Portfolio:

    def __init__(self,
                 rebalancing_date: str = None,
                 weights: dict = None,
                 name: str = None,
                 init_weights: dict = None):
        self._rebalancing_date = rebalancing_date
        self._weights = weights if weights else {}
        self._name = name
        self._init_weights = init_weights if init_weights else {}

    @staticmethod
    def empty() -> 'Portfolio':
        return Portfolio()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights: dict):
        if not isinstance(new_weights, dict):
            if hasattr(new_weights, 'to_dict'):
                new_weights = new_weights.to_dict()
            else:
                raise TypeError('weights must be a dictionary')
        self._weights = new_weights

    @property
    def rebalancing_date(self):
        return self._rebalancing_date

    @rebalancing_date.setter
    def rebalancing_date(self, new_date: str):
        if new_date and not isinstance(new_date, str):
            raise TypeError('date must be a string')
        self._rebalancing_date = new_date

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError('name must be a string')
        self._name = new_name

    def __repr__(self):
        return f'Portfolio(rebalancing_date={self.rebalancing_date}, weights={self._weights})'

    def float_weights(self,
                      return_series: pd.DataFrame,
                      end_date: str,
                      rescale: bool = False):
        if self.weights:
            return floating_weights(return_series,
                                    self.weights,
                                    self.rebalancing_date,
                                    end_date,
                                    rescale)
        else:
            return None

    def initial_weights(self,
                        selection: list[str],
                        return_series: pd.DataFrame,
                        end_date: str,
                        rescale: bool = True) -> dict[str, float]:

        if not hasattr(self, '_init_weights'):
            if self.rebalancing_date is not None and self.weights is not None:
                w_init = dict.fromkeys(selection, 0)
                w_float = self.float_weights(return_series=return_series,
                                             end_date=end_date,
                                             rescale=rescale)
                w_floated = w_float.iloc[-1]

                w_init.update({key: w_floated[key] for key in w_init.keys() & w_floated.keys()})
                self._init_weights = w_init
            else:
                self._init_weights = None  # {key: 0 for key in selection}

        return self._init_weights

    def turnover(self, portfolio: "Portfolio", return_series: pd.DataFrame, rescale=True):
        if portfolio.rebalancing_date and portfolio.rebalancing_date < self.rebalancing_date:
            w_init_floated = portfolio.float_weights(return_series, self.rebalancing_date, rescale).iloc[-1]
            return pd.Series(self._weights).sub(pd.Series(w_init_floated), fill_value=0).abs().sum()
        else:
            return None


class Strategy:

    def __init__(self, portfolios: list[Portfolio]):
        rebdates = [portfolio.rebalancing_date for portfolio in portfolios]
        assert all(rebdates[i] < rebdates[i + 1] for i in range(len(rebdates) - 1)), 'Portfolios are not sorted'

        self._portfolios = portfolios
        self._rebalancing_dates = rebdates

    @property
    def portfolios(self):
        return self._portfolios

    @portfolios.setter
    def portfolios(self, new_portfolios: list[Portfolio]):
        if not isinstance(new_portfolios, list):
            raise TypeError('portfolios must be a list')
        if not all(isinstance(portfolio, Portfolio) for portfolio in new_portfolios):
            raise TypeError('all elements in portfolios must be of type Portfolio')
        self._portfolios = new_portfolios

    def get_weights_df(self) -> pd.DataFrame:
        weights_dict = {portfolio.rebalancing_date : portfolio.weights for portfolio in self._portfolios}
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index')
        weights_df.fillna(0, inplace=True)
        return weights_df

    def get_portfolio(self, date: str) -> Portfolio:
        idx = get_lower_bound_index(date, self._rebalancing_dates)
        if idx >= 0 and self._rebalancing_dates[idx] ==  date:
            return self._portfolios[idx]
        else:
            raise ValueError(f'No portfolio found for at {date}')

    def get_weights(self, date: str) -> dict[str, float]:
        portfolio = self.get_portfolio(date)
        return portfolio.weights

    def __repr__(self):
        return f'Strategy(portfolios={self._portfolios})'

    def number_of_assets(self, th: float = 0.0001) -> pd.Series:
        return self.get_weights_df().apply(lambda x: sum(np.abs(x) > th), axis=1)

    def turnover(self, return_series, rescale=True) -> pd.Series:
        portfolios = self._portfolios
        rebdates = self._rebalancing_dates
        turnover = {rebdates[i] : portfolios[i].turnover(portfolios[i-1], return_series, rescale)
                    for i in range(1, len(rebdates))}

        return pd.Series(turnover)

    def simulate(self,
                 return_series=None,
                 fc: float = 0,
                 vc: float = 0,
                 n_days_per_year: int = 252) -> pd.DataFrame:

        rebdates = self._rebalancing_dates
        returns = []

        for i in range(len(rebdates)):
            portfolio = self._portfolios[i]

            next_rebdate = rebdates[i + 1] if i + 1 < len(rebdates) else return_series.index[-1]
            w_float = portfolio.float_weights(return_series, end_date=next_rebdate, rescale=False)  # Note that rescale is hardcoded to False.

            short_positions = list(filter(lambda x: x < 0, portfolio.weights.values()))
            long_positions = list(filter(lambda x: x >= 0, portfolio.weights.values()))

            short_expo = sum(short_positions)
            long_expo = sum(long_positions)
            margin = abs(short_expo)
            cash = max(min(1 - long_expo, 1), 0)
            loan = 1 - (long_expo + cash) - (short_expo + margin)
            w_float.insert(0, 'margin', margin)
            w_float.insert(0, 'cash', cash)
            w_float.insert(0, 'loan', loan)
            level = w_float.sum(axis=1)
            ret_tmp = level.pct_change(1)  # 1 for one day lookback
            returns.append(ret_tmp)

        portf_returns = pd.concat(returns).dropna()

        if vc != 0:
            to = self.turnover(return_series, rescale=False)  # Note that rescale is hardcoded to False.
            varcost = to * vc
            portf_returns[0] -= varcost[0]
            portf_returns[varcost[1:].index] -= varcost[1:].values
        if fc != 0:
            n_days = (portf_returns.index[1:] - portf_returns.index[:-1]).to_numpy().astype('timedelta64[D]').astype(int)
            fixcost = (1 + fc) ** (n_days / n_days_per_year) - 1
            portf_returns[1:] -= fixcost

        return portf_returns




# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def get_lower_bound_index(date: str, timeline: list[str]) -> int:
    if len(timeline) == 0 or date < timeline[0]:
        return -1
    if date >= timeline[-1]:
        return len(timeline) - 1

    start = 0
    end = len(timeline)
    while start < end:
        i = (start + end) // 2
        current = timeline[i]
        if current == date:
            return i
        elif current > date:
            end = i
        elif start < i:
            start = i
        else:
            return start
    return start

def floating_weights(X, w, start_date, end_date, rescale=True):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date < X.index[0]:
        raise ValueError('start_date must be contained in dataset')
    if end_date > X.index[-1]:
        raise ValueError('end_date must be contained in dataset')

    w = pd.Series(w, index=w.keys())
    if w.isna().any():
        raise ValueError('weights (w) contain NaN which is not allowed.')
    else:
        w = w.to_frame().T
    xnames = X.columns
    wnames = w.columns

    if not all(wnames.isin(xnames)):
        raise ValueError('Not all assets in w are contained in X.')

    X_tmp = X.loc[start_date:end_date, wnames].fillna(0)
    # TODO : To extend to short positions cases when the weights can be negative
    # short_positions = wnames[w.iloc[0,:] < 0 ]
    # if len(short_positions) > 0:
    #     X_tmp[short_positions] = X_tmp[short_positions] * (-1)
    xmat = 1 + X_tmp
    # xmat.iloc[0] = w.dropna(how='all').fillna(0).abs()
    xmat.iloc[0] = w.dropna(how='all').fillna(0)
    w_float = xmat.cumprod()

    if rescale:
        w_float_long = w_float.where(w_float >= 0).div(w_float[w_float >= 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float_short = w_float.where(w_float < 0).div(w_float[w_float < 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float = pd.DataFrame(w_float_long + w_float_short, index=xmat.index, columns=wnames)

    return w_float
