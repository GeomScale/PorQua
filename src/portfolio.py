'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''



import pandas as pd
import numpy as np


# --------------------------------------------------------------------------
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

    X_tmp = X.loc[start_date:end_date, wnames].copy().fillna(0)
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


class Portfolio:

    def __init__(self,
                 rebalancing_date: str = None,
                 weights: dict = {},
                 name: str = None,
                 init_weights: dict = {}):
        self.rebalancing_date = rebalancing_date
        self.weights = weights
        self.name = name
        self.init_weights = init_weights

    @staticmethod
    def empty() -> 'Portfolio':
        return Portfolio()

    @property
    def weights(self):
        return self._weights

    def get_weights_series(self) -> pd.Series:
        return pd.Series(self._weights)

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
        return f'Portfolio(rebalancing_date={self.rebalancing_date}, weights={self.weights})'

    def float_weights(self,
                      return_series: pd.DataFrame,
                      end_date: str,
                      rescale: bool = False):
        if self.weights is not None:
            return floating_weights(X=return_series,
                                    w=self.weights,
                                    start_date=self.rebalancing_date,
                                    end_date=end_date,
                                    rescale=rescale)
        else:
            return None

    def initial_weights(self,
                        selection: list[str],
                        return_series: pd.DataFrame,
                        end_date: str,
                        rescale: bool = True) -> dict[str, float]:

        if not hasattr(self, '_initial_weights'):
            if self.rebalancing_date is not None and self.weights is not None:
                w_init = dict.fromkeys(selection, 0)
                w_float = self.float_weights(return_series=return_series,
                                             end_date=end_date,
                                             rescale=rescale)
                w_floated = w_float.iloc[-1]

                w_init.update({key: w_floated[key] for key in w_init.keys() & w_floated.keys()})
                self._initial_weights = w_init
            else:
                self._initial_weights = None  # {key: 0 for key in selection}

        return self._initial_weights

    def turnover(self, portfolio: "Portfolio", return_series: pd.DataFrame, rescale=True):
        if portfolio.rebalancing_date is not None and portfolio.rebalancing_date < self.rebalancing_date:
            w_init = portfolio.initial_weights(selection=self.weights.keys(),
                                               return_series=return_series,
                                               end_date=self.rebalancing_date,
                                               rescale=rescale)
        else:
            w_init = self.initial_weights(selection=portfolio.weights.keys(),
                                          return_series=return_series,
                                          end_date=portfolio.rebalancing_date,
                                          rescale=rescale)

        return pd.Series(w_init).sub(pd.Series(portfolio.weights), fill_value=0).abs().sum()


class Strategy:

    def __init__(self, portfolios: list[Portfolio]):
        self.portfolios = portfolios

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

    def clear(self) -> None:
        self.portfolios.clear()
        return None

    def get_rebalancing_dates(self):
        return [portfolio.rebalancing_date for portfolio in self.portfolios]

    def get_weights(self, rebalancing_date: str) -> dict[str, float]:
        for portfolio in self.portfolios:
            if portfolio.rebalancing_date == rebalancing_date:
                return portfolio.weights
        return None

    def get_weights_df(self) -> pd.DataFrame:
        weights_dict = {}
        for portfolio in self.portfolios:
            weights_dict[portfolio.rebalancing_date] = portfolio.weights
        return pd.DataFrame(weights_dict).T

    def get_portfolio(self, rebalancing_date: str) -> Portfolio:
        if rebalancing_date in self.get_rebalancing_dates():
            idx = self.get_rebalancing_dates().index(rebalancing_date)
            return self.portfolios[idx]
        else:
            raise ValueError(f'No portfolio found for rebalancing date {rebalancing_date}')

    def has_previous_portfolio(self, rebalancing_date: str) -> bool:
        dates = self.get_rebalancing_dates()
        ans = False
        if len(dates) > 0:
            ans = dates[0] < rebalancing_date
        return ans

    def get_previous_portfolio(self, rebalancing_date: str) -> Portfolio:
        if not self.has_previous_portfolio(rebalancing_date):
            return Portfolio.empty()
        else:
            yesterday = [x for x in self.get_rebalancing_dates() if x < rebalancing_date][-1]
            return self.get_portfolio(yesterday)

    def get_initial_portfolio(self, rebalancing_date: str) -> Portfolio:
        if self.has_previous_portfolio(rebalancing_date=rebalancing_date):
            initial_portfolio = self.get_previous_portfolio(rebalancing_date)
        else:
            initial_portfolio = Portfolio(rebalancing_date=None, weights={})
        return initial_portfolio

    def __repr__(self):
        return f'Strategy(portfolios={self.portfolios})'

    def number_of_assets(self, th: float = 0.0001) -> pd.Series:
        return self.get_weights_df().apply(lambda x: sum(np.abs(x) > th), axis=1)

    def turnover(self, return_series, rescale=True) -> pd.Series:
        dates = self.get_rebalancing_dates()
        turnover = {}
        for rebalancing_date in dates:
            previous_portfolio = self.get_previous_portfolio(rebalancing_date)
            current_portfolio = self.get_portfolio(rebalancing_date)
            turnover[rebalancing_date] = current_portfolio.turnover(portfolio=previous_portfolio,
                                                                    return_series=return_series,
                                                                    rescale=rescale)
        return pd.Series(turnover)

    def simulate(self,
                 return_series=None,
                 fc: float = 0,
                 vc: float = 0,
                 n_days_per_year: int = 252) -> pd.Series:

        rebdates = self.get_rebalancing_dates()
        ret_list = []
        for rebdate in rebdates:
            next_rebdate = rebdates[rebdates.index(rebdate) + 1] if rebdate < rebdates[-1] else return_series.index[-1]

            portfolio = self.get_portfolio(rebdate)
            w_float = portfolio.float_weights(return_series=return_series,
                                              end_date=next_rebdate,
                                              rescale=False)  # Note that rescale is hardcoded to False.
            short_positions = list(filter(lambda x: x < 0, portfolio.weights.values()))
            long_positions = list(filter(lambda x: x >= 0, portfolio.weights.values()))
            margin = abs(sum(short_positions))
            cash = max(min(1 - sum(long_positions), 1), 0)
            loan = 1 - (sum(long_positions) + cash) - (sum(short_positions) + margin)
            w_float.insert(0, 'margin', margin)
            w_float.insert(0, 'cash', cash)
            w_float.insert(0, 'loan', loan)
            level = w_float.sum(axis=1)
            ret_tmp = level.pct_change(1)  # 1 for one day lookback
            ret_list.append(ret_tmp)

        portf_ret = pd.concat(ret_list).dropna()

        if vc != 0:
            to = self.turnover(return_series=return_series,
                               rescale=False)  # Note that rescale is hardcoded to False.
            varcost = to * vc
            portf_ret[0] -= varcost[0]
            portf_ret[varcost[1:].index] -= varcost[1:].values
        if fc != 0:
            n_days = (portf_ret.index[1:] - portf_ret.index[:-1]).to_numpy().astype('timedelta64[D]').astype(int)
            fixcost = (1 + fc) ** (n_days / n_days_per_year) - 1
            portf_ret[1:] -= fixcost

        return portf_ret
