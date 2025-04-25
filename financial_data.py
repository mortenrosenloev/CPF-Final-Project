import pandas as pd
import numpy as np

class FinancialData():
    ''' Class that imports 15min o, h, l, c price data and volume for FX-pairs or indices. Possible symbols are:

    Attributes
    ==========
    symbol: str

    Methods
    =======
    _retrieve_data:
        retrieves the base data
    _prepare_data:
        prepares log-returns and directional movements to the dataframe (1 = up, 0 = down)
    
    '''
    
    def __init__(self, base_url, symbol='EUR_USD', 
                 timeframe=None, verbose=False):
        self.base_url = base_url
        self.symbol = symbol
        self.tf = timeframe
        self.verbose = verbose
        self.data = pd.DataFrame()
        self._retrieve_data()
        self._calculate_trading_cost()
        
    def _retrieve_data(self):
        suffix = f'_2020-01-01_2025-03-31_M15_M'
        filename = f'{self.symbol}{suffix}.csv'
        file = self.base_url + filename
        raw = pd.read_csv(file, index_col=0, parse_dates=True)
        del raw['complete']
                       
        if self.tf:
            self.data['open'] = raw['o'].resample(self.tf, label='left').first().ffill()
            self.data['high'] = raw['h'].resample(self.tf, label='left').max().ffill()
            self.data['low'] = raw['l'].resample(self.tf, label='left').min().ffill()
            self.data['close'] = raw['c'].resample(self.tf, label='left').last().ffill()
            self.data['volume'] = raw['volume'].resample(self.tf, label='left').sum().ffill()
            self.data['volume'] = self.data['volume'].astype(float)
            # We only use data where there is a trading volume
            self.data = self.data.loc[self.data['volume'] > 0]
        else:
            self.data = raw.copy()
            self.data.columns = ['open', 'high', 'low', 'close', 'volume']
        self.data.dropna(inplace=True)

    def _calculate_trading_cost(self):
        # retrieve Ask and Bid prices for calculation of
        # average spread and log-adjusted trading cost
        raw = {}
        for price in ['A', 'B']:
            suffix = f'_2020-01-01_2025-03-31_M15_{price}'
            filename = f'{self.symbol}{suffix}.csv'
            file = self.base_url + filename
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            raw[price] = df['c'].copy()
        self.spread = (raw['A'] - raw['B']).mean()
        self.tc = np.log(raw['A'] / raw['B']).mean()