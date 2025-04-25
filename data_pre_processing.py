import numpy as np
import pandas as pd
import pandas_ta as ta
from pylab import plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer  # Normalization
from scipy.stats import skew, kurtosis # stats for auto-selection of normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  #Principal Component Analysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
#import shap
from sklearn.ensemble import RandomForestClassifier
from financial_data import FinancialData  # own class that imports financial data

class DataPreProcessing(FinancialData):
    def __init__(self, base_url, symbol='EUR_USD',
                 start='2020-01-01', end='2025-03-31', timeframe=None, 
                 bins=6, clusters=10, lags=10, normalize='auto',
                 tt_split=0.8, rolling=False, rolling_splits=None, 
                 max_train_size=None, test_size=20, pca=None, verbose=False):
        # Initialise attributes
        super().__init__(base_url, symbol, timeframe, verbose)
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.bins = bins
        self.clusters = clusters
        self.lags = lags
        self.norm = normalize
        self.tt_split = tt_split
        self.pca = pca
        self.rolling = rolling # possibility for rolling train/test split
        self.rolling_splits = rolling_splits  # Number of splits for the rolling train/test
        self.max_train_size = max_train_size  # Maximium number of bars used for training
        self.test_size = test_size  # number of bars used for test
       
        # 1. ADD GENERAL FEATURES (excl. bins, regimes and lags which require normailzation)
        self._add_features()
        
        # 2. CREATE TRAIN/TEST-INDICES FOR ROLLING AND FIXED WINDOWS
        self._create_splits()  # returns lists containing dates for train/test splits
        
        # Initialization
        self.X_train_splits = []
        self.y_train_splits = []
        self.X_test_splits = []
        self.y_test_splits = []
        self.features = self._base_features.copy()
        self.features.extend(['bin', 'regime'])
        
        # 3. LOOP THROUGH TRAIN/TEST INDEX LISTS        
        for i in range(len(self.train_index)):
            # Split dataset based on base_features (from _add_features()) before normalization
            self.X_train = self.data.loc[self.train_index[i], self._base_features].copy()
            self.X_test = self.data.loc[self.test_index[i], self._base_features].copy()
            self.train_rets = self.data.loc[self.train_index[i], 'r'].copy() # used for _add_bins_regimes
            self.test_rets = self.data.loc[self.test_index[i], 'r'].copy() # used for _add_bins_regimes
            
            # Normalize X_train and X_test
            self._normalize()
            self._add_bins_regime()
            self._add_lags()
            # Split data set again after adding lags
            self.X_train_norm = self._lags_df.loc[self.train_index[i], self.lag_features].dropna()
            self.X_test_norm = self._lags_df.loc[self.test_index[i], self.lag_features]

            if self.pca:
                self._pca()               

            # Adjust y_train, y_test to match X_train, X_test indices in case of dropnas
            train_index = self.X_train_norm.index
            y_train = self.data.loc[train_index, 'd'].copy()
            y_train = np.where(y_train == 1, 1, 0)  # convert to 0, 1 binary labels
            test_index = self.X_test_norm.index
            y_test = self.data.loc[test_index, 'd'].copy()
            y_test = np.where(y_test == 1, 1, 0)  # convert to 0, 1 binary labels
            
            # Add X_train, y-train, X_test and y_test to lists
            self.X_train_splits.append(self.X_train_norm)
            self.X_test_splits.append(self.X_test_norm)
            self.y_train_splits.append(y_train)
            self.y_test_splits.append(y_test)

        # Set last train/test split to 'default' attribute
        self.X_train = self.X_train_splits[-1]
        self.y_train = self.y_train_splits[-1]
        self.X_test = self.X_test_splits[-1]
        self.y_test = self.y_test_splits[-1]
        
    def _add_features(self):
        # DAILY RETURNS AND DIRECTION
        self.data['r'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['d'] = np.where(self.data['r'] > 0, 1, -1)
        # daily returns that will be normalized and used as features
        self.data['rets'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # TREND
        self.data.ta.adx(append=True) # Average Directional Movement Index (14)
        self.data.ta.aroon(append=True) # Aroon Indicator (25)
    
        # ROLLING FEATURES
        self.data['r_5ma'] = self.data['r'].rolling(window=5).mean()
        self.data['r_5std'] = self.data['r'].rolling(window=5).std()
        self.data['price_ma10'] = self.data['close'].rolling(window=10).mean()
        self.data['price_ma20'] = self.data['close'].rolling(window=20).mean()
        self.data['price_ma_ratio'] = self.data['close'] / self.data['price_ma10']       

        # MOMENTUM
        self.data.ta.ao(append=True)  # Awsome Oscillator (5, 34)
        self.data.ta.ppo  # Percentage Price Oscillator (26, 12, 9)
        self.data.ta.rsi(length=9, append=True)  # Relative Strength Index (9)
        self.data.ta.rsi(length=14, append=True)  # Relative Strength Index (14)
        self.data.ta.macd(append=True)
    
        # VOLATILITY
        self.data.ta.atr(append=True)  # Average True Range
        self.data['atr_volatility'] = self.data['ATRr_14'].rolling(window=20).std()  # ATR volatility
        self.data.ta.bbands(append=True)  # Bollinger Bands (20, 2)
        self.data.drop(columns=[col for col in self.data.columns if 'BBP' in col], inplace=True)

        # VOLATILITY SPIKE FEATURE
        self.data['atr_spike'] = self.data['ATRr_14'] / self.data['ATRr_14'].rolling(20).mean()

        # FEATURES CHANGES
        self.data['RSI_14_change'] = self.data['RSI_14'] - self.data['RSI_14'].shift(1)
        self.data['AO_change'] = self.data['AO_5_34'] - self.data['AO_5_34'].shift(1)
        self.data['ADX_change'] = self.data['ADX_14'] - self.data['ADX_14'].shift(1)

        # TREND SLOPE ANGLE
        self.data['ma10_slope'] = np.arctan((self.data['price_ma10'].shift(1) 
                                             - self.data['price_ma10'].shift(5)) / 4)
        self.data['ma20_slope'] = np.arctan((self.data['price_ma20'].shift(1) 
                                             - self.data['price_ma20'].shift(5)) / 4)
        self.data['rsi14_slope'] = np.arctan((self.data['RSI_14'].shift(1) 
                                              - self.data['RSI_14'].shift(5)) / 4)
        self.data['atr_slope'] = np.arctan((self.data['ATRr_14'].shift(1) 
                                            - self.data['ATRr_14'].shift(5)) / 4)
        self.data['r_5ma_slope'] = np.arctan((self.data['r_5ma'].shift(1) 
                                              - self.data['r_5ma'].shift(5)) / 4)

        # PRICE-ACTION
        self.data['co'] = self.data['close'] / self.data['open'] - 1
        self.data['hl'] = self.data['high'] / self.data['low'] - 1
        self.data['oc'] = self.data['open'] / self.data['close'] - 1
        self.data['ch'] = self.data['close'] / self.data['high'] - 1
        self.data['cl'] = self.data['close'] / self.data['low'] - 1
        self.data['ohlc4'] = (self.data['open'] + self.data['high'] 
                              + self.data['low'] + self.data['close']) / 4
       
        # VOLUME
        self.data.ta.pvo(append=True)  # Percentage Volume Oscillator (26, 12, 9)
        self.data.ta.cmo(append=True)  # Chaikin Money Flow (20)

        # CROSS FEATURES
        self.data['momentum_volatility'] = self.data['RSI_14'] * self.data['ATRr_14']
        self.data['price_rsi'] = self.data['close'] / self.data['RSI_14']
        self.data['price_atr'] = self.data['close'] / self.data['ATRr_14']
        self.data['adx_rsi_interaction'] = self.data['ADX_14'] * self.data['RSI_14']
        self.data['range_atr'] = (self.data['high'] - self.data['low']) / self.data['ATRr_14']
        self.data['price_ma10_slope'] = self.data['price_ma10'] * self.data['ma10_slope']
        self.data['momentum_price_action'] = self.data['RSI_14'] * self.data['co']  # Price action (close/open)
        
        # CALENDAR
        day = self.data.index.weekday
        tod = self.data.index.hour + self.data.index.minute / 60  # time of day

        self.data['day_sin'] = np.sin(2 * np.pi * day / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * day / 7)
        self.data['tod_sin'] = np.sin(2 * np.pi * tod / 24)
        self.data['tod_cos'] = np.cos(2 * np.pi * tod / 24)

        # Adjust length of dataframe
        self.data.dropna(inplace=True)
        if self.start < self.data.index[self.lags]:
            if self.verbose:
                print(80 * '-')
                print('The selected start date is earlier than the available data.')
                print(f'The start date is set to the first available data point: {self.data.index[self.lags]}')
                print(80 * '-')
            self.start = self.data.index[0]
        if self.end > self.data.index[-1]:
            if self.verbose:
                print(80 * '-')
                print('The selected end date is later than the available data.')
                print(f'The end date is set to the last available data point: {self.data.index[-1]}')
                print(80 * '-')
            self.end = self.data.index[-1]
        start_index = self.data.index.get_indexer([self.start], method='nearest')[0]
        end_index = self.data.index.get_indexer([self.end], method='nearest')[0]
        # add 'lags' datapoints in the beginning if such data is available
        start_index = max(0, start_index - self.lags)
        self.data = self.data.iloc[start_index:end_index].copy()
        # Features list
        exclude = ['open', 'high', 'low', 'close', 'volume', 'd', 'r']
        self._base_features = [x for x in self.data.columns if x not in exclude]

    def _create_splits(self):
        # Initialize train_index and test_index lists
        self.train_index = []
        self.test_index = []
        
        # 2.a. Rolling train/test split
        if self.rolling:
            # Default values to be used if none are given:
            if self.rolling_splits is None:
                self.rolling_splits = min(20, int(len(self.data)/50))
            if self.test_size is None:
                self.test_size = max(20, int(len(self.data)/100))
                
            ts = TimeSeriesSplit(n_splits=self.rolling_splits, 
                                 max_train_size=self.max_train_size,
                                 test_size=self.test_size)
            data_index = self.data.index
            for fold, (train_index, test_index) in enumerate(ts.split(data_index)):                   
                # correct train_index for the removal of lags-created NaN's
                if train_index[0] > 0:
                    if train_index[0] - self.lags > 0:
                        tmp = np.sort(train_index[0] - np.arange(1, self.lags + 1))
                    else:
                        tmp = np.arange(train_index[0])
                    train_index = np.sort(np.concatenate([train_index, tmp]))
    
                self.train_index.append(data_index[train_index])
                self.test_index.append(data_index[test_index])
        
        # 2.b. Fixed train/test-split
        else:
            if isinstance(self.tt_split, float):
                # If a float is given, we convert it to s data
                split_date = self.start + self.tt_split * (self.end - self.start) 
            else:
                split_date = pd.to_datetime(self.tt_split)
            
            split_date = self.data.index[self.data.index <= split_date].max()
            split_index = self.data.index.get_loc(split_date)
            self.train_index.append(self.data.index[: split_index])
            self.test_index.append(self.data.index[split_index:])

    def _normalize(self):    
        self.X_train_norm = pd.DataFrame()
        self.X_test_norm = pd.DataFrame()
        
        if self.norm == 'auto':
            self.scalers = self._automatic_scaler_selection()
            for feature, scaler in self.scalers.items():
                scaled_feature = scaler.fit_transform(
                    self.X_train[feature].values.reshape(-1, 1))
                self.X_train_norm[feature] = pd.DataFrame(scaled_feature, index=self.X_train.index)
                scaled_feature = scaler.transform(
                    self.X_test[feature].values.reshape(-1, 1))
                self.X_test_norm[feature] = pd.DataFrame(scaled_feature, index=self.X_test.index)
        else:
            norm_dict = {    
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(output_distribution='normal', random_state=100),
                'standard': StandardScaler(),
                'min-max': MinMaxScaler()
            }                  
            # Use standard_scaler if norm_type not in scaler_dict
            scaler = norm_dict.get(self.norm, StandardScaler())
            self.X_train_norm = scaler.fit_transform(self.X_train)
            self.X_test_norm = scaler.transform(self.X_test)
                    
    def _automatic_scaler_selection(self):
        # Dictionary to store chosen scalers for each feature
        scalers = {}
    
        for feature in self.X_train.columns:
            #feature_data = self.X_train[feature]
            
            # Calculate statistics
            mean = self.X_train[feature].mean()
            std = self.X_train[feature].std()
            feature_skew = skew(self.X_train[feature])
            feature_kurtosis = kurtosis(self.X_train[feature])
            
            # Define the scaler choice criteria based on statistics
            if np.abs(feature_skew) > 1 or np.abs(feature_kurtosis) > 3:  # If data is skewed or has high kurtosis
                # If the data is very skewed or has extreme outliers, use QuantileTransformer to adjust the distribution
                scaler = QuantileTransformer(output_distribution='normal')  # Or 'uniform' for uniform distribution
            elif np.abs(feature_skew) > 1 or np.abs(feature_kurtosis) > 3:  # If data is skewed or has high kurtosis
                # If data is skewed or has high kurtosis but doesn't require full distribution change, use RobustScaler
                scaler = RobustScaler()
            elif std < 1e-5:  # If standard deviation is very low, Min-Max might be better
                scaler = MinMaxScaler()
            else:  # Otherwise, StandardScaler is a safe choice
                scaler = StandardScaler()
    
            # Store the selected scaler for each feature
            scalers[feature] = scaler
        return scalers
         
    def _add_bins_regime(self):
        # BINS - QUANTILES NEED TO BE CALCULATED ONLY BASED ON TRAINING DATA
        quant = []
        step = 1 / self.bins
        for n in range(1, self.bins):
            q = n * step
            quant.append(self.train_rets.quantile(q))
               
        self.X_train_norm['bin'] = np.digitize(self.train_rets, bins=quant, right=True) / self.bins # 0-1 normalization
        self.X_test_norm['bin'] = np.digitize(self.test_rets, bins=quant, right=True) / self.bins  # 0-1 normalization
            
        # KMEANS REGIME DETECTION
        regime_features = ['AO_5_34', 'RSI_14', 'ADX_14', 'price_ma_ratio', 'atr_spike']
        
        # 1. Select regime training data without NaNs
        self.regime_data = self.X_train_norm[regime_features].copy()
    
        # 2. Fit kmeans
        kmeans = KMeans(n_clusters=self.clusters, random_state=100).fit(self.regime_data)
    
        # 3. Predict on full data (fill missing values)
        regime_input_train = self.X_train_norm[regime_features].copy()
        regime_input_test = self.X_test_norm[regime_features].copy()
        regime_input_train = regime_input_train.ffill().bfill()  # fill missing values robustly
        regime_input_test = regime_input_test.ffill().bfill()  # fill missing values robustly
        regime_labels_train = kmeans.predict(regime_input_train)
        regime_labels_test = kmeans.predict(regime_input_test)
    
        # 4. Store regime labels and perform min-max normalization
        self.X_train_norm['regime'] = pd.Series(regime_labels_train, index=self.X_train_norm.index) \
        / self.clusters
        self.X_test_norm['regime'] = pd.Series(regime_labels_test, index=self.X_test_norm.index) \
        / self.clusters
           
    def _add_lags(self):
        # Only add lags to the normalized DataFrame
        self._lags_df = pd.concat([self.X_train_norm, self.X_test_norm], axis=0)  # temporary dataframe to avoid nan's in X_test
        cols = []
        no_lag_features = ['day_sin', 'day_cos', 'tod_sin', 'tod_cos']  # features where wo only use lag1
        for col in self.features:
            for lag in range(1, self.lags + 1):
                if col not in no_lag_features:
                    col_ = f'{col}_lag{lag}'
                    cols.append(col_)
                    self._lags_df[col_] = self._lags_df[col].shift(lag)
                elif lag == 1:
                    col_ = f'{col}_lag{lag}'
                    cols.append(col_)
                    self._lags_df[col_] = self._lags_df[col].shift(lag)       
        
        self.lag_features = cols  # set the new features to the lagged data to avoid foresight
    
    def _pca(self):
        if self.rolling:
            raise ValueError('PCA cannot be used with rolling split.')

        pca = PCA(n_components=self.pca, random_state=100)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
       
    def select_features(self, top_k=50, method='mutual_info', model=None, plot=True):
        if self.rolling:
            raise ValueError('select_features cannot be used with rolling split.')
            
        if method == 'mutual_info':
            mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=100)
            feature_scores = pd.Series(mi_scores, index=self.lag_features)
            feature_scores.sort_values(ascending=False, inplace=True)
        
        elif method == 'shap':
            # Use default model if no model is specified
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=100)
            model.fit(self.X_train, self.y_train)            
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_train, check_additivity=False)
 
            # Average SHAP-importance
            shap_importance = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
            feature_scores = pd.Series(shap_importance, index=self.lag_features)
            feature_scores.sort_values(ascending=False, inplace=True)
            
            if plot:
                # Top K plot
                top_k = min(top_k, len(feature_scores)) if top_k else 20
                shap.summary_plot(shap_values[:, :, 1], self.X_train, feature_names=self.lag_features, max_display=top_k)
                
        else:
            raise ValueError("Method must be either 'mutual_info' or 'shap'")
    
        if method != 'shap' and plot:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=feature_scores.head(top_k), y=feature_scores.head(top_k).index, color='skyblue')
            plt.xlabel(f'{method.replace("_", " ").title()} Score')
            plt.ylabel('Feature')
            plt.title(f'Top {top_k} Features by {method.replace("_", " ").title()}')
            plt.tight_layout()
            plt.show()
    
        # Update features based on selection
        self.lag_features = feature_scores.head(top_k).index.tolist()
        self.X_train = self.X_train[self.lag_features]#.to_numpy()
        self.X_test = self.X_test[self.lag_features]#.to_numpy()
      
    def plot_KMean(self):
        # Finding optimal number of clusters using Elbow method
        inertia = []
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=100)
            kmeans.fit(self.regime_data)
            inertia.append(kmeans.inertia_)
        # Plot the Elbow method
        plt.plot(range(1, 20), inertia, marker='o')
        plt.title('Elbow Method for KMean to determine optimal number of clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()

    def plot_PCA(self):
        pca = PCA().fit(self.X_train)
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlim((0, len(self.lag_features) / 5))
        plt.ylim(0.5, 1.01)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.show()
