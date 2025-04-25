import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('seaborn-v0_8')
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

def create_qq_plot(fd, norm):
    n = len(fd)  # number of timeframes
    cols = 3  # sub-plot columns
    rows = (n + cols - 1) // cols  # sub-plot columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Data normalization: {norm}', fontsize=16)
    
    # Flatten axes array
    axes = axes.flatten()
    
    norm_dict = {    
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(output_distribution='normal', random_state=100),
        'standard': StandardScaler(),
        'min-max': MinMaxScaler()
    }                  
    # Use standard_scaler if norm_type not in scaler_dict
    scaler = norm_dict.get(norm, StandardScaler())
        
    for i, tf in enumerate(fd):
        df = fd[tf].data.copy()
        df['r'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        if norm is not None:
            df['r_'] = scaler.fit_transform(df['r'].values.reshape(-1, 1))
        else:
            df['r_'] = df['r']
        sm.qqplot(df['r_'], line='s', ax=axes[i])
        axes[i].set_title(f"QQ-plot: {tf}")
    
    # Remove potential empty sub-plot windows
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()