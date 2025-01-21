

import polars as pl
import numpy as np  
import pandas as pd


def generate_seasonal_features(df, X_years=5, ewma_span=10, ret_type='abs', rm_outliers=0, min_lb=0, lag_day=0, add_days=1):
    """
    Generates seasonal features for each ISO week over the last X_years.

    Parameters:
    - df: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close'].
    - X_years: Lookback period in years.
    - ewma_span: EWMA smoothing period.
    - ret_type: 'abs' for absolute returns or 'pct' for percentage returns.
    - rm_outliers: Threshold for z-scores to remove outliers (e.g., rm_outliers=5 removes z-scores > 5).
    - min_lb: Minimum lookback period for rolling calculations.

    Returns:
    - Dictionary with keys as ISO weeks ('w1', 'w2', ...) and values as feature DataFrames.
    """
    # Ensure min_lb is valid
    rolling_window = 52 * X_years  # Weekly data, so 52 weeks per year
    if min_lb == 0 or min_lb > rolling_window:
        min_lb = X_years

    # Ensure df is a Pandas DataFrame
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Get the last date in the DataFrame
    last_date = df.index[-1]

    # Calculate the additional business days
    additional_dates = [last_date + pd.tseries.offsets.BDay(i) for i in range(1, add_days)]

    # Store the original index name (optional if index name is preserved)
    original_index_name = df.index.name

    # Create a new index that includes the additional business days
    new_index = df.index.union(additional_dates)

    # Reindex the DataFrame to include the new index
    df = df.reindex(new_index)

    # Forward-fill the data
    df = df.ffill().infer_objects(copy=False)

    # Restore the index name (just in case it gets lost during reindexing)
    df.index.name = original_index_name

    # Add ISO week and ISO year columns
    df['ISO_Week'] = df.index.isocalendar().week
    df['ISO_Year'] = df.index.isocalendar().year

    if lag_day:
        df['ISO_Week'] = df['ISO_Week'].shift(lag_day)
        df['ISO_Year'] = df['ISO_Year'].shift(lag_day)

    # Calculate weekly returns
    if ret_type == 'abs':
        df['Daily_Return'] = df['Close'] - df['Close'].shift(1)
        df['Weekly_Return'] = df.groupby(['ISO_Year', 'ISO_Week'])['Daily_Return'].transform('sum')
    else:
        df['Daily_Return'] = (df['Close'] / df['Close'].shift(1)) - 1
        df['Weekly_Return'] = df.groupby(['ISO_Year', 'ISO_Week'])['Daily_Return'].transform(lambda x: (1 + x).prod() - 1)

    df.dropna(inplace=True)
    '''
    # Aggregate weekly data
    weekly_data = df.groupby(['ISO_Year', 'ISO_Week']).agg({
        'Weekly_Return': 'first',
        'ISO_Week': 'first',
        'ISO_Year': 'first'
    })

    # Ensure weekly data keeps the first date of the week as the index
    weekly_data['Date'] = df.groupby(['ISO_Year', 'ISO_Week']).apply(lambda x: x.index[0]).values
    weekly_data.set_index('Date', inplace=True)
    '''

    # Aggregate weekly data
    weekly_data = df.groupby(['ISO_Year', 'ISO_Week']).agg({
        'Weekly_Return': 'first',
        'ISO_Week': 'first',
        'ISO_Year': 'first'
    })

    # Ensure weekly data keeps the first date of the week as the index
    # Ensure weekly data keeps the first date of the week as the index
    weekly_data['Date'] = (
        df.groupby(['ISO_Year', 'ISO_Week'], group_keys=False)
        .apply(lambda x: x.index[0] if len(x) > 0 else pd.NaT)  # Explicitly work on the group's index
        .reset_index(drop=True)  # Drop grouping columns
        .values
    )
    weekly_data.set_index('Date', inplace=True)

    # Dictionary to hold features for each week
    features_dict = {}

    for week in range(1, 53):
        week_key = f'w{week}'
        df_week = weekly_data[weekly_data['ISO_Week'] == week].copy()

        if df_week.empty:
            continue

        if rm_outliers != 0:
            rolling_mean = df_week['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).mean()
            rolling_std = df_week['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).std()
            z_scores = (df_week['Weekly_Return'] - rolling_mean) / rolling_std
            outlier_mask = np.abs(z_scores) > rm_outliers
            df_week.loc[outlier_mask, 'Weekly_Return'] = np.nan

        df_week['Rolling_Mean_Return'] = df_week['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).mean()
        df_week['Rolling_Hit_Rate'] = df_week['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).apply(lambda x: np.mean(x > 0), raw=True)
        df_week['Rolling_Skew'] = df_week['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).skew()

        df_week['All_Weeks_Mean_Return'] = weekly_data['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).mean().reindex(df_week.index)
        df_week['All_Weeks_Hit_Rate'] = weekly_data['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).apply(lambda x: np.mean(x > 0), raw=True).reindex(df_week.index)
        df_week['All_Weeks_Skew'] = weekly_data['Weekly_Return'].rolling(window=rolling_window, min_periods=min_lb).skew().reindex(df_week.index)

        for col in ['Rolling_Mean_Return', 'Rolling_Hit_Rate', 'Rolling_Skew',
                    'All_Weeks_Mean_Return', 'All_Weeks_Hit_Rate', 'All_Weeks_Skew']:
            df_week[f'{col}_EWMA'] = df_week[col].ewm(span=ewma_span, adjust=False).mean()

        rolling_std_ret = df_week['All_Weeks_Mean_Return'].rolling(window=rolling_window, min_periods=min_lb).std()
        rolling_std_hr = df_week['All_Weeks_Hit_Rate'].rolling(window=rolling_window, min_periods=min_lb).std()

        df_week['ret_zs'] = (df_week['Rolling_Mean_Return'] - df_week['All_Weeks_Mean_Return']) / rolling_std_ret
        df_week['hr_zs'] = (df_week['Rolling_Hit_Rate'] - df_week['All_Weeks_Hit_Rate']) / rolling_std_hr

        df_week['ret_zs'] = df_week['ret_zs'].replace([np.inf, -np.inf], np.nan)
        df_week['hr_zs'] = df_week['hr_zs'].replace([np.inf, -np.inf], np.nan)

        for col in ['ret_zs', 'hr_zs']:
            df_week[f'{col}_ewma'] = df_week[col].ewm(span=ewma_span, adjust=False).mean()

        features_dict[week_key] = df_week[
            [
                'ISO_Year', 'ISO_Week', 'Rolling_Mean_Return', 'Rolling_Hit_Rate', 'Rolling_Skew',
                'All_Weeks_Mean_Return', 'All_Weeks_Hit_Rate', 'All_Weeks_Skew',
                'Rolling_Mean_Return_EWMA', 'Rolling_Hit_Rate_EWMA', 'Rolling_Skew_EWMA',
                'All_Weeks_Mean_Return_EWMA', 'All_Weeks_Hit_Rate_EWMA', 'All_Weeks_Skew_EWMA',
                'ret_zs', 'hr_zs', 'ret_zs_ewma', 'hr_zs_ewma'
            ]
        ]

    return features_dict, weekly_data, df


import matplotlib.pyplot as plt

def plot_weekly_returns(weekly_data, iso_week, num_years=10, asst=''):
    """
    Plots the returns for a specific ISO week over the last num_years.

    Parameters:
    - weekly_data: DataFrame containing 'Weekly_Return', 'ISO_Week', and 'ISO_Year'.
    - iso_week: The ISO week number to filter (e.g., 52 for week 52).
    - num_years: Number of years to look back.

    Returns:
    - A bar plot of the last num_years of ISO week returns.
    """
    # Filter the data for the specified ISO week
    week_data = weekly_data[weekly_data['ISO_Week'] == iso_week].copy()
    
    # Sort by ISO year and take the last num_years
    recent_week_data = week_data.sort_values('ISO_Year').tail(num_years)
    
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        recent_week_data['ISO_Year'], 
        recent_week_data['Weekly_Return'], 
        color='skyblue', 
        edgecolor='black'
    )

    num_years = np.min([num_years, len(recent_week_data)])
    plt.title(f"Last {num_years} Years of Week {iso_week} Returns {asst}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Weekly Return (%)", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example Usage:
# Plot the last 10 years of week 52 returns
#plot_weekly_returns(wklyret, iso_week=50, num_years=20, asst=tick)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_heatmap(feats, column_name, asst=''):
    """
    Plots a heatmap of a selected column from the feats dictionary.

    Parameters:
    - feats: Dictionary containing DataFrames for each week (e.g., feats['w1'], feats['w2'], ...).
    - column_name: The column name to use for heatmap values (e.g., 'Rolling_Mean_Return_EWMA').

    Returns:
    - Heatmap plot showing weeks as rows, years as columns, and values from the selected column.
    """
    # Initialize a DataFrame to hold the heatmap data
    heatmap_data = pd.DataFrame()

    # Populate the heatmap data
    for week_key, df in feats.items():
        week_number = int(week_key[1:])  # Extract the week number from the key (e.g., 'w1' -> 1)
        if column_name in df.columns:
            # Add the week column to the DataFrame
            df['Week'] = week_number
            # Append to the heatmap_data DataFrame
            heatmap_data = pd.concat([heatmap_data, df[['Week', 'ISO_Year', column_name]]], axis=0)

    # Pivot the DataFrame to get weeks as rows and years as columns
    heatmap_pivot = heatmap_data.pivot(index='Week', columns='ISO_Year', values=column_name)

    # Drop columns that are entirely NaN
    heatmap_pivot.dropna(axis=1, how='all', inplace=True)

    # Plot the heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        heatmap_pivot,
        cmap="coolwarm",
        annot=True,  # Display values in cells
        fmt=".2f",  # Format for annotation
        cbar_kws={'label': column_name},
        linewidths=0.5,
        annot_kws={"size": 8},  # Set font size for annotations
    )
    plt.title(f"Heatmap of {column_name} by Week and Year ({asst})", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Week of the Year", fontsize=12)
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()

# Example Usage:
# Assuming feats is your dictionary of DataFrames and you want to plot 'Rolling_Mean_Return_EWMA'
#plot_feature_heatmap(feats, column_name='ret_zs', asst=tick)
#plot_feature_heatmap(feats, column_name='Rolling_Hit_Rate_EWMA', asst=tick)
#plot_feature_heatmap(feats, column_name='Rolling_Hit_Rate', asst=tick)


def wkly_seasn_simple(
    df: pl.DataFrame,
    X_years: int = 5,
    ewma_span: int = 10,
    ret_type: str = 'pct',
    rm_outliers: float = 4.5,
    min_lb: int = 0,
    hr_threshold_long: list = [],
    hr_threshold_short: list = [],
    ret_threshold: list = [],
    risk_scale: list = [],
    wks_to_trade: list = [],
    vol_filter: float = (0,0,0),
    trend_filter: tuple = ('', 0),
    lag_day: int = 0,
    mthly_scale: list = [],
    trade_direction: str = 'both',
    add_days: int = 1
) -> pl.DataFrame:
    """
    Implements a trading rule based on weekly seasonality.

    Parameters:
    - df: Polars DataFrame with 'Date', 'Open', 'High', 'Low', 'Close' columns.
    - X_years: Lookback period for seasonality.
    - ewma_span: EWMA smoothing period.
    - ret_type: 'abs' for absolute returns, 'pct' for percentage returns.
    - rm_outliers: Threshold for outlier removal.
    - min_lb: Minimum lookback period for rolling calculations.
    - hr_threshold_long: Threshold for hit rate z-score to go long.
    - hr_threshold_short: Threshold for hit rate z-score to go short.
    - ret_threshold: Threshold for return z-score to validate the signal.

    Returns:
    - Polars DataFrame with columns: 'Signal', 'TradeEntry', 'TradeExit', 'InTrade'.
    """

    if len(hr_threshold_long) != len(hr_threshold_short) != len(ret_threshold):
        raise ValueError("hr_threshold_long, hr_threshold_short, and ret_threshold must be the same length.")

    if len(risk_scale) != len(ret_threshold) or len(risk_scale) == 0:
        risk_scale = [1] * len(ret_threshold)

    if len(mthly_scale) != 12:
        mthly_scale = [1] * 12
    
    # Generate seasonal features
    feats, wklyret, df = generate_seasonal_features(
        df, 
        X_years=X_years, 
        ewma_span=ewma_span, 
        ret_type=ret_type, 
        rm_outliers=rm_outliers, 
        min_lb=min_lb,
        lag_day=lag_day,
        add_days=add_days
    )
    

    if wks_to_trade == []:
        wks_to_trade = list(feats.keys())
    else:
        wks_to_trade = [f'w{w}' for w in wks_to_trade] 


    if vol_filter != (0,0,0):
        df['Volatility'] = df['Close'].pct_change().rolling(window=vol_filter[0]).std() * np.sqrt(252)
        df['Vol ZScore'] = (df['Volatility'] - df['Volatility'].rolling(window=vol_filter[1]).mean()) / df['Volatility'].rolling(window=vol_filter[1]).std()
        df['Vol Filter'] = df['Vol ZScore'] < vol_filter[2]
    else:
        df['Vol Filter'] = True


    if trend_filter[0] != '':
        if trend_filter[0] == 'MA':
            df['MA'] = df['Close'].rolling(window=trend_filter[1]).mean()
            df['tf_long'] = (df['Close'].shift(1) > df['MA']).shift(2)
            df['tf_short'] = df['Close'].shift(1) < df['MA'].shift(2)

        elif trend_filter[0] == 'EMA':
            df['MA'] = df['Close'].ewm(span=trend_filter[1]).mean()
            df['tf_short'] = df['Close'].shift(1) < df['MA'].shift(2)
            df['tf_long'] = df['Close'].shift(1) > df['MA'].shift(2)
    else:
        df['tf_long'] = True
        df['tf_short'] = True

    # Initialize columns for trading rule
    df['Signal'] = 0
    df['TradeEntry'] = None
    df['TradeExit'] = None
    df['InTrade'] = 0

    in_trade = False
    trade_start_date = None
    trade_end_date = None

    # Get a list of all dates for indexing
    all_dates = df.index

    for date in all_dates:
        row = df.loc[date]
        prev_row = df.loc[df.index[df.index.get_loc(date) - 1]]
        current_week = row['ISO_Week']
        current_year = row['ISO_Year']
        current_month = date.month
        
        week_key = f'w{current_week}'
        # Skip if the current week has no seasonal data
        if week_key not in feats:
            continue

        if week_key not in wks_to_trade:
            continue

        # Get the seasonal data for the current week up to the previous year
        seasonal_data = feats[week_key]
        seasonal_data = seasonal_data[seasonal_data['ISO_Year'] < current_year]

        # Skip if insufficient historical data
        if seasonal_data.empty:
            continue

        # Calculate current z-scores
        '''
        try:
            hr_zs = seasonal_data['Rolling_Hit_Rate_EWMA'].iloc[-2]
            ret_zs = seasonal_data['ret_zs'].iloc[-2]
        except:
            hr_zs = seasonal_data['Rolling_Hit_Rate_EWMA'].iloc[-1]
            ret_zs = seasonal_data['ret_zs'].iloc[-1]         
        '''   
        hr_zs = seasonal_data['Rolling_Hit_Rate_EWMA'].iloc[-1]
        ret_zs = seasonal_data['ret_zs'].iloc[-1]       
        # Determine trade direction (long or short)
        signal = 0

        if trade_direction in ['long', 'both']:
            for i in range(len(hr_threshold_long)):
                if hr_zs > hr_threshold_long[i] and ret_zs > ret_threshold[i] and row['Vol Filter'] and row['tf_long']:
                    signal = risk_scale[i]
        if trade_direction in ['short', 'both']:
            for i in range(len(hr_threshold_short)):
                if hr_zs < hr_threshold_short[i] and ret_zs < -ret_threshold[i] and row['Vol Filter'] and row['tf_short']:
                    signal = -risk_scale[i]

        if trade_direction not in ['long', 'short', 'both']:
            raise ValueError("Invalid trade_direction. Must be 'long', 'short', or 'both'.")

        # Only update signal if not already in a trade
        if not in_trade and signal != 0:
            # Enter trade at the close of the first trading day of the current week
            current_week_data = df[(df['ISO_Week'] == current_week) & (df['ISO_Year'] == current_year)]

            if not current_week_data.empty and len(current_week_data) > 1:
                trade_entry_date = current_week_data.index[0]  # First trading day of current week
                previous_date = df.index[df.index.get_loc(trade_entry_date) - 1]
                df.loc[trade_entry_date, 'TradeEntry'] = df.loc[previous_date, 'Close']
                df.loc[trade_entry_date, 'Signal'] = signal
                in_trade = True
                trade_start_date = trade_entry_date

                # Determine trade exit date (last trading day of the current week)
                trade_exit_date = current_week_data.index[-1]
                trade_end_date = trade_exit_date
                df.loc[trade_exit_date, 'TradeExit'] = df.loc[trade_exit_date, 'Close']
    
                # Set InTrade for the duration of the trade
                df.loc[trade_start_date:trade_end_date, 'InTrade'] = signal * mthly_scale[current_month - 1]
                df.loc[trade_end_date,'Signal'] = - signal


        # Reset in_trade status after trade ends
        if in_trade and date == trade_end_date:
            in_trade = False
            trade_start_date = None
            trade_end_date = None
            df.loc[trade_end_date,'Signal'] = - signal


    df.drop(['Daily_Return','Weekly_Return'], axis=1, errors='ignore', inplace=True)

    # Convert back to Polars DataFrame
    return pl.from_pandas(df.reset_index())