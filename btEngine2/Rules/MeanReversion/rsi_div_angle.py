import pandas as pd
import numpy as np
import polars as pl
from btEngine2.Indicators import *

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_slope(series: pd.Series, period: int) -> pd.Series:
    """
    Computes the slope of a rolling window over the series.
    """
    slopes = [np.nan] * (period - 1)
    for i in range(period - 1, len(series)):
        y = series.iloc[i - period + 1:i + 1]
        x = np.arange(period)
        # Perform linear regression to get the slope
        slope, intercept = np.polyfit(x, y, 1)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes the Average Directional Index (ADX).
    """
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = adx

    return df

def generate_angle_signal(
    df: pd.DataFrame,
    rsi_period: int,
    slope_period: int,
    rsi_overbought: float = 60,
    rsi_oversold: float = 40,
    smooth: tuple = ('ema', 5),
    lag_signal: int = 0,
    min_angle: float = 0.0,
) -> pd.DataFrame:
    """
    Computes the angle between the price slope and RSI slope and generates signals based on divergence.

    Parameters:
    - df: DataFrame with at least a 'Close' column.
    - rsi_period: The period over which to calculate the RSI.
    - slope_period: The number of periods over which to calculate the slopes.
    - rsi_overbought: RSI threshold for overbought condition.
    - rsi_oversold: RSI threshold for oversold condition.
    - smooth: Tuple indicating smoothing method ('ema' or 'sma') and period.
    - trend_filter: Tuple indicating trend filter method ('ema' or 'sma') and period.
    - lag_signal: Number of periods to lag the signal.

    Returns:
    - DataFrame with additional columns including 'Angle', 'Bullish Signal', 'Bearish Signal'.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df = df.copy()

    # Compute RSI
    df['RSI'] = compute_rsi(df['Close'], rsi_period)

    if smooth[0] == 'ema':
        df['RSI'] = df['RSI'].ewm(span=smooth[1]).mean()
    elif smooth[0] == 'sma':
        df['RSI'] = df['RSI'].rolling(window=smooth[1]).mean()
    
    # Compute Price Slope
    df['Price_Slope'] = compute_slope(df['Close'], slope_period)

    # Compute RSI Slope
    df['RSI_Slope'] = compute_slope(df['RSI'], slope_period)

    # Compute Angle between Price Slope and RSI Slope
    # Handle division by zero and invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = df['RSI_Slope'] - df['Price_Slope']
        denominator = 1 + (df['Price_Slope'] * df['RSI_Slope'])
        angle_rad = np.arctan(np.abs(numerator / denominator))
        angle_deg = np.degrees(angle_rad)
        df['Angle'] = angle_deg
    
    # Conditions for Bearish and Bullish Divergence
    df['Bearish Div'] = (df['Price_Slope'] > 0) & (df['RSI_Slope'] < 0) & (df['RSI'] > rsi_overbought) & (df['Angle'] > min_angle)
    df['Bullish Div'] = (df['Price_Slope'] < 0) & (df['RSI_Slope'] > 0) & (df['RSI'] < rsi_oversold) & (df['Angle'] > min_angle)

    # Bearish Signal: Use Angle as Signal Strength
    df['Bearish Signal'] = np.where(df['Bearish Div'], -df['Angle'], 0)
    # Bullish Signal: Use Angle as Signal Strength
    df['Bullish Signal'] = np.where(df['Bullish Div'], df['Angle'], 0)

    # Lag the Signals if needed
    if lag_signal > 0:
        df['Bearish Signal'] = df['Bearish Signal'].shift(lag_signal)
        df['Bullish Signal'] = df['Bullish Signal'].shift(lag_signal)

    # Smooth the Signals
    if smooth[0] == 'ema':
        df['Bearish Signal'] = df['Bearish Signal'].ewm(span=smooth[1]).mean()
        df['Bullish Signal'] = df['Bullish Signal'].ewm(span=smooth[1]).mean()
    elif smooth[0] == 'sma':
        df['Bearish Signal'] = df['Bearish Signal'].rolling(window=smooth[1]).mean()
        df['Bullish Signal'] = df['Bullish Signal'].rolling(window=smooth[1]).mean()


    return df


def ang_div_long(
    df: pl.DataFrame,
    rsi_period: int = 14,
    slope_period: int = 5,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    angle_threshold: float = 20.0,
    smooth: tuple = ('ema', 5),
    lag_signal: int = 0,
    min_angle: float = 0.0,
    max_hold_days: int = 10,
    min_days_in_trade: int = 1,
    early_exit: bool = True,
    lmt_order: bool = False,
    atr_period: int = 5,
    atr_multiplier: float = 0.8,
    entry_at: str = 'Open'
) -> pl.DataFrame:
    """
    Implements the trading rule based on angle divergence for long positions.

    Parameters:
    - df: Polars DataFrame with market data.
    - rsi_period: Period for RSI calculation.
    - slope_period: Period for slope calculation.
    - angle_threshold: Threshold for the angle to generate a signal.
    - smooth: Tuple indicating smoothing method ('ema' or 'sma') and period.
    - lag_signal: Number of periods to lag the signal.
    - min_angle: Minimum angle to consider for divergence.
    - max_hold_days: Maximum number of days to hold a trade.
    - min_days_in_trade: Minimum days in trade before early exit can be applied.
    - early_exit: If True, applies the early exit rule.
    - lmt_order: If True, uses limit order logic.
    - atr_period: Period for ATR calculation (used in limit order logic).
    - atr_multiplier: Multiplier for ATR to set limit price.
    - entry_at: Column name to use for entry price ('Open' or 'Close').

    Returns:
    - Polars DataFrame with additional columns: 'TradeEntry', 'TradeExit', 'InTrade'.
    """

    # Convert Polars DataFrame to Pandas DataFrame for compatibility
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()

    # Ensure 'Date' is a column if it's an index
    if df_pd.index.name == 'Date':
        df_pd = df_pd.reset_index()
    

    # Generate angle signals
    df_signals = generate_angle_signal(
        df_pd,
        rsi_period=rsi_period,
        slope_period=slope_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        smooth=smooth,
        lag_signal=lag_signal,
        min_angle=min_angle
    )

    # Create Signal column based on angle_threshold
    df_signals['Signal'] = np.where(
        df_signals['Bullish Signal'] >= angle_threshold, 1, 0
    )

    # Initialize TradeEntry, TradeExit, InTrade columns
    df_signals['TradeEntry'] = np.nan
    df_signals['TradeExit'] = np.nan
    df_signals['InTrade'] = 0

    # Calculate ATR for limit order logic
    if lmt_order:
        df_signals = compute_effective_atr(df_signals, atr_period)

    in_trade = False
    entry_index = None
    days_in_trade = 0

    i = 0
    while i < len(df_signals):
        if not in_trade:
            if df_signals['Signal'].iloc[i] == 1:
                entry_idx = i + 1  # Enter at next period
                if entry_idx < len(df_signals):
                    if lmt_order:
                        # Limit price is next day's Open minus ATR * multiplier
                        limit_price = df_pd[entry_at].iloc[entry_idx] - atr_multiplier * df_signals['ATR'].iloc[entry_idx]
                        # Check if Low <= limit_price
                        if df_pd['Low'].iloc[entry_idx] <= limit_price:
                            df_signals['TradeEntry'].iloc[entry_idx] = limit_price
                            df_signals['InTrade'].iloc[entry_idx] = 1
                            in_trade = True
                            entry_index = entry_idx
                            days_in_trade = 1
                            i = entry_idx  # Move to entry index
                        else:
                            # Limit order not filled
                            i += 1
                    else:
                        # Market order at next period's entry_at price
                        df_signals['TradeEntry'].iloc[entry_idx] = df_pd[entry_at].iloc[entry_idx]
                        df_signals['InTrade'].iloc[entry_idx] = 1
                        in_trade = True
                        entry_index = entry_idx
                        days_in_trade = 1
                        i = entry_idx  # Move to entry index
                else:
                    # Cannot enter trade because it's the last day
                    break
            else:
                i += 1
        else:
            df_signals['InTrade'].iloc[i] = 1
            days_in_trade += 1

            exit_signal = False

            # Early exit rule
            if early_exit and days_in_trade > min_days_in_trade:
                if df_pd['Close'].iloc[i] > df_pd['High'].iloc[i - 1]:
                    exit_signal = True

            # Max hold days exit
            if days_in_trade >= max_hold_days:
                exit_signal = True

            if exit_signal:
                exit_idx = i + 1  # Exit at next period
                if exit_idx < len(df_signals):
                    df_signals['TradeExit'].iloc[exit_idx] = df_pd[entry_at].iloc[exit_idx]
                    df_signals['InTrade'].iloc[exit_idx] = 1
                    in_trade = False
                    entry_index = None
                    days_in_trade = 0
                    i = exit_idx  # Move to exit index
                else:
                    # Exit at last available price
                    df_signals['TradeExit'].iloc[i] = df_pd['Close'].iloc[i]
                    df_signals['InTrade'].iloc[i] = 0
                    in_trade = False
                    entry_index = None
                    days_in_trade = 0
                    break
            else:
                i += 1

    # Convert back to Polars DataFrame

    df_result = pl.from_pandas(df_signals)

    return df_result


def ang_div_short(
    df: pl.DataFrame,
    rsi_period: int = 14,
    slope_period: int = 5,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    angle_threshold: float = 20.0,
    smooth: tuple = ('ema', 5),
    lag_signal: int = 0,
    min_angle: float = 0.0,
    max_hold_days: int = 10,
    min_days_in_trade: int = 1,
    early_exit: bool = True,
    lmt_order: bool = False,
    atr_period: int = 5,
    atr_multiplier: float = 0.8,
    entry_at: str = 'Open'
) -> pl.DataFrame:
    """
    Implements the trading rule based on angle divergence for short positions.

    Parameters:
    - df: Polars DataFrame with market data.
    - rsi_period: Period for RSI calculation.
    - slope_period: Period for slope calculation.
    - angle_threshold: Threshold for the angle to generate a signal.
    - smooth: Tuple indicating smoothing method ('ema' or 'sma') and period.
    - lag_signal: Number of periods to lag the signal.
    - min_angle: Minimum angle to consider for divergence.
    - max_hold_days: Maximum number of days to hold a trade.
    - min_days_in_trade: Minimum days in trade before early exit can be applied.
    - early_exit: If True, applies the early exit rule.
    - lmt_order: If True, uses limit order logic.
    - atr_period: Period for ATR calculation (used in limit order logic).
    - atr_multiplier: Multiplier for ATR to set limit price.
    - entry_at: Column name to use for entry price ('Open' or 'Close').

    Returns:
    - Polars DataFrame with additional columns: 'TradeEntry', 'TradeExit', 'InTrade'.
    """

    # Convert Polars DataFrame to Pandas DataFrame for compatibility
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()

    # Ensure 'Date' is a column if it's an index
    if df_pd.index.name == 'Date':
        df_pd = df_pd.reset_index()

    # Generate angle signals
    df_signals = generate_angle_signal(
        df_pd,
        rsi_period=rsi_period,
        slope_period=slope_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        smooth=smooth,
        lag_signal=lag_signal,
        min_angle=min_angle
    )

    # Create Signal column based on angle_threshold for short trades (Bearish)
    df_signals['Signal'] = np.where(
        df_signals['Bearish Signal'] >= angle_threshold, -1, 0
    )

    # Initialize TradeEntry, TradeExit, InTrade columns
    df_signals['TradeEntry'] = np.nan
    df_signals['TradeExit'] = np.nan
    df_signals['InTrade'] = 0

    # Calculate ATR for limit order logic
    if lmt_order:
        df_signals = compute_effective_atr(df_signals, atr_period)

    in_trade = False
    entry_index = None
    days_in_trade = 0

    i = 0
    while i < len(df_signals):
        if not in_trade:
            if df_signals['Signal'].iloc[i] == -1:
                entry_idx = i + 1  # Enter at next period
                if entry_idx < len(df_signals):
                    if lmt_order:
                        # Limit price is next day's Open plus ATR * multiplier (for short)
                        limit_price = df_pd[entry_at].iloc[entry_idx] + atr_multiplier * df_signals['ATR'].iloc[entry_idx]
                        # Check if High >= limit_price
                        if df_pd['High'].iloc[entry_idx] >= limit_price:
                            df_signals['TradeEntry'].iloc[entry_idx] = limit_price
                            df_signals['InTrade'].iloc[entry_idx] = -1
                            in_trade = True
                            entry_index = entry_idx
                            days_in_trade = 1
                            i = entry_idx  # Move to entry index
                        else:
                            # Limit order not filled
                            i += 1
                    else:
                        # Market order at next period's entry_at price
                        df_signals['TradeEntry'].iloc[entry_idx] = df_pd[entry_at].iloc[entry_idx]
                        df_signals['InTrade'].iloc[entry_idx] = -1
                        in_trade = True
                        entry_index = entry_idx
                        days_in_trade = 1
                        i = entry_idx  # Move to entry index
                else:
                    # Cannot enter trade because it's the last day
                    break
            else:
                i += 1
        else:
            df_signals['InTrade'].iloc[i] = -1
            days_in_trade += 1

            exit_signal = False

            # Early exit rule for short trades
            if early_exit and days_in_trade > min_days_in_trade:
                if df_pd['Close'].iloc[i] < df_pd['Low'].iloc[i - 1]:
                    exit_signal = True

            # Max hold days exit
            if days_in_trade >= max_hold_days:
                exit_signal = True

            if exit_signal:
                exit_idx = i + 1  # Exit at next period
                if exit_idx < len(df_signals):
                    df_signals['TradeExit'].iloc[exit_idx] = df_pd[entry_at].iloc[exit_idx]
                    df_signals['InTrade'].iloc[exit_idx] = 0
                    in_trade = False
                    entry_index = None
                    days_in_trade = 0
                    i = exit_idx  # Move to exit index
                else:
                    # Exit at last available price
                    df_signals['TradeExit'].iloc[i] = df_pd['Close'].iloc[i]
                    df_signals['InTrade'].iloc[i] = 0
                    in_trade = False
                    entry_index = None
                    days_in_trade = 0
                    break
            else:
                i += 1

    # Convert back to Polars DataFrame
    df_result = pl.from_pandas(df_signals)

    return df_result