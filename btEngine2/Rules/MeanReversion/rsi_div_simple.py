import pandas as pd
import polars as pl
import numpy as np

from btEngine2.Indicators import compute_effective_atr_pd

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Computes the Average True Range (ATR) for the given DataFrame.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr


def find_divergence(
    price: pd.Series,
    rsi: pd.Series,
    lookback: int,
    overbought: float,
    oversold: float,
    divergence_type: str,
    swing_lookback: int = 5
) -> pd.Series:
    """
    Detects bullish or bearish divergences using swing highs and lows without lookahead bias.
    
    Parameters:
    - price: Price series.
    - rsi: RSI series.
    - lookback: Lookback period for divergence detection.
    - overbought: Overbought level.
    - oversold: Oversold level.
    - divergence_type: 'bullish' or 'bearish'.
    - swing_lookback: Number of periods to look back to identify swing highs/lows.
    
    Returns:
    - Signal series where 1 indicates divergence detected, 0 otherwise.
    """
    signal = pd.Series(0, index=price.index)
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(price)):
        # Identify swing lows
        if divergence_type == 'bullish':
            # Check if price at i is the lowest over the past swing_lookback periods
            if price.iloc[i] == price.iloc[i - swing_lookback:i + 1].min():
                current_price_low = price.iloc[i]
                current_rsi_low = rsi.iloc[i]
                # Look for previous swing low
                for j in range(i - lookback, i):
                    if price.iloc[j] == price.iloc[j - swing_lookback:j + 1].min():
                        prev_price_low = price.iloc[j]
                        prev_rsi_low = rsi.iloc[j]
                        # Check for divergence
                        if current_price_low < prev_price_low and current_rsi_low > prev_rsi_low and current_rsi_low < oversold:
                            signal.iloc[i] = 1
                            break
        # Identify swing highs
        elif divergence_type == 'bearish':
            # Check if price at i is the highest over the past swing_lookback periods
            if price.iloc[i] == price.iloc[i - swing_lookback:i + 1].max():
                current_price_high = price.iloc[i]
                current_rsi_high = rsi.iloc[i]
                # Look for previous swing high
                for j in range(i - lookback, i):
                    if price.iloc[j] == price.iloc[j - swing_lookback:j + 1].max():
                        prev_price_high = price.iloc[j]
                        prev_rsi_high = rsi.iloc[j]
                        # Check for divergence
                        if current_price_high > prev_price_high and current_rsi_high < prev_rsi_high and current_rsi_high > overbought:
                            signal.iloc[i] = 1
                            break
    return signal

def find_divergence_fractal(
    price: pd.Series,
    rsi: pd.Series,
    overbought: float,
    oversold: float,
    divergence_type: str,
    swing_lookback: int = 5
) -> pd.Series:
    """
    Detects bullish or bearish divergences using swing highs and lows without lookahead bias.

    Parameters:
    - price: Price series.
    - rsi: RSI series.
    - overbought: Overbought level.
    - oversold: Oversold level.
    - divergence_type: 'bullish' or 'bearish'.
    - swing_lookback: Number of periods to look back to identify swing highs/lows.

    Returns:
    - Signal series where 1 indicates divergence detected, 0 otherwise.
    """
    signal = pd.Series(0, index=price.index)

    # Identify swing highs and lows using past data only
    swing_highs = []
    swing_lows = []

    for i in range(swing_lookback, len(price)):
        # Swing High
        if price.iloc[i] == price.iloc[i - swing_lookback:i + 1].max():
            swing_highs.append(i)
        # Swing Low
        if price.iloc[i] == price.iloc[i - swing_lookback:i + 1].min():
            swing_lows.append(i)

    # For bullish divergence
    if divergence_type == 'bullish':
        for idx in range(1, len(swing_lows)):
            current_idx = swing_lows[idx]
            prev_idx = swing_lows[idx - 1]
            # Price makes lower lows
            if price.iloc[current_idx] < price.iloc[prev_idx]:
                # RSI makes higher lows
                if rsi.iloc[current_idx] > rsi.iloc[prev_idx]:
                    # RSI below oversold level
                    if rsi.iloc[current_idx] < oversold:
                        signal.iloc[current_idx] = 1
    # For bearish divergence
    elif divergence_type == 'bearish':
        for idx in range(1, len(swing_highs)):
            current_idx = swing_highs[idx]
            prev_idx = swing_highs[idx - 1]
            # Price makes higher highs
            if price.iloc[current_idx] > price.iloc[prev_idx]:
                # RSI makes lower highs
                if rsi.iloc[current_idx] < rsi.iloc[prev_idx]:
                    # RSI above overbought level
                    if rsi.iloc[current_idx] > overbought:
                        signal.iloc[current_idx] = 1

    return signal.fillna(0)

def rsidiv_long(df: pd.DataFrame,
                ma: int = 10,
                rsi_settings = (3, 40, 70, 20),
                ma_chglb: int = 1,
                max_bars: int = 10,
                entry_at: str = 'Open',
                swing_lb: int = 3,
                div_type: str = 'fractal',
                limit_order: bool = False,
                atr_period: int = 5,
                atr_multiplier: float = 1.0,
                lmt_epsilon: float = 0.1,
                quick_exit: bool = False) -> pd.DataFrame:
    """
    Implements the mean reversion long strategy with optional limit orders and quick exit.
    """
    rsi_period, lookback, overbought, oversold = rsi_settings

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    df['MA'] = df['Close'].rolling(window=ma).mean()
    df['MA_prev'] = df['MA'].shift(ma_chglb)
    df['MA_slope'] = df['MA'] - df['MA_prev']
    
    # Compute RSI
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    
    # Compute ATR
    df['ATR'] = compute_effective_atr_pd(df, atr_period)['ATR']
    
    # Compute previous High and Low
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)
    
    # Bullish Divergence Signal
    if div_type == 'fractal':
        df['Bullish_Divergence'] = find_divergence_fractal(
            price=df['Close'],
            rsi=df['RSI'],
            overbought=overbought,
            oversold=oversold,
            divergence_type='bullish',
            swing_lookback=swing_lb
        )
    else:
        df['Bullish_Divergence'] = find_divergence(
            price=df['Close'],
            rsi=df['RSI'],
            lookback=lookback,
            overbought=overbought,
            oversold=oversold,
            divergence_type='bullish',
            swing_lookback=swing_lb
        )
    
    # Entry Condition
    df['Entry_Signal'] = ((df['MA_slope'] > 0) & (df['Bullish_Divergence'] == 1)).astype(int)
    
    # Bearish Divergence Signal for Exit
    if div_type == 'fractal':
        df['Bearish_Divergence'] = find_divergence_fractal(
            price=df['Close'],
            rsi=df['RSI'],
            overbought=overbought,
            oversold=oversold,
            divergence_type='bearish',
            swing_lookback=swing_lb
        )
    else:
        df['Bearish_Divergence'] = find_divergence(
            price=df['Close'],
            rsi=df['RSI'],
            lookback=lookback,
            overbought=overbought,
            oversold=oversold,
            divergence_type='bearish',
            swing_lookback=swing_lb
        )
    
    # Exit Conditions
    df['Exit_Signal'] = df['Bearish_Divergence']
    
    # Initialize columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0
    
    in_trade = False
    entry_index = None

    i = 0
    while i < len(df):
        if not in_trade:
            if df['Entry_Signal'].iloc[i]:
                entry_idx = i + 1  # Next day
                if entry_idx < len(df):
                    if limit_order:
                        # Limit order at next day's open minus ATR multiple
                        limit_price = df['Open'].iloc[entry_idx] - atr_multiplier * df['ATR'].iloc[i]
                        # Check if Low <= limit_price
                        if df['Low'].iloc[entry_idx] <= limit_price - lmt_epsilon*df['ATR'].iloc[i]:
                            df['Signal'].iloc[entry_idx] = 1  # Entry signal
                            df['TradeEntry'].iloc[entry_idx] = limit_price
                            df['InTrade'].iloc[entry_idx] = 1
                            in_trade = True
                            entry_index = entry_idx
                            i = entry_idx  # Move to the entry index
                        else:
                            # Order not filled
                            i += 1
                    else:
                        # Market order at next day's 'entry_at' price
                        df['Signal'].iloc[entry_idx] = 1  # Entry signal
                        df['TradeEntry'].iloc[entry_idx] = df[entry_at].iloc[entry_idx]
                        df['InTrade'].iloc[entry_idx] = 1
                        in_trade = True
                        entry_index = entry_idx
                        i = entry_idx  # Move to the entry index
                else:
                    # Cannot enter trade because it's the last day
                    break
            else:
                i += 1
        else:
            df['InTrade'].iloc[i] = 1
            bars_since_entry = i - entry_index

            exit_signal_prev = df['Exit_Signal'].iloc[i - 1] if i > 0 else 0

            exit_condition = exit_signal_prev or bars_since_entry >= max_bars

            # Quick exit condition
            if quick_exit and i > 0:
                if df['Close'].iloc[i] > df['High'].iloc[i - 1]:
                    exit_condition = True

            if exit_condition:
                exit_idx = i + 1  # Exit at next day's open
                if exit_idx < len(df):
                    df['Signal'].iloc[exit_idx] = -1  # Exit signal
                    df['TradeExit'].iloc[exit_idx] = df[entry_at].iloc[exit_idx]
                    df['InTrade'].iloc[exit_idx] = 1
                    in_trade = False
                    entry_index = None
                    i = exit_idx  # Move to the exit index
                else:
                    # Exit at the last available price
                    df['Signal'].iloc[i] = -1
                    df['TradeExit'].iloc[i] = df['Close'].iloc[i]
                    df['InTrade'].iloc[i] = 1
                    in_trade = False
                    entry_index = None
                    break
            else:
                i += 1

    df = pl.DataFrame(df)
    return df

def rsidiv_short(df: pd.DataFrame,
                 ma: int = 40,
                 rsi_settings = (3, 40, 70, 20),
                 ma_chglb: int = 1,
                 max_bars: int = 10,
                 entry_at: str = 'Open',
                 swing_lb: int = 3,
                 div_type: str = '',
                 limit_order: bool = False,
                 atr_period: int = 5,
                 atr_multiplier: float = 1.0,
                 lmt_epsilon: float = 0.1,
                 quick_exit: bool = False) -> pd.DataFrame:
    """
    Implements the mean reversion short strategy with optional limit orders and quick exit.
    """
    rsi_period, lookback, overbought, oversold = rsi_settings

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    df['MA'] = df['Close'].rolling(window=ma).mean()
    df['MA_prev'] = df['MA'].shift(ma_chglb)
    df['MA_slope'] = df['MA'] - df['MA_prev']
    
    # Compute RSI
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    
    # Compute ATR
    df['ATR'] = compute_effective_atr_pd(df, atr_period)['ATR']
    
    # Compute previous High and Low
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)
    
    # Bearish Divergence Signal
    if div_type == 'fractal':
        df['Bearish_Divergence'] = find_divergence_fractal(
            price=df['Close'],
            rsi=df['RSI'],
            overbought=overbought,
            oversold=oversold,
            divergence_type='bearish',
            swing_lookback=swing_lb
        )
    else:
        df['Bearish_Divergence'] = find_divergence(
            price=df['Close'],
            rsi=df['RSI'],
            lookback=lookback,
            overbought=overbought,
            oversold=oversold,
            divergence_type='bearish',
            swing_lookback=swing_lb
        )
    
    # Entry Condition
    df['Entry_Signal'] = ((df['MA_slope'] < 0) & (df['Bearish_Divergence'] == 1)).astype(int)
    
    # Bullish Divergence Signal for Exit
    if div_type == 'fractal':
        df['Bullish_Divergence'] = find_divergence_fractal(
            price=df['Close'],
            rsi=df['RSI'],
            overbought=overbought,
            oversold=oversold,
            divergence_type='bullish',
            swing_lookback=swing_lb
        )
    else:
        df['Bullish_Divergence'] = find_divergence(
            price=df['Close'],
            rsi=df['RSI'],
            lookback=lookback,
            overbought=overbought,
            oversold=oversold,
            divergence_type='bullish',
            swing_lookback=swing_lb
        )
    
    # Exit Conditions
    df['Exit_Signal'] = df['Bullish_Divergence']
    
    # Initialize columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0

    in_trade = False
    entry_index = None
    
    i = 0
    while i < len(df):
        if not in_trade:
            if df['Entry_Signal'].iloc[i]:
                entry_idx = i + 1  # Next day
                if entry_idx < len(df):
                    if limit_order:
                        # Limit order at next day's open plus ATR multiple
                        limit_price = df['Open'].iloc[entry_idx] + atr_multiplier * df['ATR'].iloc[entry_idx]
                        # Check if High >= limit_price
                        if df['High'].iloc[entry_idx] >= limit_price + lmt_epsilon*df['ATR'].iloc[i]:
                            df['Signal'].iloc[entry_idx] = -1  # Entry signal
                            df['TradeEntry'].iloc[entry_idx] = limit_price
                            df['InTrade'].iloc[entry_idx] = -1
                            in_trade = True
                            entry_index = entry_idx
                            i = entry_idx  # Move to the entry index
                        else:
                            # Order not filled
                            i += 1
                    else:
                        # Market order at next day's 'entry_at' price
                        df['Signal'].iloc[entry_idx] = -1  # Entry signal
                        df['TradeEntry'].iloc[entry_idx] = df[entry_at].iloc[entry_idx]
                        df['InTrade'].iloc[entry_idx] = -1
                        in_trade = True
                        entry_index = entry_idx
                        i = entry_idx  # Move to the entry index
                else:
                    # Cannot enter trade because it's the last day
                    break
            else:
                i += 1
        else:
            df['InTrade'].iloc[i] = -1
            bars_since_entry = i - entry_index

            exit_signal_prev = df['Exit_Signal'].iloc[i - 1] if i > 0 else 0

            exit_condition = exit_signal_prev or bars_since_entry >= max_bars

            # Quick exit condition
            if quick_exit and i > 0:
                if df['Close'].iloc[i] < df['Low'].iloc[i - 1]:
                    exit_condition = True

            if exit_condition:
                exit_idx = i + 1  # Exit at next day's open
                if exit_idx < len(df):
                    df['Signal'].iloc[exit_idx] = 1  # Exit signal
                    df['TradeExit'].iloc[exit_idx] = df[entry_at].iloc[exit_idx]
                    df['InTrade'].iloc[exit_idx] = -1
                    in_trade = False
                    entry_index = None
                    i = exit_idx  # Move to the exit index
                else:
                    # Exit at the last available price
                    df['Signal'].iloc[i] = 1
                    df['TradeExit'].iloc[i] = df['Close'].iloc[i]
                    df['InTrade'].iloc[i] = -1
                    in_trade = False
                    entry_index = None
                    break
            else:
                i += 1

    df = pl.DataFrame(df)
    return df