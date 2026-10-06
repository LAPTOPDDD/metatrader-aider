import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_data(count=5000):
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, count)
    mt5.shutdown()
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def add_features(df):
    df = df.copy()
    macd = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['macd_lag1'] = macd.shift(1)
    df['macd_lag2'] = macd.shift(2)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_lag1'] = rsi.shift(1)
    df['ret1'] = df['close'].pct_change(1)
    df['ret3'] = df['close'].pct_change(3)
    df['ret5'] = df['close'].pct_change(5)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['atr_rel'] = atr14 / df['close']
    features = ['macd_lag1', 'macd_lag2', 'rsi_lag1', 'ret1', 'ret3', 'ret5', 'bb_pos', 'bb_width', 'atr_rel']
    return df, features

def prepare_target(df, sigma=2.0):
    conv = gaussian_filter1d(df['close'].values, sigma=sigma, mode='nearest')
    deriv = np.gradient(conv)
    max_abs = np.max(np.abs(deriv))
    if max_abs > 0:
        deriv = deriv / max_abs
    return deriv

def main():
    print("--- LIGHTGBM РЕГРЕССИЯ: аппроксимация первой производной Гаусса ---")
    sigma = 2.0
    data = get_data(3000)
    if data is None:
        print("Ошибка загрузки данных")
        return

    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]

    train_feats, feature_cols = add_features(train_df)
    train_target = prepare_target(train_df, sigma)
    train_data = train_feats[feature_cols].copy()
    train_data['target'] = train_target
    train_data = train_data.dropna()
    X_train = train_data[feature_cols]
    y_train = train_data['target']

    test_feats, _ = add_features(test_df)
    test_data = test_feats[feature_cols].copy()
    test_data = test_data.dropna()
    X_test = test_data[feature_cols]

    true_target_test = prepare_target(test_df, sigma)
    n_na_test = len(test_feats) - len(X_test)
    true_target_test_aligned = true_target_test[n_na_test:]

    print(f"Обучающая выборка: {len(X_train)} строк, тестовая: {len(X_test)} строк")

    # === LightGBM регрессор (не требует torch) ===
    regressor = LGBMRegressor(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        verbose=-1
    )
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Отображение последних 500 баров для деталей
    display_bars = 500
    if len(X_test) > display_bars:
        y_pred_display = y_pred[-display_bars:]
        true_target_display = true_target_test_aligned[-display_bars:]
        test_feats_display = test_feats.iloc[-display_bars:]
    else:
        y_pred_display = y_pred
        true_target_display = true_target_test_aligned
        test_feats_display = test_feats

    # График
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(test_feats_display.index, test_feats_display['close'], color='black', alpha=0.6, label='Price')
    ax1.set_title(f'EURUSD M15 – Тестовый участок (последние {display_bars} баров)')
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    ax2.plot(test_feats_display.index, true_target_display, color='#3498db', lw=2, alpha=0.7, label='ЦЕЛЬ (Гаусс)')
    ax2.plot(test_feats_display.index, y_pred_display, color='#e74c3c', ls='--', lw=1.5, label='АППРОКСИМАТОР (LightGBM)')

    ax2.fill_between(test_feats_display.index, 0, y_pred_display,
                     where=(y_pred_display > 0), color='green', alpha=0.1)
    ax2.fill_between(test_feats_display.index, 0, y_pred_display,
                     where=(y_pred_display < 0), color='red', alpha=0.1)

    ax2.axhline(0, color='black', lw=1)
    ax2.set_title('Регрессия LightGBM: цель vs предсказание (непрерывная кривая)')
    ax2.set_ylabel('Момент тренда')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Gaussian_vs_LGBM.png', dpi=150)
    print("График сохранён: Gaussian_vs_LGBM.png")
    plt.show()

if __name__ == '__main__':
    main()