import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fast_backtest import BacktestEngine
from strategy_logic import QuantumStrategy

def get_data():
    if not mt5.initialize(): return None
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 10000)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def run_test(model_type):
    data = get_data()
    train_size = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:train_size], data.iloc[train_size:]
    
    strategy = QuantumStrategy(model_type=model_type)
    targets = strategy.prepare_targets(train_df, 3.0)
    features = strategy.add_features(train_df)
    strategy.train_student(features, targets)
    
    test_features = strategy.add_features(test_df)
    signals = strategy.predict_signals(test_features, 0.55)
    
    engine = BacktestEngine(test_df.loc[test_features.index])
    trades = engine.run(signals, sl_points=0.001, timeout_bars=10)
    return trades['pnl'].cumsum()

if __name__ == '__main__':
    print("Запуск теста CatBoost...")
    equity_catboost = run_test('lightgbm')
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_catboost, label='CatBoost')
    plt.title('Equity Curve: CatBoost')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison.png')
    print("График сохранен как model_comparison.png")
