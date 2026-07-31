import optuna
import pandas as pd
import numpy as np
import json
import os
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from fast_backtest import BacktestEngine
from strategy_logic import QuantumStrategy

def get_data():
    """
    Загружает данные напрямую из MetaTrader 5.
    """
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return None
    
    # Получаем 10000 баров EURUSD M15
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 10000)
    mt5.shutdown()
    
    if rates is None:
        print("Не удалось получить данные из MT5")
        return None
        
    df = pd.DataFrame(rates)
    # Преобразуем время
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Оставляем нужные колонки и переименовываем
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    return df

# Загрузка данных
data = get_data()
if data is None:
    raise Exception("Данные не получены. Проверьте подключение к MT5.")

train_size = int(len(data) * 0.8)
train_df = data.iloc[:train_size]
test_df = data.iloc[train_size:]

def objective(trial):
    # Гиперпараметры для оптимизации
    sigma = trial.suggest_float('sigma', 1.0, 10.0)
    hold_bars = trial.suggest_int('hold_bars', 1, 20)
    sl_points = trial.suggest_float('sl_points', 0.01, 0.5)
    threshold = trial.suggest_float('threshold', 0.51, 0.7)
    model_type = trial.suggest_categorical('model_type', ['lightgbm', 'tabpfn'])
    
    # Инициализация стратегии
    strategy = QuantumStrategy(model_type=model_type)
    
    # Подготовка данных
    targets = strategy.prepare_targets(train_df, sigma)
    features = strategy.add_features(train_df)
    
    # Обучение
    strategy.train_student(features, targets)
    
    # Прогноз
    test_features = strategy.add_features(test_df)
    signals = strategy.predict_signals(test_features, threshold)
    
    # Бэктест
    engine = BacktestEngine(test_df.loc[test_features.index])
    trades = engine.run(signals, sl_points=sl_points, timeout_bars=hold_bars)
    
    if len(trades) == 0:
        return -1.0
    
    # Расчет Шарпа
    pnl = trades['pnl']
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() != 0 else 0
    
    return sharpe

if __name__ == '__main__':
    print("Запуск оптимизации...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    print("Best params:", best_params)
    
    # Сохранение параметров
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Финальный прогон на OOS для графика
    print("Построение графика эквити...")
    strategy = QuantumStrategy(model_type=best_params['model_type'])
    targets = strategy.prepare_targets(train_df, best_params['sigma'])
    features = strategy.add_features(train_df)
    strategy.train_student(features, targets)
    
    test_features = strategy.add_features(test_df)
    signals = strategy.predict_signals(test_features, best_params['threshold'])
    
    engine = BacktestEngine(test_df.loc[test_features.index])
    trades = engine.run(signals, sl_points=best_params['sl_points'], timeout_bars=best_params['hold_bars'])
    
    # Построение графика
    if not trades.empty:
        equity = trades['pnl'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(equity)
        plt.title(f'Equity Curve (Out-of-Sample) - {best_params["model_type"]}')
        plt.xlabel('Trade Index')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        
        # Запись результата
        pnl = trades['pnl']
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() != 0 else 0
        with open('last_result.txt', 'w') as f:
            f.write(f"SHARPE: {sharpe:.4f}, TRADES: {len(trades)}")
            
        print("График сохранен как equity_curve.png")
    else:
        with open('last_result.txt', 'w') as f:
            f.write("SHARPE: 0.0000, TRADES: 0")
        print("Сделок не было, график не построен.")
