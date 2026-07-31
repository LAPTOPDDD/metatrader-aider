import optuna
import pandas as pd
import numpy as np
from fast_backtest import BacktestEngine
from strategy_logic import QuantumStrategy

# Предположим, что данные загружаются глобально или передаются
# Для примера создадим заглушку данных
def get_data():
    dates = pd.date_range(start='2020-01-01', periods=2000, freq='H')
    return pd.DataFrame({
        'open': np.random.randn(2000).cumsum() + 100,
        'high': np.random.randn(2000).cumsum() + 101,
        'low': np.random.randn(2000).cumsum() + 99,
        'close': np.random.randn(2000).cumsum() + 100
    }, index=dates)

data = get_data()
train_size = int(len(data) * 0.8)
train_df = data.iloc[:train_size]
test_df = data.iloc[train_size:]

def objective(trial):
    # Гиперпараметры для оптимизации
    sigma = trial.suggest_float('sigma', 1.0, 10.0)
    hold_bars = trial.suggest_int('hold_bars', 1, 20)
    sl_points = trial.suggest_float('sl_points', 0.01, 0.5)
    threshold = trial.suggest_float('threshold', 0.51, 0.7)
    
    # Инициализация стратегии
    strategy = QuantumStrategy()
    
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
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best params:", study.best_params)
