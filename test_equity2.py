import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
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

def main():
    # 1. Загрузка параметров (используем те же, что и для основной модели)
    try:
        with open('best_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print("Файл best_params.json не найден. Используем параметры по умолчанию.")
        params = {'sigma': 3.0, 'threshold': 0.55, 'sl_points': 0.001, 'hold_bars': 10}

    # 2. Загрузка данных
    data = get_data()
    if data is None:
        print("Данные не получены. Проверьте подключение к MT5.")
        return
    
    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]
    
    # 3. Инициализация и обучение стратегии с TabPFN
    print("Обучение TabPFN на тренировочных данных...")
    strategy = QuantumStrategy(model_type='tabpfn')
    targets = strategy.prepare_targets(train_df, params.get('sigma', 3.0))
    features = strategy.add_features(train_df)
    strategy.train_student(features, targets)
    
    # 4. Прогноз на тесте
    print("Генерация сигналов на тестовых данных (TabPFN)...")
    test_features = strategy.add_features(test_df)
    signals = strategy.predict_signals(test_features, params.get('threshold', 0.55))
    
    # 5. Бэктест
    print("Запуск бэктеста...")
    engine = BacktestEngine(test_df.loc[test_features.index])
    trades = engine.run(signals, sl_points=params.get('sl_points', 0.001), timeout_bars=params.get('hold_bars', 10))
    
    # 6. Построение графика
    if not trades.empty:
        equity = trades['pnl'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(equity)
        plt.title('Equity Curve (Out-of-Sample) - TabPFN')
        plt.xlabel('Trade Index')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.savefig('equity_curve_tabpfn.png')
        
        # Расчет метрик
        pnl = trades['pnl']
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() != 0 else 0
        
        print(f"Бэктест завершен.")
        print(f"Количество сделок: {len(trades)}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print("График сохранен как equity_curve_tabpfn.png")
    else:
        print("Сделок не было, график не построен.")

if __name__ == '__main__':
    main()
