import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from fast_backtest import BacktestEngine
from strategy_logic import QuantumStrategy

# Фикс DLL на уровне главного скрипта
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_data():
    """Загрузка данных из MT5"""
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return None
    
    # Берем 10 000 баров EURUSD M15
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 10000)
    mt5.shutdown()
    
    if rates is None:
        print("Не удалось получить данные из MT5")
        return None
        
    df = pd.DataFrame(rates)
    # ПРЕОБРАЗОВАНИЕ ВРЕМЕНИ В ИНДЕКС
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def main():
    print("--- ЗАПУСК СИСТЕМЫ TABPFN ---")
    
    # 1. Загрузка параметров
    try:
        with open('best_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print("Используем дефолтные параметры.")
        params = {'sigma': 2.5, 'threshold': 0.60, 'sl_points': 0.002, 'hold_bars': 5}

    # 2. Данные
    data = get_data()
    if data is None: return
    
    # Разбивка 80/20
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split]
    test_df = data.iloc[split:]
    
    # 3. Инициализация и Обучение
    # Оставляем только TabPFN
    strategy = QuantumStrategy(model_type='tabpfn')
    
    print("Шаг: Разметка Гаусса (Учитель)...")
    targets = strategy.prepare_targets(train_df, params['sigma'])
    
    print("Шаг: Расчет признаков...")
    features_df = strategy.add_features(train_df)
    
    print("Шаг: Обучение Трансформера (Ученик)...")
    strategy.train_student(features_df, targets)
    
    # 4. Прогноз
    print("Шаг: Генерация сигналов на тесте...")
    test_features = strategy.add_features(test_df)
    signals = strategy.predict_signals(test_features, params['threshold'])
    
    # 5. Бэктест
    print("Шаг: Запуск бэктеста...")
    engine = BacktestEngine(test_df.loc[test_features.index])
    trades = engine.run(signals, sl_points=params['sl_points'], timeout_bars=params['hold_bars'])
    
    # 6. Результаты
    if not trades.empty:
        equity = trades['pnl'].cumsum()
        plt.figure(figsize=(12, 6))
        plt.plot(equity, color='royalblue', lw=2)
        plt.title(f"Equity Curve - TabPFN Student (OOS)")
        plt.xlabel('Trades')
        plt.ylabel('PnL')
        plt.grid(True, alpha=0.3)
        plt.savefig('equity_curve_tabpfn_final.png')
        
        pnl = trades['pnl']
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() != 0 else 0
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"Сделок: {len(trades)}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print("График: equity_curve_tabpfn_final.png")
        plt.show()
    else:
        print("Сделок не было.")

if __name__ == '__main__':
    main()