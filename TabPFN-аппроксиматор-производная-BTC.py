import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from strategy_logic import QuantumStrategy

# Фикс для Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_data(count=5000):
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos("BITCOIN", mt5.TIMEFRAME_M15, 0, count)
    mt5.shutdown()
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def main():
    print("--- ВИЗУАЛИЗАЦИЯ: УЧИТЕЛЬ (ЦЕЛЬ) vs УЧЕНИК (АППРОКСИМАТОР) ---")
    
    # 1. Данные и инициализация
    sigma = 2.0
    threshold = 0.58
    data = get_data(3000)
    if data is None:
        print("Ошибка загрузки данных")
        return

    split = int(len(data) * 0.9)
    train_df, test_df = data.iloc[:split], data.iloc[split:]

    strategy = QuantumStrategy()
    
    # 2. Обучение модели (TabPFN или другой классификатор)
    print("Обучение TabPFN...")
    targets = strategy.prepare_targets(train_df, sigma)
    train_feats = strategy.add_features(train_df)
    strategy.train_student(train_feats, targets)

    # 3. Аппроксимация на тесте
    test_feats = strategy.add_features(test_df)
    result = strategy.predict_signals(test_feats, threshold)
    
    # Извлекаем аппроксимацию (второй элемент)
    if isinstance(result, (tuple, list)):
        approx_signal = result[1] if len(result) > 1 else result[0]
    else:
        approx_signal = result
    
    if hasattr(approx_signal, 'values'):
        approx_vals = approx_signal.values
    else:
        approx_vals = np.array(approx_signal)

    # 4. Расчёт Истинного Учителя (производная Гаусса) на тесте
    true_deriv = gaussian_filter1d(test_feats['close'].values, sigma=sigma, order=1)
    true_norm = true_deriv / (np.max(np.abs(true_deriv)) + 1e-9)

    # 5. ПОСТРОЕНИЕ ГРАФИКА — ТОЛЬКО ТЕСТОВЫЙ УЧАСТОК, ПОЛНОСТЬЮ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 2]})

    # Верхний график: цена на тестовом участке
    ax1.plot(test_feats.index, test_feats['close'], color='black', alpha=0.6, label='Price')
    ax1.set_title('EURUSD M15 – Тестовый участок (модель не видела эти данные)')
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    # Нижний график: цель vs аппроксимация на ВСЁМ тестовом участке
    ax2.plot(test_feats.index, true_norm, color='#3498db', lw=2, alpha=0.7, label='ЦЕЛЬ (Идеальный Гаусс)')
    ax2.plot(test_feats.index, approx_vals, color='#e74c3c', ls='--', lw=1.5, label='АППРОКСИМАТОР (TabPFN)')

    # Зоны сигналов (зелёная – выше нуля, красная – ниже нуля)
    ax2.fill_between(test_feats.index, 0, approx_vals,
                     where=(approx_vals > 0), color='green', alpha=0.1)
    ax2.fill_between(test_feats.index, 0, approx_vals,
                     where=(approx_vals < 0), color='red', alpha=0.1)

    ax2.axhline(0, color='black', lw=1)
    ax2.set_title('Сравнение на тестовом участке (полный масштаб)')
    ax2.set_ylabel('Trend Momentum [-1, 1]')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Gaussian_vs_TabPFN_full_test.png', dpi=150)
    print("График сравнения сохранен: Gaussian_vs_TabPFN_full_test.png")
    plt.show()

if __name__ == '__main__':
    main()