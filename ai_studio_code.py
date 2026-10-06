import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

class FastBacktester:
    def __init__(self, df, spread=0.00015):
        self.df = df
        self.spread = spread
        self.equity = [1.0]
        self.positions = [] # {type, entry_price, sl, hold_timer}
        
    def run(self, signals, hold_bars, sl_pips):
        # signals - массив вероятностей от LightGBM
        # Имитируем MT5: идем бар за баром
        balance = 1.0
        active_pos = None # Только 1 позиция за раз
        
        for i in range(len(self.df) - 1):
            price = self.df.close.iloc[i]
            
            # 1. Проверка стоп-лосса и тайм-аута у активной позиции
            if active_pos:
                active_pos['hold'] += 1
                low, high = self.df.low.iloc[i], self.df.high.iloc[i]
                
                # SL check
                hit_sl = False
                if active_pos['type'] == 1 and low <= active_pos['sl']: hit_sl = True
                if active_pos['type'] == -1 and high >= active_pos['sl']: hit_sl = True
                
                if hit_sl or active_pos['hold'] >= hold_bars:
                    # Закрываем
                    exit_price = active_pos['sl'] if hit_sl else price
                    pnl = (exit_price - active_pos['entry']) * active_pos['type']
                    balance += pnl - self.spread
                    active_pos = None
            
            # 2. Открытие новой позиции (Execution Lag: сигнал i -> вход на i+1)
            if not active_pos:
                sig = signals[i] # Сигнал от модели
                if sig != 0:
                    active_pos = {
                        'type': sig,
                        'entry': self.df.close.iloc[i+1], # Вход на открытии следующего
                        'sl': self.df.close.iloc[i+1] - sig * sl_pips * 0.0001,
                        'hold': 0
                    }
            
            self.equity.append(balance)
        return np.array(self.equity)

# Логика обучения для Агента
def train_intelligent_agent(df, sigma=2.0):
    # Учитель: Гаусс с математическим отступом
    # Ученик: LightGBM
    # ... эту часть Aider допишет сам на основе нашего ТЗ ...
    pass