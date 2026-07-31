import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, spread: float = 0.0001):
        """
        data: DataFrame с колонками ['open', 'high', 'low', 'close']
        spread: фиксированный спред в долях цены
        """
        self.data = data
        self.spread = spread
        self.results = []

    def run(self, signals: pd.Series, sl_points: float, timeout_bars: int):
        """
        signals: Series с 1 (Buy), -1 (Sell), 0 (Hold)
        sl_points: размер стоп-лосса в абсолютных единицах цены
        timeout_bars: через сколько баров закрывать сделку принудительно
        """
        position = 0  # 1, -1, 0
        entry_price = 0.0
        entry_index = -1
        
        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            signal = signals.iloc[i]
            
            # Если есть открытая позиция
            if position != 0:
                # Проверка на закрытие по таймауту
                if (i - entry_index) >= timeout_bars:
                    self._close_position(i, current_bar['close'], position, entry_price)
                    position = 0
                    continue
                
                # Проверка на StopLoss
                if position == 1: # Buy
                    if current_bar['low'] <= (entry_price - sl_points):
                        self._close_position(i, entry_price - sl_points, position, entry_price)
                        position = 0
                        continue
                elif position == -1: # Sell
                    if current_bar['high'] >= (entry_price + sl_points):
                        self._close_position(i, entry_price + sl_points, position, entry_price)
                        position = 0
                        continue
            
            # Открытие новой позиции (если нет текущей)
            if position == 0 and signal != 0:
                position = signal
                # Учитываем спред при входе
                if position == 1:
                    entry_price = current_bar['close'] * (1 + self.spread)
                else:
                    entry_price = current_bar['close'] * (1 - self.spread)
                entry_index = i
                
        return pd.DataFrame(self.results)

    def _close_position(self, index, exit_price, side, entry_price):
        # Учитываем спред при выходе
        final_exit = exit_price * (1 - self.spread) if side == 1 else exit_price * (1 + self.spread)
        pnl = (final_exit - entry_price) * side if side == 1 else (entry_price - final_exit) * side
        self.results.append({
            'exit_index': index,
            'pnl': pnl
        })

# Тест на случайных сигналах
if __name__ == '__main__':
    # Генерируем фиктивные данные
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100
    }, index=dates)
    
    # Случайные сигналы
    signals = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates)
    
    engine = BacktestEngine(df, spread=0.0002)
    trades = engine.run(signals, sl_points=0.5, timeout_bars=5)
    
    print(f"Backtest completed. Total trades: {len(trades)}")
    print(trades.head())
