import MetaTrader5 as mt5
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from strategy_logic import QuantumStrategy

# Константы
SYMBOL = "EURUSD"
SPREAD = 0.0001  # Фиксированный спред для контроля проскальзывания
SLIPPAGE_LIMIT = 0.5 * SPREAD

def load_best_params():
    with open('best_params.json', 'r') as f:
        return json.load(f)

def get_market_data(n=1000):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']
    return df[['open', 'high', 'low', 'close', 'volume']]

def execute_trade(signal, expected_price, equity_history):
    tick = mt5.symbol_info_tick(SYMBOL)
    
    if signal == 1: # Buy
        market_price = tick.ask
        if abs(market_price - expected_price) > SLIPPAGE_LIMIT:
            print(f"Slippage Control: Buy отклонен. Разница {abs(market_price - expected_price):.6f}")
            return
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY,
            "price": market_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Quantum Bot Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)
        print("Buy executed")
        # Упрощенный учет PnL для графика (в реале нужно отслеживать закрытие)
        equity_history.append(equity_history[-1] - (market_price * 0.0001)) # Примерный учет комиссии/спреда

    elif signal == -1: # Sell
        market_price = tick.bid
        if abs(market_price - expected_price) > SLIPPAGE_LIMIT:
            print(f"Slippage Control: Sell отклонен. Разница {abs(market_price - expected_price):.6f}")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_SELL,
            "price": market_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Quantum Bot Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)
        print("Sell executed")
        equity_history.append(equity_history[-1] - (market_price * 0.0001))

def plot_equity(equity_history):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_history)
    plt.title("Live Equity Curve")
    plt.xlabel("Trades")
    plt.ylabel("Balance")
    plt.savefig("live_equity.png")
    plt.close()

def main():
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return

    params = load_best_params()
    strategy = QuantumStrategy()
    equity_history = [10000.0] # Начальный баланс
    
    print("Загрузка данных и обучение модели...")
    df = get_market_data(n=2000)
    if df is None:
        print("Не удалось получить данные для обучения")
        return
        
    targets = strategy.prepare_targets(df, params.get('sigma', 3.0))
    features = strategy.add_features(df)
    strategy.train_student(features, targets)
    print("Модель обучена. Бот запущен.")
    
    while True:
        df = get_market_data(n=100)
        if df is None:
            time.sleep(60)
            continue
            
        features = strategy.add_features(df)
        if features.empty:
            time.sleep(60)
            continue
            
        last_features = features.iloc[[-1]]
        signal = strategy.predict_signals(last_features, params['threshold']).iloc[0]
        
        if signal != 0:
            current_close = df['close'].iloc[-1]
            expected_price = current_close * (1 + SPREAD) if signal == 1 else current_close * (1 - SPREAD)
            
            execute_trade(signal, expected_price, equity_history)
            plot_equity(equity_history)
            
        time.sleep(60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        mt5.shutdown()
        print("Бот остановлен.")
