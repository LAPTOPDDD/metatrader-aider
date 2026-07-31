import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMClassifier

class QuantumStrategy:
    def __init__(self):
        self.model = LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
        self.features = ['macd', 'rsi', 'bb_pos', 'prev_ret']

    def prepare_targets(self, data: pd.DataFrame, sigma: float):
        """
        Расчет таргета через производную Гаусса.
        MATH_OFFSET отсекает края, где фильтр Гаусса наиболее подвержен искажениям.
        """
        math_offset = int(4 * sigma)
        
        # Первая производная через Гаусс
        derivative = gaussian_filter1d(data['close'], sigma=sigma, order=1)
        
        # Таргет: 1 если производная > 0, иначе 0
        targets = (derivative > 0).astype(int)
        
        # Применяем отступ: последние бары (и первые) не используем для обучения
        # чтобы избежать "краевых галлюцинаций"
        targets[:math_offset] = np.nan
        targets[-math_offset:] = np.nan
        
        return targets

    def add_features(self, data: pd.DataFrame):
        """
        Добавление казуальных признаков (лаг 1).
        """
        df = data.copy()
        
        # MACD (12, 26)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26).shift(1)
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.clip(lower=0)).ewm(span=14).mean()
        loss = (-delta.clip(upper=0)).ewm(span=14).mean()
        rs = gain / loss
        df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
        
        # BB Position
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_pos'] = ((df['close'] - sma20) / (2 * std20)).shift(1)
        
        # Prev Return
        df['prev_ret'] = df['close'].pct_change().shift(1)
        
        return df.dropna()

    def train_student(self, train_data: pd.DataFrame, targets: pd.Series):
        """
        Обучение Ученика (LightGBM) предсказывать таргет Учителя.
        """
        # Выравниваем данные по индексам
        common_idx = train_data.index.intersection(targets.index)
        X = train_data.loc[common_idx]
        y = targets.loc[common_idx]
        
        # Удаляем NaN, которые могли возникнуть из-за MATH_OFFSET или лагов
        mask = ~y.isna()
        self.model.fit(X[mask], y[mask])

    def predict_signals(self, test_data: pd.DataFrame, threshold: float):
        """
        Генерация сигналов на основе вероятностей модели.
        """
        probs = self.model.predict_proba(test_data)[:, 1]
        signals = np.zeros_like(probs)
        
        signals[probs > threshold] = 1
        signals[probs < (1 - threshold)] = -1
        
        return pd.Series(signals, index=test_data.index)
