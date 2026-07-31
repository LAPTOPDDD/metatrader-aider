import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

class QuantumStrategy:
    def __init__(self, model_type='lightgbm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        # Используем CatBoost как основную модель
        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            allow_writing_files=False
        )
            
        self.features = [
            'macd', 'rsi', 'bb_pos', 'prev_ret', 'atr', 'roc', 
            'volatility', 'hour', 'day_of_week', 'volume', 
            'vol_change', 'vol_regime', 'ret_3', 'ret_5'
        ]

    def prepare_targets(self, data: pd.DataFrame, sigma: float):
        """
        Расчет таргета через производную Гаусса с адаптивным порогом.
        """
        math_offset = int(4 * sigma)
        
        # Первая производная через Гаусс
        derivative = gaussian_filter1d(data['close'], sigma=sigma, order=1)
        
        # Адаптивный порог: отсекаем шум
        threshold = np.std(derivative) * 0.7
        
        # Таргет: 1 если производная > порога, 0 если < -порога, иначе NaN (нейтрально)
        targets = np.where(derivative > threshold, 1, np.where(derivative < -threshold, 0, np.nan))
        
        # Применяем отступ
        targets[:math_offset] = np.nan
        targets[-math_offset:] = np.nan
        
        return pd.Series(targets, index=data.index)

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
        df['ret_3'] = df['close'].pct_change(3).shift(1)
        df['ret_5'] = df['close'].pct_change(5).shift(1)
        
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean().shift(1)
        
        # ROC (Rate of Change) - импульс
        df['roc'] = df['close'].pct_change(periods=5).shift(1)
        
        # Volatility - режим рынка
        df['volatility'] = df['close'].pct_change().rolling(20).std().shift(1)
        
        # Временные признаки
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Volume (если есть в данных)
        if 'volume' in df.columns:
            df['volume'] = df['volume'].shift(1)
            df['vol_change'] = df['volume'].pct_change().shift(1)
        else:
            df['volume'] = 0
            df['vol_change'] = 0
            
        # Volatility Regime
        df['vol_regime'] = df['atr'] / df['atr'].rolling(50).mean()
        
        return df.dropna()

    def train_student(self, train_data: pd.DataFrame, targets: pd.Series):
        """
        Обучение Ученика.
        """
        # Выравниваем данные по индексам
        common_idx = train_data.index.intersection(targets.index)
        X = train_data.loc[common_idx]
        y = targets.loc[common_idx]
        
        # Удаляем NaN
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Масштабирование
        X_scaled = self.scaler.fit_transform(X_clean)
        
        self.model.fit(X_scaled, y_clean)

    def predict_signals(self, test_data: pd.DataFrame, threshold: float):
        """
        Генерация сигналов на основе вероятностей модели.
        """
        # Нормализация тестовых данных
        X_scaled = self.scaler.transform(test_data)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        signals = np.zeros_like(probs)
        
        signals[probs > threshold] = 1
        signals[probs < (1 - threshold)] = -1
        
        return pd.Series(signals, index=test_data.index)
