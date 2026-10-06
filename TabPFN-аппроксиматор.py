import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tabpfn import TabPFNClassifier
import torch

class QuantumStrategy:
    def __init__(self, device='cpu'):
        # Инициализируем только TabPFN. 
        # N_ensemble_configurations=32 дает высокую точность аппроксимации
        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=32)
        self.features = []
        self.sigma = 2.0

    def prepare_targets(self, data: pd.DataFrame, sigma: float):
        """
        УЧИТЕЛЬ (Гаусс): Расчет идеального таргета.
        Используем 1-ю производную для определения направления тренда.
        """
        self.sigma = sigma
        math_offset = int(4 * sigma)
        
        # Сглаженная производная (идеал, видящий будущее)
        derivative = gaussian_filter1d(data['close'], sigma=sigma, order=1)
        
        # Бинарный таргет: 1 - рост, 0 - падение
        targets = (derivative > 0).astype(int)
        
        # Применяем MATH_OFFSET: зануляем края, где Гаусс нестабилен
        targets_series = pd.Series(targets, index=data.index)
        targets_series.iloc[-math_offset:] = np.nan 
        targets_series.iloc[:math_offset] = np.nan
        
        return targets_series

    def add_features(self, data: pd.DataFrame):
        """
        Генерация казуальных признаков (только прошлое).
        """
        df = data.copy()
        
        # 1. Индикаторы (все со сдвигом .shift(1), чтобы не заглядывать в текущий бар)
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema12 - ema26).shift(1)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = (100 - (100 / (1 + gain / (loss + 1e-9)))).shift(1)
        
        # BB Position
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_pos'] = ((df['close'] - sma20) / (2 * std20 + 1e-9)).shift(1)
        
        # Изменение волатильности
        df['atr'] = (df['high'] - df['low']).rolling(14).mean().shift(1)
        df['vol_regime'] = df['atr'] / (df['atr'].rolling(50).mean() + 1e-9)
        
        # Доходности разных периодов
        df['ret1'] = df['close'].pct_change(1).shift(1)
        df['ret5'] = df['close'].pct_change(5).shift(1)
        
        # Временные фичи (Сезонность)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day'] = df.index.dayofweek
        else:
            df['hour'] = 0
            df['day'] = 0

        return df.dropna()

    def train_student(self, train_data: pd.DataFrame, targets: pd.Series):
        """
        ОБУЧЕНИЕ УЧЕНИКА (TabPFN): Дистилляция знаний Учителя.
        """
        # Динамическое определение фичей
        exclude = ['open', 'high', 'low', 'close', 'volume', 'Target', 'Conv_Deriv', 'True_Deriv']
        self.features = [c for c in train_data.columns if c not in exclude]
        
        # Синхронизируем фичи и таргеты
        common_idx = train_data.index.intersection(targets.index)
        X = train_data.loc[common_idx, self.features]
        y = targets.loc[common_idx]
        
        # Убираем NaN (краевые эффекты Гаусса)
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # TabPFN — трансформер, он лучше всего работает на контексте из последних 1000-1200 баров
        X_context = X_clean.iloc[-1100:]
        y_context = y_clean.iloc[-1100:]
        
        print(f"TabPFN Student обучается на {len(X_context)} барах...")
        self.model.fit(X_context, y_context)

    def predict_signals(self, test_data: pd.DataFrame, threshold: float):
        """
        Генерация сигналов на основе предсказаний трансформера.
        """
        X = test_data[self.features]
        # Получаем вероятность класса 1 (рост по Гауссу)
        probs = self.model.predict_proba(X)[:, 1]
        
        signals = np.zeros_like(probs)
        # Если вероятность выше порога — Buy (1)
        # Если вероятность ниже (1 - порог) — Sell (-1)
        signals[probs > threshold] = 1
        signals[probs < (1 - threshold)] = -1
        
        return pd.Series(signals, index=test_data.index), probs