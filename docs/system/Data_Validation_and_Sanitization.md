# Валидация и санитизация данных

## 1. Принципы валидации

### 1.1 Философия
- **Validate Early**: Проверять данные на входе в систему
- **Fail Fast**: Отклонять некорректные данные до обработки
- **Explicit Schemas**: Явное описание ожидаемой структуры
- **Defensive Programming**: Не доверять никаким внешним данным

### 1.2 Уровни валидации
1. **Syntactic**: Тип данных, формат, структура
2. **Semantic**: Логическая корректность (например, close в пределах [low, high])
3. **Business Rules**: Предметная область (например, объём >= 0)
4. **Cross-field**: Согласованность между полями

## 2. Schema Validation

### 2.1 Pydantic Models

#### OHLCV Data
```python
from pydantic import BaseModel, validator, Field
from datetime import datetime
from decimal import Decimal

class OHLCVCandle(BaseModel):
    """Модель одной свечи OHLCV"""

    timestamp: datetime
    ticker: str = Field(..., min_length=1, max_length=50)
    open: Decimal = Field(..., gt=0, description="Цена открытия")
    high: Decimal = Field(..., gt=0, description="Максимальная цена")
    low: Decimal = Field(..., gt=0, description="Минимальная цена")
    close: Decimal = Field(..., gt=0, description="Цена закрытия")
    volume: Decimal = Field(..., ge=0, description="Объём")

    class Config:
        frozen = True  # Immutable
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }

    @validator('ticker')
    def ticker_uppercase(cls, v):
        """Нормализация тикера в верхний регистр"""
        return v.upper().strip()

    @validator('high')
    def high_gte_low_open_close(cls, v, values):
        """High должен быть >= low, open, close"""
        if 'low' in values and v < values['low']:
            raise ValueError(f'high ({v}) must be >= low ({values["low"]})')
        if 'open' in values and v < values['open']:
            raise ValueError(f'high ({v}) must be >= open ({values["open"]})')
        if 'close' in values and v < values['close']:
            raise ValueError(f'high ({v}) must be >= close ({values["close"]})')
        return v

    @validator('low')
    def low_lte_open_close(cls, v, values):
        """Low должен быть <= open, close"""
        if 'open' in values and v > values['open']:
            raise ValueError(f'low ({v}) must be <= open ({values["open"]})')
        if 'close' in values and v > values['close']:
            raise ValueError(f'low ({v}) must be <= close ({values["close"]})')
        return v

    @validator('volume')
    def volume_reasonable(cls, v):
        """Проверка разумности объёма (не слишком большой)"""
        MAX_VOLUME = Decimal('1e15')  # 1 квадриллион
        if v > MAX_VOLUME:
            raise ValueError(f'volume ({v}) exceeds maximum reasonable value')
        return v

    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return self.dict()

class OHLCVDataFrame(BaseModel):
    """Валидация целого DataFrame"""

    candles: list[OHLCVCandle]

    @validator('candles')
    def check_duplicates(cls, v):
        """Проверка на дубликаты по timestamp + ticker"""
        seen = set()
        for candle in v:
            key = (candle.timestamp, candle.ticker)
            if key in seen:
                raise ValueError(f'Duplicate candle: {key}')
            seen.add(key)
        return v

    @validator('candles')
    def check_chronological_order(cls, v):
        """Проверка хронологического порядка"""
        if len(v) < 2:
            return v

        ticker_timestamps = {}
        for candle in v:
            if candle.ticker not in ticker_timestamps:
                ticker_timestamps[candle.ticker] = []
            ticker_timestamps[candle.ticker].append(candle.timestamp)

        for ticker, timestamps in ticker_timestamps.items():
            if timestamps != sorted(timestamps):
                raise ValueError(f'Candles for {ticker} are not in chronological order')

        return v
```

#### Feature Data
```python
class FeatureVector(BaseModel):
    """Вектор признаков"""

    timestamp: datetime
    ticker: str
    features: dict[str, float] = Field(..., description="Признаки: имя -> значение")

    @validator('features')
    def check_no_inf_nan(cls, v):
        """Проверка на inf и NaN"""
        for name, value in v.items():
            if not isinstance(value, (int, float)):
                continue
            if math.isnan(value):
                raise ValueError(f'Feature {name} contains NaN')
            if math.isinf(value):
                raise ValueError(f'Feature {name} contains Inf')
        return v

    @validator('features')
    def check_expected_features(cls, v, values):
        """Проверка наличия ожидаемых признаков (если задан schema)"""
        # Эта проверка выполняется если есть ожидаемый список признаков
        expected_features = getattr(cls, '_expected_features', None)
        if expected_features:
            missing = expected_features - set(v.keys())
            if missing:
                raise ValueError(f'Missing expected features: {missing}')
        return v
```

#### Configuration
```python
class PipelineConfig(BaseModel):
    """Конфигурация пайплайна"""

    pipeline_name: str = Field(..., min_length=1)
    data_source: str = Field(..., regex=r'^(tinkoff_api|local_file)$')
    tickers: list[str] = Field(..., min_items=1)
    date_from: datetime
    date_to: datetime
    timeframe: str = Field(..., regex=r'^(1m|5m|15m|1h|4h|1d)$')

    @validator('date_to')
    def date_to_after_date_from(cls, v, values):
        """date_to должен быть позже date_from"""
        if 'date_from' in values and v <= values['date_from']:
            raise ValueError('date_to must be after date_from')
        return v

    @validator('tickers', each_item=True)
    def ticker_format(cls, v):
        """Валидация формата тикера"""
        if not re.match(r'^[A-Z0-9]+$', v):
            raise ValueError(f'Invalid ticker format: {v}')
        return v
```

### 2.2 JSON Schema для конфигураций

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Model Training Configuration",
  "type": "object",
  "required": ["model_type", "hyperparameters", "training"],
  "properties": {
    "model_type": {
      "type": "string",
      "enum": ["lightgbm", "catboost", "xgboost", "lstm", "tabnet"]
    },
    "hyperparameters": {
      "type": "object",
      "additionalProperties": true
    },
    "training": {
      "type": "object",
      "required": ["epochs", "batch_size"],
      "properties": {
        "epochs": {
          "type": "integer",
          "minimum": 1,
          "maximum": 1000
        },
        "batch_size": {
          "type": "integer",
          "minimum": 1,
          "maximum": 10000
        },
        "learning_rate": {
          "type": "number",
          "exclusiveMinimum": 0,
          "maximum": 1
        }
      }
    }
  }
}
```

```python
import jsonschema

def validate_config(config: dict, schema_path: Path):
    """Валидировать конфигурацию по JSON Schema"""
    with open(schema_path) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as e:
        raise ConfigValidationError(
            f"Configuration validation failed: {e.message}\n"
            f"Path: {' -> '.join(str(p) for p in e.path)}"
        )
```

## 3. Sanitization (Очистка данных)

### 3.1 Outlier Detection

```python
class OutlierDetector:
    """Детектор выбросов в данных"""

    def __init__(self, method: str = 'iqr', threshold: float = 3.0):
        """
        Args:
            method: 'iqr', 'zscore', 'isolation_forest'
            threshold: Порог для определения выброса
        """
        self.method = method
        self.threshold = threshold

    def detect(self, series: pd.Series) -> pd.Series:
        """
        Определить выбросы в серии

        Returns:
            Boolean mask, где True = outlier
        """
        if self.method == 'iqr':
            return self._detect_iqr(series)
        elif self.method == 'zscore':
            return self._detect_zscore(series)
        elif self.method == 'isolation_forest':
            return self._detect_isolation_forest(series)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_iqr(self, series: pd.Series) -> pd.Series:
        """IQR метод"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    def _detect_zscore(self, series: pd.Series) -> pd.Series:
        """Z-score метод"""
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        return z_scores > self.threshold

    def _detect_isolation_forest(self, series: pd.Series) -> pd.Series:
        """Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.01, random_state=42)
        predictions = clf.fit_predict(series.values.reshape(-1, 1))
        return predictions == -1

class DataSanitizer:
    """Очистка и нормализация данных"""

    def __init__(self, config: dict):
        self.config = config
        self.outlier_detector = OutlierDetector(
            method=config.get('outlier_method', 'iqr'),
            threshold=config.get('outlier_threshold', 3.0)
        )

    def sanitize_ohlcv(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Очистить OHLCV данные

        Returns:
            (cleaned_df, sanitization_report)
        """
        report = {
            'rows_original': len(df),
            'rows_removed': 0,
            'rows_corrected': 0,
            'issues': []
        }

        df = df.copy()

        # 1. Удалить дубликаты
        duplicates = df.duplicated(subset=['timestamp', 'ticker'])
        if duplicates.any():
            n_dupl = duplicates.sum()
            df = df[~duplicates]
            report['rows_removed'] += n_dupl
            report['issues'].append(f'Removed {n_dupl} duplicate rows')

        # 2. Проверить OHLC консистентность
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            # Попытка коррекции: high = max, low = min
            for idx in df[invalid_ohlc].index:
                prices = [
                    df.loc[idx, 'open'],
                    df.loc[idx, 'high'],
                    df.loc[idx, 'low'],
                    df.loc[idx, 'close']
                ]
                df.loc[idx, 'high'] = max(prices)
                df.loc[idx, 'low'] = min(prices)

            report['rows_corrected'] += n_invalid
            report['issues'].append(f'Corrected {n_invalid} OHLC inconsistencies')

        # 3. Проверить отрицательные значения
        negative_values = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0) |
            (df['volume'] < 0)
        )

        if negative_values.any():
            n_neg = negative_values.sum()
            df = df[~negative_values]
            report['rows_removed'] += n_neg
            report['issues'].append(f'Removed {n_neg} rows with negative values')

        # 4. Детектировать price spikes (выбросы)
        for price_col in ['open', 'high', 'low', 'close']:
            outliers = self.outlier_detector.detect(df[price_col])
            if outliers.any():
                n_outliers = outliers.sum()
                # Решение: удалить или заполнить?
                if self.config.get('remove_outliers', False):
                    df = df[~outliers]
                    report['rows_removed'] += n_outliers
                    report['issues'].append(
                        f'Removed {n_outliers} outliers in {price_col}'
                    )
                else:
                    # Forward fill
                    df.loc[outliers, price_col] = np.nan
                    df[price_col].fillna(method='ffill', inplace=True)
                    report['rows_corrected'] += n_outliers
                    report['issues'].append(
                        f'Filled {n_outliers} outliers in {price_col}'
                    )

        # 5. Проверить пропуски
        missing = df.isnull().sum()
        if missing.any():
            report['issues'].append(f'Missing values: {missing.to_dict()}')
            # Заполнить forward fill
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            report['rows_corrected'] += missing.sum()

        report['rows_final'] = len(df)

        return df, report
```

### 3.2 Missing Data Handling

```python
class MissingDataHandler:
    """Обработка пропущенных данных"""

    @staticmethod
    def analyze_missing(df: pd.DataFrame) -> dict:
        """Анализ паттернов пропусков"""
        total_rows = len(df)
        missing_info = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / total_rows * 100),
                    'first_missing_idx': int(df[col].isnull().idxmax()),
                    'longest_gap': MissingDataHandler._longest_gap(df[col])
                }

        return missing_info

    @staticmethod
    def _longest_gap(series: pd.Series) -> int:
        """Найти самый длинный последовательный пропуск"""
        is_null = series.isnull()
        gaps = is_null.ne(is_null.shift()).cumsum()
        gap_lengths = is_null.groupby(gaps).sum()
        return int(gap_lengths.max()) if len(gap_lengths) > 0 else 0

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        strategy: dict[str, str]
    ) -> pd.DataFrame:
        """
        Заполнить пропуски по стратегии

        Args:
            strategy: {column_name: method}
                methods: 'ffill', 'bfill', 'mean', 'median', 'zero', 'drop'
        """
        df = df.copy()

        for col, method in strategy.items():
            if col not in df.columns:
                continue

            if method == 'ffill':
                df[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df[col].fillna(method='bfill', inplace=True)
            elif method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'zero':
                df[col].fillna(0, inplace=True)
            elif method == 'drop':
                df = df[df[col].notnull()]
            else:
                raise ValueError(f"Unknown fill method: {method}")

        return df
```

### 3.3 Gap Detection

```python
class GapDetector:
    """Детектор пропусков во временных рядах"""

    def __init__(self, expected_interval: timedelta):
        """
        Args:
            expected_interval: Ожидаемый интервал между свечами
        """
        self.expected_interval = expected_interval

    def detect_gaps(self, df: pd.DataFrame) -> list[dict]:
        """
        Найти временные пропуски

        Returns:
            Список пропусков: [{start, end, duration_bars, reason}]
        """
        df = df.sort_values('timestamp')
        gaps = []

        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df['time_diff'] = ticker_df['timestamp'].diff()

            # Найти пропуски больше ожидаемого интервала
            gap_mask = ticker_df['time_diff'] > self.expected_interval * 1.5

            for idx in ticker_df[gap_mask].index:
                prev_timestamp = ticker_df.loc[idx - 1, 'timestamp']
                curr_timestamp = ticker_df.loc[idx, 'timestamp']
                duration = curr_timestamp - prev_timestamp
                expected_bars = duration / self.expected_interval

                gaps.append({
                    'ticker': ticker,
                    'start': prev_timestamp,
                    'end': curr_timestamp,
                    'duration': duration,
                    'expected_bars': int(expected_bars),
                    'reason': self._classify_gap(duration)
                })

        return gaps

    def _classify_gap(self, duration: timedelta) -> str:
        """Классифицировать причину пропуска"""
        if duration < timedelta(hours=1):
            return 'minor_gap'
        elif duration < timedelta(days=1):
            return 'intraday_gap'
        elif duration < timedelta(days=3):
            return 'weekend'
        else:
            return 'major_gap'
```

## 4. Look-ahead Detection

### 4.1 Temporal Validation

```python
class LookAheadDetector:
    """Детектор look-ahead bias в признаках и таргетах"""

    @staticmethod
    def check_feature_calculation(
        feature_func: callable,
        df: pd.DataFrame,
        feature_name: str
    ) -> bool:
        """
        Проверить, использует ли функция будущие данные

        Метод: Добавить будущие данные и проверить, изменится ли результат
        """
        # Вычислить признак на исходных данных
        original_result = feature_func(df)

        # Добавить фиктивные будущие данные
        future_df = df.copy()
        last_timestamp = df['timestamp'].max()
        future_row = df.iloc[-1].copy()
        future_row['timestamp'] = last_timestamp + timedelta(days=1)
        future_row['close'] *= 2  # Значительное изменение
        future_df = pd.concat([future_df, future_row.to_frame().T])

        # Пересчитать признак
        modified_result = feature_func(future_df)

        # Сравнить результаты на исходной части
        original_values = original_result.iloc[:-1]
        modified_values = modified_result.iloc[:-2]  # Исключить добавленную строку

        # Если значения изменились - есть look-ahead
        has_lookahead = not np.allclose(
            original_values,
            modified_values,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True
        )

        if has_lookahead:
            logger.warning(
                f"Look-ahead detected in feature '{feature_name}': "
                f"Values changed when future data was added"
            )

        return has_lookahead

    @staticmethod
    def validate_label_timing(
        labels: pd.Series,
        features: pd.DataFrame
    ) -> bool:
        """
        Проверить, что таргеты созданы корректно относительно признаков

        Таргет для timestamp T должен быть рассчитан по данным > T
        """
        # Проверить, что индексы совпадают
        if not labels.index.equals(features.index):
            raise ValueError("Labels and features must have same index")

        # Дополнительные проверки зависят от способа создания таргетов
        # Например, для horizon-based labels:
        # - Проверить, что horizon параметр > 0
        # - Проверить, что label для timestamp T не использует данные <= T

        return True
```

## 5. Data Quality Metrics

### 5.1 Quality Score

```python
class DataQualityChecker:
    """Оценка качества данных"""

    def calculate_quality_score(self, df: pd.DataFrame) -> dict:
        """
        Рассчитать метрики качества

        Returns:
            Словарь с метриками качества (0-100)
        """
        metrics = {}

        # 1. Completeness: Полнота данных
        total_cells = df.size
        non_null_cells = df.notnull().sum().sum()
        metrics['completeness'] = (non_null_cells / total_cells) * 100

        # 2. Validity: Доля валидных значений
        valid_mask = self._check_validity(df)
        metrics['validity'] = (valid_mask.sum().sum() / total_cells) * 100

        # 3. Consistency: OHLC консистентность
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            consistent = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            )
            metrics['consistency'] = (consistent.sum() / len(df)) * 100

        # 4. Uniqueness: Доля уникальных записей
        if 'timestamp' in df.columns and 'ticker' in df.columns:
            duplicates = df.duplicated(subset=['timestamp', 'ticker'])
            metrics['uniqueness'] = ((len(df) - duplicates.sum()) / len(df)) * 100

        # 5. Timeliness: Актуальность (последняя дата)
        if 'timestamp' in df.columns:
            last_date = df['timestamp'].max()
            age_days = (datetime.now() - last_date).days
            metrics['timeliness_age_days'] = age_days
            metrics['timeliness_score'] = max(0, 100 - age_days)  # 1% за день

        # Общий score (среднее)
        scores = [v for k, v in metrics.items() if k.endswith('_score') or k in ['completeness', 'validity', 'consistency', 'uniqueness']]
        metrics['overall_quality'] = np.mean(scores) if scores else 0

        return metrics

    def _check_validity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проверить валидность каждого значения"""
        valid_mask = pd.DataFrame(True, index=df.index, columns=df.columns)

        # Numeric columns должны быть конечными
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            valid_mask[col] = np.isfinite(df[col])

        # Price columns должны быть положительными
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        for col in price_cols:
            valid_mask[col] &= (df[col] > 0)

        # Volume должен быть неотрицательным
        if 'volume' in df.columns:
            valid_mask['volume'] &= (df['volume'] >= 0)

        return valid_mask
```

## 6. Конфигурация валидации

### 6.1 Validation Rules

```yaml
# configs/validation/data_validation.yaml
ohlcv_validation:
  required_columns: ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']

  numeric_ranges:
    open: {min: 0, max: 1000000}
    high: {min: 0, max: 1000000}
    low: {min: 0, max: 1000000}
    close: {min: 0, max: 1000000}
    volume: {min: 0, max: 1e15}

  consistency_checks:
    - name: "high_gte_low"
      condition: "high >= low"
      action: "correct"  # or "reject"

    - name: "high_gte_open_close"
      condition: "high >= open AND high >= close"
      action: "correct"

    - name: "low_lte_open_close"
      condition: "low <= open AND low <= close"
      action: "correct"

  outlier_detection:
    method: "iqr"  # or "zscore", "isolation_forest"
    threshold: 3.0
    action: "flag"  # "flag", "remove", "fill"

  missing_data:
    max_missing_percent: 5
    fill_strategy:
      open: "ffill"
      high: "ffill"
      low: "ffill"
      close: "ffill"
      volume: "zero"

  gap_detection:
    enabled: true
    max_gap_multiplier: 1.5  # Относительно expected_interval

  quality_thresholds:
    completeness: 95
    validity: 99
    consistency: 99
    uniqueness: 100

feature_validation:
  allowed_nan_percent: 10
  check_lookahead: true
  check_inf: true

  feature_ranges:
    # Можно задать ожидаемые диапазоны для признаков
    sma_20: {min: 0, max: null}
    rsi_14: {min: 0, max: 100}
    volume: {min: 0, max: null}
```

## 7. Integration

### 7.1 Validation Pipeline

```python
class ValidationPipeline:
    """Пайплайн валидации данных"""

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.sanitizer = DataSanitizer(self.config['ohlcv_validation'])
        self.quality_checker = DataQualityChecker()

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        data_type: str = 'ohlcv'
    ) -> tuple[pd.DataFrame, dict]:
        """
        Выполнить полную валидацию и очистку

        Returns:
            (cleaned_df, validation_report)
        """
        report = {
            'timestamp': datetime.now(),
            'data_type': data_type,
            'status': 'success',
            'errors': [],
            'warnings': []
        }

        try:
            # 1. Schema validation
            if data_type == 'ohlcv':
                self._validate_ohlcv_schema(df, report)

            # 2. Sanitization
            df_clean, sanitization_report = self.sanitizer.sanitize_ohlcv(df)
            report['sanitization'] = sanitization_report

            # 3. Quality metrics
            quality_metrics = self.quality_checker.calculate_quality_score(df_clean)
            report['quality_metrics'] = quality_metrics

            # 4. Check quality thresholds
            thresholds = self.config['ohlcv_validation']['quality_thresholds']
            for metric, threshold in thresholds.items():
                if quality_metrics.get(metric, 0) < threshold:
                    report['warnings'].append(
                        f"{metric} ({quality_metrics[metric]:.1f}%) "
                        f"below threshold ({threshold}%)"
                    )

            return df_clean, report

        except Exception as e:
            report['status'] = 'failed'
            report['errors'].append(str(e))
            raise DataValidationError(f"Validation failed: {e}") from e

    def _validate_ohlcv_schema(self, df: pd.DataFrame, report: dict):
        """Валидация схемы OHLCV"""
        required_cols = self.config['ohlcv_validation']['required_columns']
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Попытка валидации через Pydantic (sample)
        sample_size = min(100, len(df))
        sample = df.sample(sample_size)

        validation_errors = []
        for idx, row in sample.iterrows():
            try:
                OHLCVCandle(**row.to_dict())
            except Exception as e:
                validation_errors.append(f"Row {idx}: {e}")

        if validation_errors:
            report['warnings'].extend(validation_errors[:10])  # Первые 10
            if len(validation_errors) > 10:
                report['warnings'].append(
                    f"... and {len(validation_errors) - 10} more validation errors"
                )
```

## 8. Best Practices

### 8.1 Рекомендации
- ✅ Валидировать данные на входе в систему
- ✅ Использовать Pydantic для type safety
- ✅ Определять схемы для всех конфигураций
- ✅ Логировать все проблемы с данными
- ✅ Создавать отчёты о качестве данных
- ✅ Автоматизировать детект look-ahead bias

### 8.2 Антипаттерны
- ❌ Молчаливое игнорирование некорректных данных
- ❌ Автоматическое удаление выбросов без анализа
- ❌ Отсутствие проверки целостности OHLC
- ❌ Пропуск валидации "доверенных" источников
