# Безопасность

## 1. Общие принципы

### 1.1 Defense in Depth
Многоуровневая защита:
- Сетевой уровень (firewall)
- Приложение (валидация, авторизация)
- Данные (шифрование)
- Мониторинг и реагирование

### 1.2 Principle of Least Privilege
Минимально необходимые права доступа для каждого компонента.

### 1.3 Security by Design
Безопасность учитывается на этапе проектирования, а не добавляется потом.

## 2. Управление секретами

### 2.1 Хранение секретов

**❌ НИКОГДА не делать:**
```python
# НЕ хардкодить секреты в коде!
API_TOKEN = "t.xxxxxxxxxxxxxxxxxxx"
DATABASE_PASSWORD = "mypassword123"
AWS_SECRET_KEY = "aws_secret_key_here"
```

**✅ Правильно:**

#### Переменные окружения
```python
import os

API_TOKEN = os.getenv('TINKOFF_API_TOKEN')
if not API_TOKEN:
    raise ValueError("TINKOFF_API_TOKEN must be set")
```

#### .env файлы (НЕ коммитить в Git!)
```bash
# .env (добавить в .gitignore!)
TINKOFF_API_TOKEN=t.xxxxxxxxx
DATABASE_URL=postgresql://user:pass@localhost/db
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=my-bucket
```

```python
# Загрузка с python-dotenv
from dotenv import load_dotenv
import os

load_dotenv()  # Загружает переменные из .env

api_token = os.getenv('TINKOFF_API_TOKEN')
```

#### Keyring для локального хранения
```python
import keyring

# Сохранить секрет
keyring.set_password("trading-platform", "tinkoff_api_token", "t.xxxxxx")

# Получить секрет
api_token = keyring.get_password("trading-platform", "tinkoff_api_token")
```

#### Централизованное хранилище (для production)
```python
# Использование AWS Secrets Manager / HashiCorp Vault
import boto3

def get_secret(secret_name: str) -> dict:
    """Получить секрет из AWS Secrets Manager"""
    client = boto3.client('secretsmanager', region_name='us-east-1')

    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Использование
secrets = get_secret('trading-platform/production')
api_token = secrets['tinkoff_api_token']
```

### 2.2 Ротация секретов

```python
class SecretManager:
    """Управление секретами с автоматической ротацией"""

    def __init__(self):
        self.secrets_cache = {}
        self.last_refresh = {}
        self.refresh_interval = timedelta(hours=1)

    def get_secret(self, key: str) -> str:
        """
        Получить секрет с автоматическим обновлением

        Args:
            key: Ключ секрета

        Returns:
            Значение секрета
        """
        now = datetime.now()

        # Проверить кэш и время обновления
        if key in self.secrets_cache:
            if now - self.last_refresh.get(key, now) < self.refresh_interval:
                return self.secrets_cache[key]

        # Загрузить свежий секрет
        secret_value = self._fetch_from_vault(key)
        self.secrets_cache[key] = secret_value
        self.last_refresh[key] = now

        return secret_value

    def _fetch_from_vault(self, key: str) -> str:
        """Загрузить секрет из хранилища"""
        # Реализация зависит от используемого хранилища
        pass
```

### 2.3 .gitignore для секретов

```gitignore
# .gitignore - обязательно добавить
.env
.env.local
.env.*.local
*.key
*.pem
*.p12
secrets/
credentials/
*.secret
config/production.yaml  # Если содержит секреты
```

## 3. API Security

### 3.1 Защита API ключей

```python
class SecureAPIClient:
    """API клиент с безопасным управлением токенами"""

    def __init__(self, token: str = None):
        """
        Args:
            token: API токен (опционально, загрузит из env)
        """
        self.token = token or self._load_token()
        self._validate_token()

    def _load_token(self) -> str:
        """Загрузить токен из безопасного источника"""
        token = os.getenv('TINKOFF_API_TOKEN')
        if not token:
            # Попробовать keyring
            token = keyring.get_password("trading-platform", "tinkoff_api_token")

        if not token:
            raise ValueError(
                "API token not found. Set TINKOFF_API_TOKEN environment variable "
                "or store in system keyring"
            )

        return token

    def _validate_token(self):
        """Проверить формат токена"""
        if not self.token.startswith('t.'):
            raise ValueError("Invalid token format")

        if len(self.token) < 50:
            raise ValueError("Token too short, possibly corrupted")

    def _get_headers(self) -> dict:
        """Получить заголовки с токеном"""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def make_request(self, endpoint: str, **kwargs):
        """Выполнить API запрос"""
        headers = self._get_headers()

        # НЕ логировать headers с токеном!
        logger.debug(f"Making request to {endpoint}")

        response = requests.post(endpoint, headers=headers, **kwargs)
        return response
```

### 3.2 Rate Limiting для безопасности

```python
from collections import defaultdict
from time import time

class SecurityRateLimiter:
    """Rate limiter для предотвращения abuse"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """
        Проверить, разрешен ли запрос для клиента

        Args:
            client_id: Идентификатор клиента

        Returns:
            True если запрос разрешен
        """
        now = time()
        cutoff = now - self.window_seconds

        # Очистить старые запросы
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff
        ]

        # Проверить лимит
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Добавить текущий запрос
        self.requests[client_id].append(now)
        return True
```

## 4. Input Validation

### 4.1 Валидация пользовательского ввода

```python
import re
from pathlib import Path
from typing import Any

class InputValidator:
    """Валидация входных данных для предотвращения инъекций"""

    @staticmethod
    def validate_ticker(ticker: str) -> str:
        """
        Валидация тикера

        Разрешены только буквы и цифры
        """
        if not re.match(r'^[A-Z0-9]+$', ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")

        if len(ticker) > 10:
            raise ValueError(f"Ticker too long: {ticker}")

        return ticker

    @staticmethod
    def validate_file_path(
        file_path: str,
        allowed_extensions: list = ['.parquet', '.csv'],
        allowed_directories: list = ['data/', 'artifacts/']
    ) -> Path:
        """
        Валидация пути к файлу

        Предотвращает path traversal attacks
        """
        path = Path(file_path).resolve()

        # Проверить расширение
        if path.suffix not in allowed_extensions:
            raise ValueError(f"File extension not allowed: {path.suffix}")

        # Проверить что путь в разрешенной директории
        is_allowed = False
        for allowed_dir in allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path.relative_to(allowed_path)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            raise SecurityError(
                f"File path outside allowed directories: {file_path}"
            )

        return path

    @staticmethod
    def sanitize_sql_input(value: Any) -> Any:
        """
        Санитизация SQL ввода

        Предотвращает SQL injection
        """
        if isinstance(value, str):
            # Удалить потенциально опасные символы
            dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
            for char in dangerous_chars:
                if char in value:
                    raise ValueError(f"Dangerous character detected: {char}")

        return value

    @staticmethod
    def validate_config_yaml(config_path: str) -> dict:
        """
        Безопасная загрузка YAML конфигурации

        Использует safe_load для предотвращения code execution
        """
        import yaml

        path = InputValidator.validate_file_path(
            config_path,
            allowed_extensions=['.yaml', '.yml'],
            allowed_directories=['configs/']
        )

        with open(path) as f:
            # ВСЕГДА используйте safe_load, НЕ load!
            config = yaml.safe_load(f)

        return config
```

### 4.2 Sanitization

```python
import html
import bleach

class DataSanitizer:
    """Очистка данных от потенциально опасного содержимого"""

    @staticmethod
    def sanitize_html(text: str) -> str:
        """Удалить HTML теги"""
        return html.escape(text)

    @staticmethod
    def sanitize_user_input(text: str, max_length: int = 1000) -> str:
        """
        Общая санитизация пользовательского ввода

        Args:
            text: Входной текст
            max_length: Максимальная длина

        Returns:
            Очищенный текст
        """
        # Ограничить длину
        text = text[:max_length]

        # Удалить управляющие символы
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Экранировать HTML
        text = html.escape(text)

        return text.strip()
```

## 5. Защита данных

### 5.1 Шифрование чувствительных данных

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

class DataEncryptor:
    """Шифрование чувствительных данных"""

    def __init__(self, password: str):
        """
        Args:
            password: Мастер-пароль для шифрования
        """
        self.key = self._derive_key(password)
        self.cipher = Fernet(self.key)

    def _derive_key(self, password: str) -> bytes:
        """Получить ключ шифрования из пароля"""
        salt = b'trading_platform_salt_2025'  # В production использовать random salt

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, data: str) -> str:
        """
        Зашифровать данные

        Args:
            data: Данные для шифрования

        Returns:
            Зашифрованная строка (base64)
        """
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Расшифровать данные

        Args:
            encrypted_data: Зашифрованные данные

        Returns:
            Расшифрованная строка
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()

# Использование
encryptor = DataEncryptor(password=os.getenv('ENCRYPTION_PASSWORD'))

# Шифрование API токена перед сохранением
encrypted_token = encryptor.encrypt(api_token)
# Сохранить encrypted_token в файл/БД

# Расшифрование при использовании
api_token = encryptor.decrypt(encrypted_token)
```

### 5.2 Шифрование файлов

```python
def encrypt_file(input_file: Path, output_file: Path, password: str):
    """Зашифровать файл"""
    encryptor = DataEncryptor(password)

    with open(input_file, 'rb') as f:
        data = f.read()

    encrypted = encryptor.cipher.encrypt(data)

    with open(output_file, 'wb') as f:
        f.write(encrypted)

def decrypt_file(input_file: Path, output_file: Path, password: str):
    """Расшифровать файл"""
    encryptor = DataEncryptor(password)

    with open(input_file, 'rb') as f:
        encrypted_data = f.read()

    decrypted = encryptor.cipher.decrypt(encrypted_data)

    with open(output_file, 'wb') as f:
        f.write(decrypted)
```

### 5.3 Маскирование PII

```python
import re

class PIIMasker:
    """Маскирование персональной информации"""

    @staticmethod
    def mask_email(email: str) -> str:
        """Маскировать email: user@example.com -> u***@example.com"""
        if '@' not in email:
            return email

        user, domain = email.split('@')
        masked_user = user[0] + '***' if len(user) > 1 else '***'
        return f"{masked_user}@{domain}"

    @staticmethod
    def mask_api_token(token: str) -> str:
        """Маскировать API токен: показать только первые и последние символы"""
        if len(token) <= 10:
            return '***'
        return f"{token[:4]}...{token[-4:]}"

    @staticmethod
    def mask_card_number(card: str) -> str:
        """Маскировать номер карты: 1234567890123456 -> 1234-****-****-3456"""
        digits = re.sub(r'\D', '', card)
        if len(digits) != 16:
            return '****'
        return f"{digits[:4]}-****-****-{digits[-4:]}"

    @staticmethod
    def mask_sensitive_data(text: str) -> str:
        """Автоматически маскировать чувствительные данные в тексте"""
        # Email
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '***@***.***',
            text
        )

        # API tokens (начинаются с t.)
        text = re.sub(
            r't\.[A-Za-z0-9_-]{50,}',
            't.***REDACTED***',
            text
        )

        # Номера карт
        text = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '****-****-****-****',
            text
        )

        return text

# Использование в логах
logger.info(PIIMasker.mask_sensitive_data(
    f"User {user_email} authenticated with token {api_token}"
))
```

## 6. Аудит безопасности

### 6.1 Security Logging

```python
class SecurityAuditLogger:
    """Логирование событий безопасности"""

    def __init__(self):
        self.logger = logging.getLogger('security_audit')

        # Отдельный handler для security логов
        handler = logging.FileHandler('logs/security_audit.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_authentication_attempt(
        self,
        success: bool,
        user: str = None,
        ip_address: str = None
    ):
        """Логировать попытку аутентификации"""
        self.logger.info(json.dumps({
            'event': 'authentication_attempt',
            'success': success,
            'user': user,
            'ip_address': ip_address,
            'timestamp': datetime.now().isoformat()
        }))

    def log_api_key_usage(self, api_key_id: str, endpoint: str):
        """Логировать использование API ключа"""
        self.logger.info(json.dumps({
            'event': 'api_key_usage',
            'api_key_id': api_key_id,  # НЕ логировать сам ключ!
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat()
        }))

    def log_suspicious_activity(
        self,
        activity_type: str,
        details: dict
    ):
        """Логировать подозрительную активность"""
        self.logger.warning(json.dumps({
            'event': 'suspicious_activity',
            'type': activity_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }))
```

### 6.2 Dependency Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  dependency-scan:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run Bandit (security linter)
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json

      - name: Run Safety (dependency vulnerability scan)
        run: |
          pip install safety
          safety check --json

      - name: Run Trivy (container scan)
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
```

### 6.3 Regular Security Audits

```bash
# Скрипт для регулярного аудита
#!/bin/bash

echo "Running security audit..."

# 1. Сканирование зависимостей
pip-audit

# 2. Поиск секретов в Git истории
trufflehog git file://.

# 3. Статический анализ безопасности
bandit -r src/ -ll

# 4. Проверка конфигураций
safety check

# 5. Анализ Docker образов
trivy image trading-platform:latest

echo "Security audit complete"
```

## 7. Best Practices

### 7.1 Checklist
- [ ] Все секреты в переменных окружения или vault
- [ ] .env файл добавлен в .gitignore
- [ ] Входные данные валидируются
- [ ] Чувствительные данные зашифрованы
- [ ] Логируются security events
- [ ] Регулярное сканирование зависимостей
- [ ] API rate limiting настроен
- [ ] PII данные маскируются в логах
- [ ] HTTPS для всех внешних соединений
- [ ] Регулярная ротация секретов

### 7.2 Антипаттерны
- ❌ Хардкод секретов в коде
- ❌ Коммит .env файлов в Git
- ❌ Логирование паролей/токенов
- ❌ Использование yaml.load() вместо safe_load()
- ❌ Отсутствие валидации пользовательского ввода
- ❌ SQL queries с конкатенацией строк
- ❌ Хранение паролей в plaintext
- ❌ Отсутствие мониторинга security events
