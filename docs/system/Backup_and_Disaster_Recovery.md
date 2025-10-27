# Резервное копирование и восстановление

## 1. Принципы резервного копирования

### 1.1 Стратегия 3-2-1
- **3 копии данных**: Оригинал + 2 backup
- **2 разных носителя**: Локальный диск + внешнее хранилище
- **1 копия offsite**: Облако (S3/MinIO) или удалённый сервер

### 1.2 Приоритеты backup
1. **Critical** (ежедневно): Модели, конфигурации, код
2. **Important** (еженедельно): Данные, признаки, результаты бэктестов
3. **Nice-to-have** (по необходимости): Логи, кэши, временные файлы

## 2. Что бэкапить

### 2.1 Артефакты для backup

```yaml
# configs/backup/backup_policy.yaml
backup_items:
  # Модели (критично)
  models:
    path: "artifacts/models/"
    frequency: "daily"
    retention_days: 180
    priority: "critical"
    versioning: true

  # Конфигурации (критично)
  configs:
    path: "configs/"
    frequency: "on_change"
    retention_days: 365
    priority: "critical"
    versioning: true

  # Данные (важно)
  datasets:
    path: "artifacts/data/"
    frequency: "weekly"
    retention_days: 90
    priority: "important"
    incremental: true

  # Feature store (важно)
  features:
    path: "artifacts/features/"
    frequency: "weekly"
    retention_days: 60
    priority: "important"
    incremental: true

  # Результаты бэктестов (важно)
  backtests:
    path: "artifacts/backtests/"
    frequency: "weekly"
    retention_days: 90
    priority: "important"

  # Эксперименты MLflow (важно)
  mlflow_artifacts:
    path: "mlruns/"
    frequency: "daily"
    retention_days: 90
    priority: "important"

  # База данных (если используется)
  database:
    frequency: "daily"
    retention_days: 30
    priority: "critical"

  # Код (критично)
  source_code:
    path: "."
    frequency: "on_push"  # Git hook
    retention_days: 365
    priority: "critical"
    exclude: [".git", "venv", "__pycache__", "*.pyc"]

  # Логи (опционально)
  logs:
    path: "logs/"
    frequency: "weekly"
    retention_days: 30
    priority: "low"
    compression: true

# Исключения (не бэкапить)
exclude_patterns:
  - "__pycache__/"
  - "*.pyc"
  - ".git/"
  - "venv/"
  - "node_modules/"
  - ".env"
  - "*.tmp"
  - "cache/"
```

## 3. Backup System

### 3.1 Backup Manager

```python
from pathlib import Path
import shutil
import hashlib
from datetime import datetime, timedelta
import boto3

class BackupManager:
    """Управление резервными копиями"""

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.backup_items = self.config['backup_items']
        self.s3_client = self._setup_s3()
        self.backup_log = []

    def run_scheduled_backups(self):
        """Выполнить запланированные backup'ы"""
        for name, item_config in self.backup_items.items():
            if self._should_backup(name, item_config):
                try:
                    self.backup_item(name, item_config)
                    logger.info(f"Backup completed: {name}")
                except Exception as e:
                    logger.error(f"Backup failed for {name}: {e}")

    def backup_item(self, name: str, config: dict):
        """
        Создать backup для элемента

        Args:
            name: Название элемента
            config: Конфигурация backup
        """
        source_path = Path(config['path'])

        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return

        # Создать временный архив
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{name}_{timestamp}"
        temp_archive = Path(f"/tmp/{backup_name}.tar.gz")

        # Архивировать
        self._create_archive(
            source_path,
            temp_archive,
            exclude=self.config.get('exclude_patterns', [])
        )

        # Рассчитать хеш
        file_hash = self._calculate_hash(temp_archive)

        # Метаданные
        metadata = {
            'name': name,
            'timestamp': timestamp,
            'source_path': str(source_path),
            'size_bytes': temp_archive.stat().st_size,
            'hash': file_hash,
            'priority': config['priority']
        }

        # Загрузить в S3
        s3_key = f"backups/{name}/{backup_name}.tar.gz"
        self._upload_to_s3(temp_archive, s3_key, metadata)

        # Сохранить локально (опционально)
        if config.get('keep_local', False):
            local_backup_dir = Path(f"backups/{name}")
            local_backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(temp_archive, local_backup_dir / f"{backup_name}.tar.gz")

        # Очистить временный файл
        temp_archive.unlink()

        # Логировать backup
        self.backup_log.append(metadata)
        self._save_backup_log()

        # Очистить старые backup'ы
        self._cleanup_old_backups(name, config)

    def _create_archive(self, source: Path, output: Path, exclude: list):
        """Создать tar.gz архив"""
        import tarfile

        with tarfile.open(output, 'w:gz') as tar:
            # Добавить фильтр исключений
            def filter_func(tarinfo):
                for pattern in exclude:
                    if pattern in tarinfo.name:
                        return None
                return tarinfo

            tar.add(source, arcname=source.name, filter=filter_func)

    def _calculate_hash(self, filepath: Path) -> str:
        """Рассчитать SHA256 хеш файла"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _upload_to_s3(self, filepath: Path, s3_key: str, metadata: dict):
        """Загрузить файл в S3"""
        bucket = os.getenv('BACKUP_S3_BUCKET', 'trading-platform-backups')

        # Добавить метаданные как теги
        self.s3_client.upload_file(
            str(filepath),
            bucket,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'hash': metadata['hash'],
                    'priority': metadata['priority']
                }
            }
        )

        logger.info(f"Uploaded backup to s3://{bucket}/{s3_key}")

    def _should_backup(self, name: str, config: dict) -> bool:
        """Проверить, нужно ли делать backup"""
        frequency = config['frequency']

        if frequency == 'on_change':
            # Проверить изменения с последнего backup
            return self._has_changes_since_last_backup(name, config)

        # Проверить время последнего backup
        last_backup = self._get_last_backup_time(name)
        if not last_backup:
            return True

        now = datetime.now()
        if frequency == 'daily':
            return (now - last_backup) >= timedelta(days=1)
        elif frequency == 'weekly':
            return (now - last_backup) >= timedelta(weeks=1)
        elif frequency == 'hourly':
            return (now - last_backup) >= timedelta(hours=1)

        return False

    def _cleanup_old_backups(self, name: str, config: dict):
        """Удалить старые backup'ы согласно retention policy"""
        retention_days = config.get('retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        bucket = os.getenv('BACKUP_S3_BUCKET')
        prefix = f"backups/{name}/"

        # Получить список backup'ов
        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return

        for obj in response['Contents']:
            # Извлечь timestamp из имени
            backup_date = self._extract_date_from_key(obj['Key'])

            if backup_date and backup_date < cutoff_date:
                self.s3_client.delete_object(
                    Bucket=bucket,
                    Key=obj['Key']
                )
                logger.info(f"Deleted old backup: {obj['Key']}")

    def _setup_s3(self):
        """Настроить S3 клиент"""
        return boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
```

### 3.2 Incremental Backups

```python
class IncrementalBackupManager(BackupManager):
    """Инкрементальные backup'ы для больших датасетов"""

    def backup_incremental(self, name: str, config: dict):
        """Создать инкрементальный backup"""
        source_path = Path(config['path'])

        # Получить список изменённых файлов
        last_backup_time = self._get_last_backup_time(name)

        if not last_backup_time:
            # Первый backup - полный
            return self.backup_item(name, config)

        # Найти изменённые файлы
        changed_files = []
        for file in source_path.rglob('*'):
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime > last_backup_time:
                    changed_files.append(file)

        if not changed_files:
            logger.info(f"No changes detected for {name}, skipping backup")
            return

        # Создать инкрементальный архив
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{name}_incremental_{timestamp}"
        temp_archive = Path(f"/tmp/{backup_name}.tar.gz")

        with tarfile.open(temp_archive, 'w:gz') as tar:
            for file in changed_files:
                tar.add(file, arcname=file.relative_to(source_path))

        # Загрузить
        s3_key = f"backups/{name}/incremental/{backup_name}.tar.gz"
        metadata = {
            'name': name,
            'type': 'incremental',
            'timestamp': timestamp,
            'files_count': len(changed_files),
            'base_backup': self._get_last_full_backup(name)
        }
        self._upload_to_s3(temp_archive, s3_key, metadata)

        temp_archive.unlink()
```

## 4. Восстановление (Restore)

### 4.1 Restore Manager

```python
class RestoreManager:
    """Восстановление из backup"""

    def __init__(self, s3_client):
        self.s3_client = s3_client

    def list_available_backups(self, item_name: str = None) -> list:
        """Получить список доступных backup'ов"""
        bucket = os.getenv('BACKUP_S3_BUCKET')
        prefix = f"backups/{item_name}/" if item_name else "backups/"

        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        backups = []
        for obj in response.get('Contents', []):
            backups.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
                'name': self._extract_name_from_key(obj['Key'])
            })

        return sorted(backups, key=lambda x: x['last_modified'], reverse=True)

    def restore_item(
        self,
        item_name: str,
        target_path: Path,
        backup_timestamp: str = None,
        verify_hash: bool = True
    ):
        """
        Восстановить элемент из backup

        Args:
            item_name: Название элемента
            target_path: Куда восстановить
            backup_timestamp: Конкретный backup или None для последнего
            verify_hash: Проверять целостность
        """
        # Найти нужный backup
        if backup_timestamp:
            s3_key = f"backups/{item_name}/{item_name}_{backup_timestamp}.tar.gz"
        else:
            # Последний backup
            backups = self.list_available_backups(item_name)
            if not backups:
                raise ValueError(f"No backups found for {item_name}")
            s3_key = backups[0]['key']

        # Скачать
        temp_archive = Path(f"/tmp/restore_{item_name}.tar.gz")
        bucket = os.getenv('BACKUP_S3_BUCKET')

        self.s3_client.download_file(bucket, s3_key, str(temp_archive))
        logger.info(f"Downloaded backup from s3://{bucket}/{s3_key}")

        # Проверить хеш
        if verify_hash:
            metadata = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            expected_hash = metadata['Metadata'].get('hash')

            if expected_hash:
                actual_hash = self._calculate_hash(temp_archive)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch! Expected {expected_hash}, got {actual_hash}"
                    )
                logger.info("Hash verification passed")

        # Распаковать
        target_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(temp_archive, 'r:gz') as tar:
            tar.extractall(target_path)

        logger.info(f"Restored {item_name} to {target_path}")

        # Очистить временный файл
        temp_archive.unlink()

    def restore_incremental_chain(
        self,
        item_name: str,
        target_path: Path,
        up_to_timestamp: str = None
    ):
        """Восстановить из полного backup + инкрементальные"""
        # 1. Восстановить последний полный backup
        full_backups = [
            b for b in self.list_available_backups(item_name)
            if 'incremental' not in b['key']
        ]

        if not full_backups:
            raise ValueError(f"No full backups found for {item_name}")

        base_backup = full_backups[0]
        self.restore_item(item_name, target_path, verify_hash=True)

        # 2. Применить инкрементальные backup'ы
        incremental_backups = [
            b for b in self.list_available_backups(item_name)
            if 'incremental' in b['key']
        ]

        for inc_backup in sorted(incremental_backups, key=lambda x: x['last_modified']):
            if up_to_timestamp and inc_backup['name'] > up_to_timestamp:
                break

            # Скачать и применить
            self._apply_incremental(inc_backup['key'], target_path)

    def _calculate_hash(self, filepath: Path) -> str:
        """Рассчитать SHA256"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
```

### 4.2 Point-in-Time Recovery

```python
class PointInTimeRecovery:
    """Восстановление на конкретную точку времени"""

    def __init__(self, backup_manager, restore_manager):
        self.backup_manager = backup_manager
        self.restore_manager = restore_manager

    def restore_to_datetime(self, target_datetime: datetime, items: list = None):
        """
        Восстановить состояние системы на указанную дату/время

        Args:
            target_datetime: Целевая дата восстановления
            items: Список элементов или None для всех
        """
        if items is None:
            items = self.backup_manager.backup_items.keys()

        restore_plan = {}

        for item_name in items:
            # Найти ближайший backup до target_datetime
            backups = self.restore_manager.list_available_backups(item_name)

            suitable_backup = None
            for backup in backups:
                if backup['last_modified'] <= target_datetime:
                    suitable_backup = backup
                    break

            if suitable_backup:
                restore_plan[item_name] = suitable_backup
            else:
                logger.warning(
                    f"No backup found for {item_name} before {target_datetime}"
                )

        # Выполнить восстановление
        for item_name, backup_info in restore_plan.items():
            config = self.backup_manager.backup_items[item_name]
            target_path = Path(config['path'])

            # Сделать backup текущего состояния
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            current_backup_path = Path(f"backups/before_restore/{item_name}_{timestamp}")
            if target_path.exists():
                shutil.copytree(target_path, current_backup_path)

            # Восстановить
            try:
                self.restore_manager.restore_item(
                    item_name,
                    target_path,
                    verify_hash=True
                )
                logger.info(f"Restored {item_name} to {target_datetime}")
            except Exception as e:
                logger.error(f"Failed to restore {item_name}: {e}")
                # Откатить
                if current_backup_path.exists():
                    shutil.rmtree(target_path)
                    shutil.copytree(current_backup_path, target_path)
```

## 5. Disaster Recovery Plan

### 5.1 DR Сценарии

```yaml
# configs/backup/disaster_recovery.yaml
disaster_scenarios:
  # Сценарий 1: Полная потеря локального диска
  total_disk_failure:
    rto: 4h  # Recovery Time Objective
    rpo: 24h  # Recovery Point Objective

    steps:
      - name: "Setup new environment"
        actions:
          - "Install OS and dependencies"
          - "Clone repository from GitHub"
          - "Setup Python environment"

      - name: "Restore configurations"
        actions:
          - "Download configs from S3"
          - "Restore secrets from vault"

      - name: "Restore critical data"
        priority: "critical"
        actions:
          - "Restore latest models"
          - "Restore MLflow experiments"

      - name: "Restore important data"
        priority: "important"
        actions:
          - "Restore datasets"
          - "Restore feature store"

      - name: "Verify and test"
        actions:
          - "Run health checks"
          - "Test model inference"
          - "Verify backtest reproduction"

  # Сценарий 2: Corrupt model/data
  data_corruption:
    rto: 1h
    rpo: 24h

    steps:
      - "Identify corrupted items"
      - "Restore from last known good backup"
      - "Verify integrity"
      - "Resume operations"

  # Сценарий 3: Accidental deletion
  accidental_deletion:
    rto: 30m
    rpo: 24h

    steps:
      - "Identify deleted items"
      - "Restore from backup"
      - "Verify restoration"
```

### 5.2 DR Automation

```python
class DisasterRecoveryOrchestrator:
    """Автоматизация disaster recovery"""

    def __init__(self, dr_config_path: Path):
        with open(dr_config_path) as f:
            self.dr_config = yaml.safe_load(f)

    def execute_recovery_scenario(self, scenario_name: str):
        """Выполнить сценарий восстановления"""
        scenario = self.dr_config['disaster_scenarios'].get(scenario_name)

        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        logger.info(
            f"Starting disaster recovery: {scenario_name}\n"
            f"RTO: {scenario['rto']}, RPO: {scenario['rpo']}"
        )

        start_time = time.time()

        for step in scenario['steps']:
            logger.info(f"Executing step: {step['name']}")

            try:
                self._execute_step(step)
            except Exception as e:
                logger.error(f"Step failed: {step['name']}: {e}")
                raise

        duration = time.time() - start_time
        logger.info(
            f"Disaster recovery completed in {duration/3600:.2f} hours"
        )

        # Проверить RTO
        rto_seconds = self._parse_duration(scenario['rto'])
        if duration > rto_seconds:
            logger.warning(
                f"Recovery took longer than RTO: "
                f"{duration/3600:.2f}h > {rto_seconds/3600:.2f}h"
            )

    def _execute_step(self, step: dict):
        """Выполнить шаг восстановления"""
        if isinstance(step, dict):
            for action in step.get('actions', []):
                self._execute_action(action)
        else:
            self._execute_action(step)

    def _execute_action(self, action: str):
        """Выполнить конкретное действие"""
        # Mapping действий на функции
        # Реализация зависит от конкретных действий
        logger.info(f"Executing action: {action}")
```

## 6. Testing Backups

### 6.1 Backup Verification

```python
class BackupVerifier:
    """Проверка корректности backup'ов"""

    def verify_backup(self, backup_path: Path) -> dict:
        """
        Проверить backup

        Returns:
            Отчёт о проверке
        """
        report = {
            'valid': True,
            'checks': [],
            'errors': []
        }

        # 1. Проверка целостности архива
        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                members = tar.getmembers()
                report['checks'].append({
                    'name': 'archive_integrity',
                    'status': 'pass',
                    'files_count': len(members)
                })
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"Archive corrupted: {e}")

        # 2. Проверка хеша
        if backup_path.exists():
            actual_hash = self._calculate_hash(backup_path)
            # Сравнить с сохранённым хешем из метаданных
            report['checks'].append({
                'name': 'hash_check',
                'status': 'pass',
                'hash': actual_hash
            })

        # 3. Проверка размера
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.001:  # Менее 1KB - подозрительно
            report['valid'] = False
            report['errors'].append(f"Backup too small: {size_mb:.2f} MB")

        return report

    def test_restore(self, backup_path: Path, verify: bool = True) -> bool:
        """
        Тестовое восстановление

        Args:
            backup_path: Путь к backup
            verify: Проверять корректность восстановленных данных

        Returns:
            True если успешно
        """
        import tempfile

        # Восстановить во временную директорию
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)

                if verify:
                    # Дополнительные проверки восстановленных данных
                    self._verify_restored_data(Path(temp_dir))

                logger.info(f"Test restore successful for {backup_path}")
                return True

            except Exception as e:
                logger.error(f"Test restore failed: {e}")
                return False
```

### 6.2 Automated Backup Tests

```python
class BackupTestScheduler:
    """Периодическое тестирование backup'ов"""

    def __init__(self, restore_manager, verifier):
        self.restore_manager = restore_manager
        self.verifier = verifier

    def run_weekly_tests(self):
        """Еженедельные тесты восстановления"""
        # Выбрать случайные backup'ы для тестирования
        all_backups = self.restore_manager.list_available_backups()

        # Тестировать последние backup'ы критичных элементов
        critical_items = ['models', 'configs', 'database']

        for item_name in critical_items:
            backups = [b for b in all_backups if item_name in b['key']]
            if backups:
                latest = backups[0]

                # Скачать и проверить
                temp_path = Path(f"/tmp/test_backup_{item_name}.tar.gz")
                # ... скачать backup

                result = self.verifier.test_restore(temp_path)

                if not result:
                    # Отправить алерт
                    logger.error(f"Backup test failed for {item_name}!")
                    # send_alert(...)
```

## 7. Best Practices

### 7.1 Рекомендации
- ✅ Автоматизировать backup'ы (не полагаться на ручные действия)
- ✅ Регулярно тестировать восстановление
- ✅ Хранить backup'ы offsite (облако, удалённый сервер)
- ✅ Шифровать backup'ы с чувствительными данными
- ✅ Документировать процедуры восстановления
- ✅ Версионировать backup'ы (не перезаписывать)
- ✅ Мониторить успешность backup'ов

### 7.2 Антипаттерны
- ❌ Backup'ы только на том же диске
- ❌ Никогда не тестировать восстановление
- ❌ Отсутствие автоматизации
- ❌ Незащищённые backup'ы с секретами
- ❌ Неограниченное хранение (заполнение диска)
- ❌ Отсутствие мониторинга backup'ов

### 7.3 Checklist
- [ ] Backup'ы настроены для всех критичных данных
- [ ] Backup'ы автоматические по расписанию
- [ ] Копии хранятся offsite (S3/облако)
- [ ] Проверяется целостность (хеши)
- [ ] Регулярно тестируется восстановление (хотя бы раз в квартал)
- [ ] Документирован disaster recovery план
- [ ] Установлены алерты на failed backups
- [ ] Retention policy настроена и работает
