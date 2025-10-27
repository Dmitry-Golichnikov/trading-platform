# Этап 19: CI/CD и тестирование

## Цель
Настроить полный CI/CD пайплайн с автоматическими проверками и тестами.

## Зависимости
Все предыдущие этапы

## CI/CD Pipeline

### GitHub Actions Workflows

```
.github/workflows/
├── ci.yml                        # Main CI pipeline
├── tests.yml                     # Tests
├── docker.yml                    # Docker builds
├── docs.yml                      # Documentation
└── nightly.yml                   # Nightly checks
```

### 1. CI Pipeline (ci.yml)
```yaml
name: CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Lint with flake8
        run: flake8 src/ tests/

      - name: Type check with mypy
        run: mypy src/

      - name: Format check with black
        run: black --check src/ tests/

      - name: Check import order
        run: isort --check src/ tests/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Security scan with bandit
        run: bandit -r src/

      - name: Dependency scan
        run: safety check --json

  validate-configs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate YAML configs
        run: python scripts/validate_configs.py
```

### 2. Tests Pipeline (tests.yml)
```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/ -v

  regression-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run regression tests
        run: pytest tests/regression/ -v
```

### 3. Docker Build (docker.yml)
```yaml
name: Docker

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push CPU
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            username/trading-platform:cpu-latest
            username/trading-platform:cpu-${{ github.sha }}

      - name: Build and push GPU
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.gpu
          push: true
          tags: |
            username/trading-platform:gpu-latest
            username/trading-platform:gpu-${{ github.sha }}
```

### 4. Documentation (docs.yml)
```yaml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Sphinx docs
        run: |
          pip install sphinx sphinx-rtd-theme
          cd docs && make html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

### 5. Nightly Checks (nightly.yml)
```yaml
name: Nightly

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  dependency-updates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for outdated dependencies
        run: pip list --outdated

  performance-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: pytest tests/benchmarks/ --benchmark-only

  reference-backtests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run reference backtests
        run: python scripts/run_reference_backtests.py
```

## Test Strategy (docs/system/Testing_Strategy.md)

### Test Pyramid
- 60% Unit tests
- 30% Integration tests
- 10% E2E tests

### Coverage Requirements
- Minimum: 80%
- Critical modules: 90%+

### Performance Tests
```python
# tests/benchmarks/test_performance.py
def test_feature_calculation_performance(benchmark):
    data = generate_large_dataset(100_000)
    benchmark(calculate_all_features, data)

@pytest.mark.slow
def test_full_backtest_performance():
    """Full backtest should complete in < 5 minutes for 1M bars"""
    start = time.time()
    run_backtest(large_dataset)
    duration = time.time() - start
    assert duration < 300
```

## Pre-commit Checks

Все PR должны пройти:
- [ ] Linting (flake8, mypy)
- [ ] Formatting (black, isort)
- [ ] Unit tests (coverage > 80%)
- [ ] Integration tests
- [ ] Security scan
- [ ] Config validation

## Критерии готовности

- [ ] GitHub Actions workflows настроены
- [ ] CI запускается на каждый push/PR
- [ ] Все типы тестов работают
- [ ] Docker images собираются
- [ ] Documentation деплоится
- [ ] Nightly checks настроены
- [ ] Branch protection rules
- [ ] Coverage tracking (Codecov)
- [ ] Badges в README

## Промпты

```
Настрой CI/CD:

1. Создай .github/workflows/:
   - ci.yml - linting, type checking
   - tests.yml - unit, integration, regression tests
   - docker.yml - build and push Docker images
   - docs.yml - build and deploy documentation
   - nightly.yml - scheduled checks

2. Настрой pre-commit hooks (уже есть из Этапа 00)

3. Создай scripts/:
   - validate_configs.py - валидация YAML
   - run_reference_backtests.py - reference tests
   - check_performance_regression.py

4. Обнови README с badges:
   - Build status
   - Test coverage
   - Docker pulls
   - License

5. Branch protection для main:
   - Require PR
   - Require passing checks
   - Require code review
```

## Важные замечания

**Secrets:**
Хранить в GitHub Secrets:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
- CODECOV_TOKEN

**Caching:**
Кэшировать зависимости для ускорения:
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

**Matrix Testing:**
Тестировать на нескольких версиях Python (3.9, 3.10, 3.11).

**Artifacts:**
Сохранять артефакты тестов:
```yaml
- uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-results
    path: test-results/
```

## Следующий этап
[Этап 20: Документация и финализация](Этап_20_Документация_и_финализация.md)
