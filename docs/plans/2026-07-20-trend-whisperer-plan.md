# TrendWhisperer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TrendWhisperer — a standalone, forge-dependent repo that refactors `Projects/sale_forecasting/Time+series+forecasting.ipynb`'s per-store sales forecasting (VAR/VARMAX/VECM) into a tested `src/{core,providers,services}` pipeline, adds a GenAI narrative/anomaly-explanation layer via `forge.llm`, and produces a self-contained HTML report via `forge.report`.

**Architecture:** `src/core/interfaces.py` defines `ForecastProvider`; `src/providers/` holds three swappable statsmodels-backed implementations (VAR/VARMAX/VECM); `src/services/` holds preprocessing, diagnostics, orchestration (forecast), narrative generation, and report assembly; `src/app.py` is the CLI driver; `scripts/fetch_data.py` pulls the Kaggle Rossmann dataset.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, statsmodels, `sameer-forge[llm,stats]` (pinned to v0.2.0), `kaggle` CLI/package, pytest.

## Global Constraints

- Python `>=3.11`, target-version `py311`, line-length 120 (matches the scaffolded `pyproject.toml`).
- Ruff pinned to `0.11.13` (matches the scaffold's generated CI — do not change).
- No `typing.List`/`Dict`/`Optional` — use `list[str]`, `dict[str, Any]`, `X | None`.
- No `print()` outside `src/app.py` and `scripts/fetch_data.py` (the only CLI-facing entry points) — everything in `src/services/` and `src/providers/` uses `logging`.
- Commit messages use `feat:`/`fix:`/`test:`/`docs:`/`chore:`/`init:` imperative-mood prefixes, no AI/Claude/Anthropic attribution anywhere.
- Task 1 is repo genesis: its commits land on the default branch (`main`, created by `forge new`'s `git init`), after which a `dev` branch is created from that same commit and pushed. This mirrors how `the-forge` itself began (initial commits directly on `main` before `dev` existed) — it is not a violation of "never commit to main/dev," since there is no prior history to protect at genesis.
- After Task 1, all further work (Tasks 2-12) happens on a single branch `feature/forecast-pipeline`, branched from `dev` — never committed directly to `dev` or `main`. This branch gets one PR into `dev` at the end of this plan; promoting `dev` to `main` (tag, release) is a separate follow-up step, out of scope for this plan (matches how the forge scaffold-cli plan handled its own release).
- Real network calls (Kaggle download, live LLM calls to Ollama/Claude/OpenAI) are never made in tests — every test uses synthetic in-memory data and stub providers.
- Working directory for Task 1: `/Users/sameermaurya/Downloads/dev` (creates the repo). Working directory for Tasks 2-12: `/Users/sameermaurya/Downloads/dev/trend-whisperer`.

---

### Task 1: Scaffold the repo, create the GitHub remote, land the spec/plan

**Files:**
- Create: the entire `trend-whisperer` repo via `forge new`
- Modify: `requirements.txt` (add extras + deps `forge new` doesn't know about)
- Create: `docs/specs/2026-07-20-trend-whisperer-design.md`, `docs/plans/2026-07-20-trend-whisperer-plan.md`

**Interfaces:**
- Produces: a working `trend-whisperer` repo at `/Users/sameermaurya/Downloads/dev/trend-whisperer`, with `dev` and `main` branches pushed to `github.com/mauryasameer/trend-whisperer`, ready for `feature/forecast-pipeline` to branch off `dev`.

- [ ] **Step 1: Bootstrap a Python 3.12 venv with sameer-forge installed, to run `forge new`**

```bash
/opt/homebrew/bin/python3.12 -m venv /tmp/trend-whisperer-bootstrap
source /tmp/trend-whisperer-bootstrap/bin/activate
pip install -q --upgrade pip
pip install -e /Users/sameermaurya/Downloads/dev/the-forge
python -c "import forge; print(forge.__version__)"
```

Expected: prints `0.2.0` (the-forge's current released version — if it prints something else, STOP and report NEEDS_CONTEXT, since Step 4's `requirements.txt` dependency line depends on this being `0.2.0`).

- [ ] **Step 2: Scaffold the project**

```bash
cd /Users/sameermaurya/Downloads/dev
forge new trend-whisperer
```

Expected: prints a created-files list; `ls trend-whisperer` shows `src/`, `tests/`, `scripts/`, `.github/`, `conftest.py`, `pyproject.toml`, `requirements.txt`, `VERSION`, `CHANGELOG.md`, `task.md`, `README.md`, `.gitignore`; `trend-whisperer/.git` exists.

- [ ] **Step 3: Extend the generated `requirements.txt`**

Read `trend-whisperer/requirements.txt` first — it will contain exactly one line:
`sameer-forge @ git+https://github.com/mauryasameer/the-forge.git@v0.2.0`

Replace that line's package spec to include the `llm` and `stats` extras (needed for `OllamaProvider`/`ClaudeProvider`/`OpenAIProvider` and `statsmodels` respectively), and add `scipy` and `kaggle` as direct dependencies (this project imports `scipy.stats.mstats` and shells out to the `kaggle` CLI directly, so both need to be explicit, not just transitive):

```
sameer-forge[llm,stats] @ git+https://github.com/mauryasameer/the-forge.git@v0.2.0
scipy>=1.11
kaggle>=1.6
```

- [ ] **Step 4: Land the spec and plan documents**

```bash
cd /Users/sameermaurya/Downloads/dev/trend-whisperer
mkdir -p docs/specs docs/plans
cp "/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/551ffc43-a065-4d3f-b66e-ad4166c6f479/scratchpad/2026-07-20-trend-whisperer-design.md" docs/specs/
cp "/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/551ffc43-a065-4d3f-b66e-ad4166c6f479/scratchpad/2026-07-20-trend-whisperer-plan.md" docs/plans/
```

(If either source path no longer exists because the scratchpad was cleaned up, STOP and report NEEDS_CONTEXT — do not fabricate replacement content.)

- [ ] **Step 5: Commit the scaffold**

```bash
cd /Users/sameermaurya/Downloads/dev/trend-whisperer
git add -A
git status --short
git commit -m "init: scaffold trend-whisperer via forge new"
git log --oneline -1
git branch --show-current
```

Expected: commit succeeds; the branch shown is whatever `forge new`'s `git init` created as the default (almost certainly `main`).

- [ ] **Step 6: Create the `dev` branch and the GitHub repo, push both**

```bash
cd /Users/sameermaurya/Downloads/dev/trend-whisperer
git checkout -b dev
gh repo create mauryasameer/trend-whisperer --public --source=. --remote=origin
git push -u origin main
git push -u origin dev
```

Expected: `gh repo create` succeeds and prints the new repo URL; both pushes succeed. If `gh repo create` fails because a repo of that name already exists, STOP and report BLOCKED with the exact error — do not force-overwrite or delete anything.

- [ ] **Step 7: Verify the venv can install from the real generated requirements.txt**

```bash
deactivate
rm -rf /tmp/trend-whisperer-bootstrap
/opt/homebrew/bin/python3.12 -m venv /Users/sameermaurya/Downloads/dev/trend-whisperer/.venv
source /Users/sameermaurya/Downloads/dev/trend-whisperer/.venv/bin/activate
pip install -q --upgrade pip
pip install -r /Users/sameermaurya/Downloads/dev/trend-whisperer/requirements.txt
pip install -r /Users/sameermaurya/Downloads/dev/trend-whisperer/requirements.txt --quiet && echo "install ok"
python -c "import forge, scipy, kaggle; print('deps ok')"
```

Expected: `deps ok` prints with no errors. This venv (`trend-whisperer/.venv`, already covered by the scaffold's `.gitignore` pattern for nothing-in-particular — verify `.venv` is ignored; if not, add `.venv/` to `trend-whisperer/.gitignore` and commit that as a follow-up in this same step) is what Tasks 2-12 will use.

- [ ] **Step 8: Report**

Report back with:
- **Status:** DONE | BLOCKED | NEEDS_CONTEXT
- Commits created (short SHA + subject)
- Confirmation of: `forge.__version__` printed in Step 1, GitHub repo URL from Step 6, `deps ok` from Step 7
- The report file path (write full details to `/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/551ffc43-a065-4d3f-b66e-ad4166c6f479/scratchpad/trend-whisperer-task-1-report.md`)

---

### Task 2: Core interfaces — `ForecastProvider`

**Files:**
- Create: `src/core/interfaces.py`
- Test: `tests/unit/test_interfaces.py`

**Interfaces:**
- Produces: `ForecastProvider(ABC)` with abstract methods `fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None` and `forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame`. Consumed by every provider in Tasks 5-7 and referenced (as a type) by `forecast_service.py` in Task 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_interfaces.py
import pandas as pd
import pytest

from src.core.interfaces import ForecastProvider


class _StubProvider(ForecastProvider):
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"forecast": [0.0] * steps})


def test_cannot_instantiate_abstract_provider():
    with pytest.raises(TypeError):
        ForecastProvider()


def test_stub_provider_implements_interface():
    provider = _StubProvider()
    provider.fit(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}))
    result = provider.forecast(steps=3, exog=pd.DataFrame({"b": [1, 2, 3]}))
    assert len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.interfaces'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastProvider(ABC):
    """Abstract interface for all time-series forecasting backends.

    Concrete implementations live in src/providers/. Swap providers by
    changing one constructor argument — no service code changes.
    """

    @abstractmethod
    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        """Fit the model on historical endogenous/exogenous data."""
        ...

    @abstractmethod
    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        """Forecast `steps` periods ahead given future exogenous values."""
        ...
```

Also create `src/core/__init__.py` (empty) if it doesn't already exist as part of the scaffold's `.gitkeep`-only `src/core/` directory.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_interfaces.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/core/interfaces.py src/core/__init__.py tests/unit/test_interfaces.py
git commit -m "feat: add ForecastProvider interface"
```

---

### Task 3: Preprocessing service

**Files:**
- Create: `src/services/preprocessing.py`
- Test: `tests/unit/test_preprocessing.py`

**Interfaces:**
- Produces: `winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series`, `one_hot_day_of_week(df: pd.DataFrame, column: str = "DayOfWeek") -> pd.DataFrame`, `seasonal_features(series: pd.Series, period: int = 7, prefix: str = "") -> pd.DataFrame` (returns a DataFrame with `f"{prefix}Seasonality"` and `f"{prefix}Trend"` columns). Consumed by `forecast_service.py` in Task 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_preprocessing.py
import numpy as np
import pandas as pd

from src.services.preprocessing import one_hot_day_of_week, seasonal_features, winsorize_series


def test_winsorize_series_caps_extremes():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    result = winsorize_series(series, limits=(0.2, 0.2))
    assert result.max() < 1000.0
    assert len(result) == len(series)


def test_winsorize_series_preserves_index():
    series = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    result = winsorize_series(series)
    assert list(result.index) == [10, 11, 12]


def test_one_hot_day_of_week_adds_columns():
    df = pd.DataFrame({"DayOfWeek": [0, 1, 2, 0]})
    result = one_hot_day_of_week(df)
    assert "DayOfWeek_0" in result.columns
    assert "DayOfWeek_1" in result.columns
    assert "DayOfWeek_2" in result.columns
    assert result["DayOfWeek_0"].tolist() == [1, 0, 0, 1]


def test_one_hot_day_of_week_preserves_original_columns():
    df = pd.DataFrame({"DayOfWeek": [0, 1], "Sales": [100, 200]})
    result = one_hot_day_of_week(df)
    assert "Sales" in result.columns
    assert result["Sales"].tolist() == [100, 200]


def test_seasonal_features_returns_seasonality_and_trend_columns():
    values = [10 + (i % 7) + i * 0.1 for i in range(21)]
    series = pd.Series(values)
    result = seasonal_features(series, period=7, prefix="Sales")
    assert "SalesSeasonality" in result.columns
    assert "SalesTrend" in result.columns
    assert len(result) == len(series)
    assert not np.isnan(result["SalesSeasonality"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.preprocessing'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/preprocessing.py
from __future__ import annotations

import pandas as pd


def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    from scipy.stats import mstats

    return pd.Series(mstats.winsorize(series.to_numpy(), limits=list(limits)), index=series.index)


def one_hot_day_of_week(df: pd.DataFrame, column: str = "DayOfWeek") -> pd.DataFrame:
    dummies = pd.get_dummies(df[column], prefix=column).astype(int)
    return pd.concat([df, dummies], axis=1)


def seasonal_features(series: pd.Series, period: int = 7, prefix: str = "") -> pd.DataFrame:
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    return pd.DataFrame(
        {
            f"{prefix}Seasonality": result.seasonal,
            f"{prefix}Trend": result.trend,
        }
    )
```

Create `src/services/__init__.py` (empty) if it doesn't already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_preprocessing.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/preprocessing.py src/services/__init__.py tests/unit/test_preprocessing.py
git commit -m "feat: add preprocessing service (winsorize, day-of-week, seasonal features)"
```

---

### Task 4: Diagnostics service

**Files:**
- Create: `src/services/diagnostics.py`
- Test: `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: `forge.eval.timeseries.adf_stationarity(series: np.ndarray) -> dict[str, float | bool]` (already exists in the-forge, returns keys `adf_stat`, `p_value`, `is_stationary`, `critical_1pct`, `critical_5pct`).
- Produces: `check_stationarity(series: pd.Series) -> dict[str, float | bool]`, `granger_causality(df: pd.DataFrame, maxlag: int = 1) -> dict[int, dict[str, float]]`, `cointegration_rank(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> int`. Consumed by `forecast_service.py` in Task 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_diagnostics.py
import numpy as np
import pandas as pd

from src.services.diagnostics import check_stationarity, cointegration_rank, granger_causality


def test_check_stationarity_detects_stationary_series():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 200))
    result = check_stationarity(series)
    assert result["is_stationary"] is True
    assert "p_value" in result


def test_check_stationarity_detects_nonstationary_series():
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    result = check_stationarity(series)
    assert result["is_stationary"] is False


def test_granger_causality_returns_p_values_per_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = np.roll(x, 1) + rng.normal(0, 0.1, 100)
    df = pd.DataFrame({"y": y, "x": x})
    result = granger_causality(df, maxlag=1)
    assert 1 in result
    assert "p_value" in result[1]
    assert isinstance(result[1]["p_value"], float)


def test_cointegration_rank_returns_nonnegative_int():
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 1, 100))
    df = pd.DataFrame({"a": base + rng.normal(0, 0.1, 100), "b": base * 2 + rng.normal(0, 0.1, 100)})
    rank = cointegration_rank(df)
    assert isinstance(rank, int)
    assert rank >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.diagnostics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/diagnostics.py
from __future__ import annotations

import pandas as pd

from forge.eval.timeseries import adf_stationarity


def check_stationarity(series: pd.Series) -> dict[str, float | bool]:
    return adf_stationarity(series.to_numpy())


def granger_causality(df: pd.DataFrame, maxlag: int = 1) -> dict[int, dict[str, float]]:
    from statsmodels.tsa.stattools import grangercausalitytests

    results = grangercausalitytests(df.dropna(), maxlag, verbose=False)
    return {lag: {"p_value": float(result[0]["ssr_ftest"][1])} for lag, result in results.items()}


def cointegration_rank(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> int:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(df, det_order, k_ar_diff)
    traces = result.lr1
    cvts = result.cvt
    rank = 0
    for i in range(len(traces)):
        if traces[i] > cvts[i, 1]:
            rank = i + 1
    return rank
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_diagnostics.py -v`
Expected: PASS (4 passed). Note: `test_check_stationarity_detects_nonstationary_series` and the causality/cointegration tests depend on statistical properties of seeded random data — if a specific seed produces a borderline/flaky result, try `np.random.default_rng` seeds `42`/`0`/`1` as given first (chosen to be comfortably non-borderline); only change seeds if a test genuinely fails, and note the change in your report.

- [ ] **Step 5: Commit**

```bash
git add src/services/diagnostics.py tests/unit/test_diagnostics.py
git commit -m "feat: add diagnostics service (stationarity, causality, cointegration)"
```

---

### Task 5: VAR forecast provider

**Files:**
- Create: `src/providers/var_provider.py`
- Test: `tests/unit/test_var_provider.py`

**Interfaces:**
- Consumes: `ForecastProvider` from `src/core/interfaces.py` (Task 2).
- Produces: `VARForecastProvider(ForecastProvider)`, constructor `__init__(self, maxlags: int | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_var_provider.py
import numpy as np
import pandas as pd
import pytest

from src.providers.var_provider import VARForecastProvider


def _synthetic_data(n: int = 60):
    rng = np.random.default_rng(7)
    endog = pd.DataFrame(
        {
            "Sales": 100 + np.arange(n) * 0.5 + rng.normal(0, 2, n),
            "Customers": 20 + np.arange(n) * 0.1 + rng.normal(0, 1, n),
        }
    )
    exog = pd.DataFrame(
        {
            "Promo": rng.integers(0, 2, n).astype(float),
        }
    )
    return endog, exog


def test_forecast_before_fit_raises():
    provider = VARForecastProvider()
    with pytest.raises(RuntimeError):
        provider.forecast(steps=3, exog=pd.DataFrame({"Promo": [0.0, 1.0, 0.0]}))


def test_fit_then_forecast_returns_expected_shape():
    endog, exog = _synthetic_data()
    provider = VARForecastProvider()
    provider.fit(endog, exog)

    future_exog = pd.DataFrame({"Promo": [0.0, 1.0, 0.0, 1.0, 0.0]})
    result = provider.forecast(steps=5, exog=future_exog)

    assert len(result) == 5
    assert list(result.columns) == ["Sales", "Customers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_var_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.providers.var_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/var_provider.py
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR

from src.core.interfaces import ForecastProvider


class VARForecastProvider(ForecastProvider):
    def __init__(self, maxlags: int | None = None) -> None:
        self._maxlags = maxlags
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VAR(endog, exog=exog)
        self._result = model.fit(maxlags=self._maxlags) if self._maxlags else model.fit()

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VARForecastProvider.forecast() called before fit()")
        lag_order = self._result.k_ar
        forecast_input = self._result.endog[-lag_order:] if lag_order > 0 else self._result.endog
        values = self._result.forecast(y=forecast_input, steps=steps, exog_future=exog.to_numpy())
        return pd.DataFrame(values, columns=self._endog_columns)
```

Create `src/providers/__init__.py` (empty) if it doesn't already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_var_provider.py -v`
Expected: PASS (2 passed). If `VAR.fit()` raises on the synthetic data (e.g. a `ValueError` about lag order vs. sample size), increase `n` in `_synthetic_data` (try 100) rather than changing the provider code — the provider's job is to wrap `statsmodels.tsa.api.VAR` faithfully, not to work around insufficient test data.

- [ ] **Step 5: Commit**

```bash
git add src/providers/var_provider.py src/providers/__init__.py tests/unit/test_var_provider.py
git commit -m "feat: add VAR forecast provider"
```

---

### Task 6: VARMAX forecast provider

**Files:**
- Create: `src/providers/varmax_provider.py`
- Test: `tests/unit/test_varmax_provider.py`

**Interfaces:**
- Consumes: `ForecastProvider` from `src/core/interfaces.py` (Task 2).
- Produces: `VARMAXForecastProvider(ForecastProvider)`, constructor `__init__(self, order: tuple[int, int] = (6, 1), trend: str = "n")`. This is the plan's default provider (matches the notebook's final model, and `src/app.py`'s `--provider` default in Task 11).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_varmax_provider.py
import numpy as np
import pandas as pd
import pytest

from src.providers.varmax_provider import VARMAXForecastProvider


def _synthetic_data(n: int = 60):
    rng = np.random.default_rng(11)
    endog = pd.DataFrame(
        {
            "Sales": 100 + np.arange(n) * 0.5 + rng.normal(0, 2, n),
            "Customers": 20 + np.arange(n) * 0.1 + rng.normal(0, 1, n),
        }
    )
    exog = pd.DataFrame(
        {
            "Promo": rng.integers(0, 2, n).astype(float),
        }
    )
    return endog, exog


def test_forecast_before_fit_raises():
    provider = VARMAXForecastProvider()
    with pytest.raises(RuntimeError):
        provider.forecast(steps=3, exog=pd.DataFrame({"Promo": [0.0, 1.0, 0.0]}))


def test_fit_then_forecast_returns_expected_shape():
    endog, exog = _synthetic_data()
    provider = VARMAXForecastProvider(order=(1, 0), trend="n")
    provider.fit(endog, exog)

    future_exog = pd.DataFrame({"Promo": [0.0, 1.0, 0.0, 1.0, 0.0]})
    result = provider.forecast(steps=5, exog=future_exog)

    assert len(result) == 5
    assert list(result.columns) == ["Sales", "Customers"]
```

Note: the test uses `order=(1, 0)` rather than the design's production default `(6, 1)` — a `(6, 1)` order needs substantially more observations to fit reliably and quickly in a unit test; the constructor's default stays `(6, 1)` per the design (used in production via `src/app.py`), this test just exercises the class with a cheaper, still-representative order.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_varmax_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.providers.varmax_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/varmax_provider.py
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VARMAX

from src.core.interfaces import ForecastProvider


class VARMAXForecastProvider(ForecastProvider):
    def __init__(self, order: tuple[int, int] = (6, 1), trend: str = "n") -> None:
        self._order = order
        self._trend = trend
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VARMAX(endog, exog=exog, order=self._order, trend=self._trend)
        self._result = model.fit(maxiter=1000, disp=False)

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VARMAXForecastProvider.forecast() called before fit()")
        values = self._result.forecast(steps=steps, exog=exog)
        return pd.DataFrame(values.to_numpy(), columns=self._endog_columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_varmax_provider.py -v`
Expected: PASS (2 passed). Convergence warnings from `model.fit(maxiter=1000, disp=False)` on synthetic data are acceptable and not a failure — only a raised exception or wrong-shaped output is a problem. If fitting genuinely fails to converge to a usable result, try increasing `n` in `_synthetic_data` before changing provider code.

- [ ] **Step 5: Commit**

```bash
git add src/providers/varmax_provider.py tests/unit/test_varmax_provider.py
git commit -m "feat: add VARMAX forecast provider"
```

---

### Task 7: VECM forecast provider

**Files:**
- Create: `src/providers/vecm_provider.py`
- Test: `tests/unit/test_vecm_provider.py`

**Interfaces:**
- Consumes: `ForecastProvider` from `src/core/interfaces.py` (Task 2).
- Produces: `VECMForecastProvider(ForecastProvider)`, constructor `__init__(self, k_ar_diff: int = 1, coint_rank: int = 1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vecm_provider.py
import numpy as np
import pandas as pd
import pytest

from src.providers.vecm_provider import VECMForecastProvider


def _synthetic_data(n: int = 80):
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 1, n))
    endog = pd.DataFrame(
        {
            "Sales": 100 + base + rng.normal(0, 1, n),
            "Customers": 20 + base * 0.2 + rng.normal(0, 0.5, n),
        }
    )
    exog = pd.DataFrame(
        {
            "Promo": rng.integers(0, 2, n).astype(float),
        }
    )
    return endog, exog


def test_forecast_before_fit_raises():
    provider = VECMForecastProvider()
    with pytest.raises(RuntimeError):
        provider.forecast(steps=3, exog=pd.DataFrame({"Promo": [0.0, 1.0, 0.0]}))


def test_fit_then_forecast_returns_expected_shape():
    endog, exog = _synthetic_data()
    provider = VECMForecastProvider(k_ar_diff=1, coint_rank=1)
    provider.fit(endog, exog)

    future_exog = pd.DataFrame({"Promo": [0.0, 1.0, 0.0, 1.0, 0.0]})
    result = provider.forecast(steps=5, exog=future_exog)

    assert len(result) == 5
    assert list(result.columns) == ["Sales", "Customers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vecm_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.providers.vecm_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/vecm_provider.py
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from src.core.interfaces import ForecastProvider


class VECMForecastProvider(ForecastProvider):
    def __init__(self, k_ar_diff: int = 1, coint_rank: int = 1) -> None:
        self._k_ar_diff = k_ar_diff
        self._coint_rank = coint_rank
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VECM(endog, exog=exog, k_ar_diff=self._k_ar_diff, coint_rank=self._coint_rank)
        self._result = model.fit()

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VECMForecastProvider.forecast() called before fit()")
        values = self._result.predict(steps=steps, exog_fc=exog.to_numpy())
        return pd.DataFrame(values, columns=self._endog_columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vecm_provider.py -v`
Expected: PASS (2 passed). If `VECM(...).fit()` raises on the synthetic data (e.g. a rank/dimension error), increase `n` in `_synthetic_data` before changing provider code.

- [ ] **Step 5: Commit**

```bash
git add src/providers/vecm_provider.py tests/unit/test_vecm_provider.py
git commit -m "feat: add VECM forecast provider"
```

---

### Task 8: Forecast orchestration service

**Files:**
- Create: `src/services/forecast_service.py`
- Test: `tests/unit/test_forecast_service.py`

**Interfaces:**
- Consumes: `src.services.preprocessing.{winsorize_series, one_hot_day_of_week, seasonal_features}` (Task 3), `src.services.diagnostics.check_stationarity` (Task 4), `forge.eval.timeseries.{evaluate_forecast, TimeSeriesMetrics}`, `forge.viz.timeseries.{plot_forecast, plot_decomposition}` (both already exist in the-forge), `ForecastProvider` (Task 2).
- Produces: `StoreForecastResult` (attributes: `store_id: int`, `metrics: TimeSeriesMetrics`, `forecast_fig`, `decomposition_fig`), `run_store_forecast(store_id: int, store_df: pd.DataFrame, provider: ForecastProvider, test_size: int = 100) -> StoreForecastResult | None` (returns `None` on any failure, never raises), `run_forecast_pipeline(data: pd.DataFrame, provider_factory: Callable[[], ForecastProvider], store_ids: list[int], test_size: int = 100) -> list[StoreForecastResult]`. Consumed by `src/app.py` (Task 11) and `narrative_service.py`/`report_service.py` (Tasks 9-10, via `StoreForecastResult`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forecast_service.py
import numpy as np
import pandas as pd
import pytest

from src.core.interfaces import ForecastProvider
from src.services.forecast_service import run_forecast_pipeline, run_store_forecast


class _StubProvider(ForecastProvider):
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"Sales": [100.0] * steps, "Customers": [10.0] * steps})


class _FailingProvider(ForecastProvider):
    def fit(self, endog, exog):
        raise ValueError("synthetic fit failure")

    def forecast(self, steps, exog):
        raise RuntimeError("should never be called")


def _make_store_df(n: int = 40, store_id: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "Store": store_id,
            "Sales": 100 + np.arange(n) + rng.normal(0, 1, n),
            "Customers": 20 + np.arange(n) * 0.2 + rng.normal(0, 0.5, n),
            "Promo": rng.integers(0, 2, n).astype(float),
            "SchoolHoliday": 0.0,
            "CompetitionDistance": 500.0,
            "DayOfWeek": [i % 7 for i in range(n)],
        }
    )


def test_run_store_forecast_returns_result_on_success():
    store_df = _make_store_df()
    result = run_store_forecast(store_id=1, store_df=store_df, provider=_StubProvider(), test_size=5)
    assert result is not None
    assert result.store_id == 1
    assert result.metrics is not None
    assert result.forecast_fig is not None
    assert result.decomposition_fig is not None


def test_run_store_forecast_returns_none_on_provider_failure():
    store_df = _make_store_df()
    result = run_store_forecast(store_id=1, store_df=store_df, provider=_FailingProvider(), test_size=5)
    assert result is None


def test_run_forecast_pipeline_isolates_failures():
    data = pd.concat([_make_store_df(store_id=1), _make_store_df(store_id=2)], ignore_index=True)

    def provider_factory():
        return _StubProvider()

    results = run_forecast_pipeline(data, provider_factory, store_ids=[1, 2], test_size=5)
    assert len(results) == 2
    assert {r.store_id for r in results} == {1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forecast_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.forecast_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/forecast_service.py
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from forge.eval.timeseries import TimeSeriesMetrics, evaluate_forecast
from forge.viz.timeseries import plot_decomposition, plot_forecast
from src.core.interfaces import ForecastProvider
from src.services import diagnostics, preprocessing

logger = logging.getLogger(__name__)


class StoreForecastResult:
    def __init__(self, store_id: int, metrics: TimeSeriesMetrics, forecast_fig, decomposition_fig) -> None:
        self.store_id = store_id
        self.metrics = metrics
        self.forecast_fig = forecast_fig
        self.decomposition_fig = decomposition_fig


def run_store_forecast(
    store_id: int,
    store_df: pd.DataFrame,
    provider: ForecastProvider,
    test_size: int = 100,
) -> StoreForecastResult | None:
    try:
        store_df = store_df.copy()
        store_df["Sales"] = preprocessing.winsorize_series(store_df["Sales"])
        store_df["Customers"] = preprocessing.winsorize_series(store_df["Customers"])

        sales_features = preprocessing.seasonal_features(store_df["Sales"], prefix="Sales")
        cust_features = preprocessing.seasonal_features(store_df["Customers"], prefix="Cust")
        store_df = pd.concat([store_df, sales_features, cust_features], axis=1)
        store_df = preprocessing.one_hot_day_of_week(store_df)

        train_df, test_df = store_df.iloc[:-test_size], store_df.iloc[-test_size:]
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"test_size={test_size} leaves no train or test rows for store {store_id}")

        diagnostics.check_stationarity(train_df["Sales"].dropna())

        endog_cols = ["Sales", "Customers"]
        exog_cols = [
            "Promo",
            "SchoolHoliday",
            "CompetitionDistance",
            "SalesSeasonality",
            "SalesTrend",
            "CustSeasonality",
            "CustTrend",
        ]
        train_endog = train_df[endog_cols].astype("float32").dropna()
        train_exog = train_df.loc[train_endog.index, exog_cols].astype("float32")

        provider.fit(train_endog, train_exog)

        test_exog = test_df[exog_cols].astype("float32")
        forecast_df = provider.forecast(steps=len(test_exog), exog=test_exog)

        actual = test_df["Sales"].to_numpy()[: len(forecast_df)]
        predicted = forecast_df["Sales"].to_numpy()
        metrics = evaluate_forecast(actual, predicted)

        forecast_fig = plot_forecast(actual, predicted, title=f"Store {store_id} — Sales Forecast")
        decomposition_fig = plot_decomposition(
            store_df["Sales"].dropna().to_numpy(), period=7, title=f"Store {store_id} — Decomposition"
        )

        return StoreForecastResult(store_id, metrics, forecast_fig, decomposition_fig)
    except Exception:
        logger.exception("Forecast failed for store %s", store_id)
        return None


def run_forecast_pipeline(
    data: pd.DataFrame,
    provider_factory: Callable[[], ForecastProvider],
    store_ids: list[int],
    test_size: int = 100,
) -> list[StoreForecastResult]:
    results: list[StoreForecastResult] = []
    for store_id in store_ids:
        store_df = data.loc[data["Store"] == store_id].reset_index(drop=True)
        provider = provider_factory()
        result = run_store_forecast(store_id, store_df, provider, test_size=test_size)
        if result is not None:
            results.append(result)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forecast_service.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/forecast_service.py tests/unit/test_forecast_service.py
git commit -m "feat: add forecast orchestration service"
```

---

### Task 9: Narrative service

**Files:**
- Create: `src/services/narrative_service.py`
- Test: `tests/unit/test_narrative_service.py`

**Interfaces:**
- Consumes: `forge.llm.base.LLMProvider` (abstract, already exists — `.generate(prompt, system=None, **kwargs) -> LLMResponse`), `forge.llm.prompt.PromptTemplate` (already exists), `StoreForecastResult` (Task 8, uses only `.store_id` and `.metrics`, where `.metrics` is a `TimeSeriesMetrics` with `.rmse`, `.mae`, `.smape` floats).
- Produces: `NarrativeResult` (attributes: `trend_narrative: str`, `anomaly_narrative: str | None`), `generate_narrative(result: StoreForecastResult, llm: LLMProvider, anomaly_threshold: float = 20.0) -> NarrativeResult`. Consumed by `src/app.py` (Task 11) and `report_service.py` (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_narrative_service.py
from forge.eval.timeseries import TimeSeriesMetrics
from forge.llm.base import LLMResponse

from src.services.narrative_service import generate_narrative
from src.services.forecast_service import StoreForecastResult


class _StubLLM:
    def __init__(self, content: str = "stub narrative", raise_error: bool = False):
        self._content = content
        self._raise_error = raise_error
        self.prompts_seen: list[str] = []

    def generate(self, prompt, system=None, **kwargs):
        self.prompts_seen.append(prompt)
        if self._raise_error:
            raise RuntimeError("LLM unreachable")
        return LLMResponse(content=self._content, model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _make_result(smape: float, store_id: int = 1) -> StoreForecastResult:
    metrics = TimeSeriesMetrics(rmse=1.0, mae=1.0, mape=1.0, smape=smape)
    return StoreForecastResult(store_id=store_id, metrics=metrics, forecast_fig=None, decomposition_fig=None)


def test_trend_narrative_always_generated():
    result = _make_result(smape=5.0)
    llm = _StubLLM(content="trend text")
    narrative = generate_narrative(result, llm)
    assert narrative.trend_narrative == "trend text"


def test_anomaly_narrative_generated_above_threshold():
    result = _make_result(smape=25.0)
    llm = _StubLLM(content="narrative text")
    narrative = generate_narrative(result, llm, anomaly_threshold=20.0)
    assert narrative.anomaly_narrative is not None
    assert len(llm.prompts_seen) == 2


def test_anomaly_narrative_not_generated_below_threshold():
    result = _make_result(smape=10.0)
    llm = _StubLLM(content="narrative text")
    narrative = generate_narrative(result, llm, anomaly_threshold=20.0)
    assert narrative.anomaly_narrative is None
    assert len(llm.prompts_seen) == 1


def test_trend_narrative_falls_back_on_llm_failure():
    result = _make_result(smape=5.0)
    llm = _StubLLM(raise_error=True)
    narrative = generate_narrative(result, llm)
    assert narrative.trend_narrative == "narrative unavailable"
    assert narrative.anomaly_narrative is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_narrative_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.narrative_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/narrative_service.py
from __future__ import annotations

import logging

from forge.llm.base import LLMProvider
from forge.llm.prompt import PromptTemplate

from src.services.forecast_service import StoreForecastResult

logger = logging.getLogger(__name__)

TREND_PROMPT = PromptTemplate(
    "Store {store_id} sales forecast summary. RMSE={rmse:.2f}, MAE={mae:.2f}, "
    "SMAPE={smape:.2f}%. Write a 2-3 sentence plain-English summary of the "
    "sales trend and seasonality for a store manager."
)

ANOMALY_PROMPT = PromptTemplate(
    "Store {store_id}'s forecast SMAPE is {smape:.2f}%, above the {threshold:.0f}% "
    "anomaly threshold. Candidate drivers available: Promo, SchoolHoliday, "
    "CompetitionDistance. In 2-3 sentences, hypothesize which of these likely "
    "explains the forecast miss."
)


class NarrativeResult:
    def __init__(self, trend_narrative: str, anomaly_narrative: str | None) -> None:
        self.trend_narrative = trend_narrative
        self.anomaly_narrative = anomaly_narrative


def generate_narrative(
    result: StoreForecastResult,
    llm: LLMProvider,
    anomaly_threshold: float = 20.0,
) -> NarrativeResult:
    try:
        trend_prompt = TREND_PROMPT.render(
            store_id=result.store_id,
            rmse=result.metrics.rmse,
            mae=result.metrics.mae,
            smape=result.metrics.smape,
        )
        trend_narrative = llm.generate(trend_prompt).content
    except Exception:
        logger.exception("Narrative generation failed for store %s", result.store_id)
        return NarrativeResult(trend_narrative="narrative unavailable", anomaly_narrative=None)

    anomaly_narrative = None
    if result.metrics.smape > anomaly_threshold:
        try:
            anomaly_prompt = ANOMALY_PROMPT.render(
                store_id=result.store_id,
                smape=result.metrics.smape,
                threshold=anomaly_threshold,
            )
            anomaly_narrative = llm.generate(anomaly_prompt).content
        except Exception:
            logger.exception("Anomaly narrative failed for store %s", result.store_id)
            anomaly_narrative = "anomaly explanation unavailable"

    return NarrativeResult(trend_narrative, anomaly_narrative)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_narrative_service.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/narrative_service.py tests/unit/test_narrative_service.py
git commit -m "feat: add narrative service with SMAPE-threshold anomaly gating"
```

---

### Task 10: Report service

**Files:**
- Create: `src/services/report_service.py`
- Test: `tests/unit/test_report_service.py`

**Interfaces:**
- Consumes: `forge.report.builder.{ReportBuilder, ReportSection}` (already exist — `ReportBuilder(title: str, subtitle: str = "")`, `.add_section(ReportSection) -> ReportBuilder`, `.save(path) -> Path`; `ReportSection(title: str, content: str = "", figures: list[Figure] = [], metrics: dict = {})`), `StoreForecastResult` (Task 8), `NarrativeResult` (Task 9).
- Produces: `build_report(title: str, results: list[StoreForecastResult], narratives: dict[int, NarrativeResult]) -> ReportBuilder`. Consumed by `src/app.py` (Task 11).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_report_service.py
import matplotlib.pyplot as plt

from forge.eval.timeseries import TimeSeriesMetrics

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import NarrativeResult
from src.services.report_service import build_report


def _make_result(store_id: int) -> StoreForecastResult:
    metrics = TimeSeriesMetrics(rmse=1.0, mae=1.0, mape=1.0, smape=5.0)
    fig, _ = plt.subplots()
    return StoreForecastResult(store_id=store_id, metrics=metrics, forecast_fig=fig, decomposition_fig=fig)


def test_build_report_includes_each_store_section():
    results = [_make_result(1), _make_result(2)]
    narratives = {
        1: NarrativeResult(trend_narrative="Store 1 is trending up.", anomaly_narrative=None),
        2: NarrativeResult(trend_narrative="Store 2 is stable.", anomaly_narrative="Promo effect suspected."),
    }
    report = build_report("Test Report", results, narratives)
    html = report.to_html()

    assert "Store 1" in html
    assert "Store 2" in html
    assert "trending up" in html
    assert "Promo effect suspected" in html


def test_build_report_handles_missing_narrative():
    results = [_make_result(3)]
    report = build_report("Test Report", results, narratives={})
    html = report.to_html()
    assert "Store 3" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_report_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.report_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/report_service.py
from __future__ import annotations

from forge.report.builder import ReportBuilder, ReportSection

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import NarrativeResult


def build_report(
    title: str,
    results: list[StoreForecastResult],
    narratives: dict[int, NarrativeResult],
) -> ReportBuilder:
    report = ReportBuilder(title)
    for result in results:
        narrative = narratives.get(result.store_id)
        content_parts = []
        if narrative is not None:
            content_parts.append(narrative.trend_narrative)
            if narrative.anomaly_narrative:
                content_parts.append(narrative.anomaly_narrative)
        content = " ".join(content_parts)

        section = ReportSection(
            title=f"Store {result.store_id}",
            content=content,
            metrics=result.metrics.to_dict(),
            figures=[result.forecast_fig, result.decomposition_fig],
        )
        report.add_section(section)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_report_service.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/report_service.py tests/unit/test_report_service.py
git commit -m "feat: add report assembly service"
```

---

### Task 11: Data fetch script + CLI driver + integration test

**Files:**
- Create: `scripts/fetch_data.py`
- Create: `src/app.py`
- Test: `tests/unit/test_fetch_data.py`, `tests/integration/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 2-10 — `src.providers.{var_provider.VARForecastProvider, varmax_provider.VARMAXForecastProvider, vecm_provider.VECMForecastProvider}`, `src.services.forecast_service.run_forecast_pipeline`, `src.services.narrative_service.generate_narrative`, `src.services.report_service.build_report`, plus `forge.llm.{ollama.OllamaProvider, claude.ClaudeProvider, openai_provider.OpenAIProvider}` (all three already exist, all three constructible with no required arguments).
- Produces: `src/app.py`'s `main(argv: list[str] | None = None) -> int`, module-level dicts `PROVIDERS: dict[str, type]` and `LLM_PROVIDERS: dict[str, type]` (deliberately mutable/patchable module attributes — the integration test monkeypatches them to avoid real model fitting and real LLM calls).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_fetch_data.py
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import fetch_data


def test_fetch_fails_cleanly_without_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fetch_data, "KAGGLE_CREDENTIALS", tmp_path / "nope.json")
    with pytest.raises(SystemExit) as exc_info:
        fetch_data.fetch_rossmann_data()
    assert exc_info.value.code == 1
    assert "Kaggle API credentials not found" in capsys.readouterr().err


def test_fetch_downloads_and_extracts(monkeypatch, tmp_path):
    creds = tmp_path / "kaggle.json"
    creds.write_text("{}")
    monkeypatch.setattr(fetch_data, "KAGGLE_CREDENTIALS", creds)
    monkeypatch.setattr(fetch_data, "DATA_DIR", tmp_path / "data")

    def _fake_run(cmd, check):
        zip_path = (tmp_path / "data") / "rossmann-store-sales.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "Store,Sales\n1,100\n")

    with patch("subprocess.run", side_effect=_fake_run):
        result_dir = fetch_data.fetch_rossmann_data()

    assert (result_dir / "train.csv").is_file()
    assert not (result_dir / "rossmann-store-sales.zip").exists()
```

```python
# tests/integration/test_pipeline.py
import pandas as pd

from forge.llm.base import LLMResponse

from src import app


class _StubForecastProvider:
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"Sales": [100.0] * steps, "Customers": [10.0] * steps})


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        return LLMResponse(content="stub narrative", model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _make_synthetic_csv(tmp_path):
    rows = []
    for store in [1, 2]:
        for day in range(40):
            rows.append(
                {
                    "Store": store,
                    "Sales": 100 + day + store,
                    "Customers": 10 + day,
                    "Promo": day % 2,
                    "SchoolHoliday": 0,
                    "CompetitionDistance": 500.0,
                    "DayOfWeek": day % 7,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_pipeline_writes_report(tmp_path, monkeypatch):
    monkeypatch.setitem(app.PROVIDERS, "varmax", _StubForecastProvider)
    monkeypatch.setitem(app.LLM_PROVIDERS, "ollama", _StubLLM)

    data_path = _make_synthetic_csv(tmp_path)
    output_path = tmp_path / "report.html"

    exit_code = app.main(
        [
            "--data",
            str(data_path),
            "--top-n",
            "2",
            "--provider",
            "varmax",
            "--llm-provider",
            "ollama",
            "--test-size",
            "5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    html = output_path.read_text()
    assert "Store 1" in html or "Store 2" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_fetch_data.py tests/integration/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.fetch_data'` (and, once that's created, `No module named 'src.app'`)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fetch_data.py
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"
KAGGLE_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"


def fetch_rossmann_data() -> Path:
    if not KAGGLE_CREDENTIALS.exists():
        print(
            f"error: Kaggle API credentials not found at {KAGGLE_CREDENTIALS}.\n"
            "Set up your Kaggle API token: https://www.kaggle.com/docs/api#authentication",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "rossmann-store-sales", "-p", str(DATA_DIR)],
        check=True,
    )

    zip_path = DATA_DIR / "rossmann-store-sales.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    return DATA_DIR


if __name__ == "__main__":
    fetch_rossmann_data()
```

Create `scripts/__init__.py` (empty) if it doesn't already exist, so `from scripts import fetch_data` works in tests.

```python
# src/app.py
from __future__ import annotations

import argparse
import logging

import pandas as pd

from forge.llm.claude import ClaudeProvider
from forge.llm.ollama import OllamaProvider
from forge.llm.openai_provider import OpenAIProvider
from src.providers.var_provider import VARForecastProvider
from src.providers.varmax_provider import VARMAXForecastProvider
from src.providers.vecm_provider import VECMForecastProvider
from src.services.forecast_service import run_forecast_pipeline
from src.services.narrative_service import generate_narrative
from src.services.report_service import build_report

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, type] = {
    "var": VARForecastProvider,
    "varmax": VARMAXForecastProvider,
    "vecm": VECMForecastProvider,
}

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def top_n_stores(data: pd.DataFrame, n: int) -> list[int]:
    volumes = data.groupby("Store")["Sales"].sum().sort_values(ascending=False)
    return volumes.head(n).index.tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trend-whisperer")
    parser.add_argument("--data", default="src/data/train.csv")
    parser.add_argument("--stores", default=None, help="comma-separated store IDs; default top-N by volume")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="varmax")
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--output", default="reports/model_report.html")
    args = parser.parse_args(argv)

    data = pd.read_csv(args.data)
    store_ids = (
        [int(s) for s in args.stores.split(",")] if args.stores else top_n_stores(data, args.top_n)
    )

    provider_cls = PROVIDERS[args.provider]
    results = run_forecast_pipeline(data, provider_cls, store_ids, test_size=args.test_size)

    llm = LLM_PROVIDERS[args.llm_provider]()
    narratives = {r.store_id: generate_narrative(r, llm) for r in results}

    report = build_report("TrendWhisperer — Sales Forecast Report", results, narratives)
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_fetch_data.py tests/integration/test_pipeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests across every task pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_data.py scripts/__init__.py src/app.py tests/unit/test_fetch_data.py tests/integration/test_pipeline.py
git commit -m "feat: add data fetch script and CLI driver"
```

---

### Task 12: README and task.md

**Files:**
- Modify: `README.md`, `task.md`, `pyproject.toml` (fill in the empty `description` field)

**Interfaces:**
- Consumes: nothing new — documents Tasks 1-11.

- [ ] **Step 1: Get the actual test count**

Run: `pytest tests/ --collect-only -q | tail -1`

- [ ] **Step 2: Rewrite `README.md`**

Replace the scaffold's placeholder content with:

```markdown
# TrendWhisperer

![Version](https://img.shields.io/badge/version-0.1.0-blue)

Per-store sales forecasting (VAR/VARMAX/VECM) over the Kaggle Rossmann Store Sales dataset,
built on [sameer-forge](https://github.com/mauryasameer/the-forge). Adds a GenAI narrative and
anomaly-explanation layer via `forge.llm`, and renders a self-contained HTML report via
`forge.report`.

## Setup

```bash
pip install -r requirements.txt
```

Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) per
[Kaggle's API docs](https://www.kaggle.com/docs/api#authentication), then:

```bash
python scripts/fetch_data.py
```

## Usage

```bash
python -m src.app --provider varmax --llm-provider ollama --top-n 5
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--data` | `src/data/train.csv` | Path to the Rossmann training CSV |
| `--stores` | none (uses `--top-n`) | Comma-separated store IDs to forecast |
| `--top-n` | `5` | Number of top-by-volume stores to forecast if `--stores` is unset |
| `--provider` | `varmax` | Forecasting backend: `var` / `varmax` / `vecm` |
| `--llm-provider` | `ollama` | Narrative backend: `ollama` / `claude` / `openai` |
| `--test-size` | `100` | Holdout periods per store for evaluation |
| `--output` | `reports/model_report.html` | Report output path |

Output is a single HTML report with one section per forecasted store: metrics
(RMSE/MAE/MAPE/SMAPE), forecast and decomposition plots, an LLM-generated trend narrative, and
(when that store's SMAPE exceeds 20%) an anomaly-driver explanation.

## Architecture

- `src/core/interfaces.py` — `ForecastProvider` abstract interface
- `src/providers/` — `VARForecastProvider`, `VARMAXForecastProvider`, `VECMForecastProvider`
- `src/services/preprocessing.py` — winsorizing, day-of-week encoding, seasonal feature extraction
- `src/services/diagnostics.py` — stationarity, Granger causality, cointegration rank
- `src/services/forecast_service.py` — per-store orchestration, failure-isolated
- `src/services/narrative_service.py` — LLM trend/anomaly narrative generation
- `src/services/report_service.py` — HTML report assembly
- `scripts/fetch_data.py` — Kaggle dataset fetch

## Testing

```bash
pytest tests/ -v
```

<!-- test count filled in from Step 1's actual pytest output -->
```

(Replace the HTML comment placeholder with the real count as a sentence, e.g. "58 tests, zero real network/LLM calls." — use the number from Step 1, not a guess.)

- [ ] **Step 3: Update `task.md`**

Replace the empty `## Backlog` section with:

```markdown
## Backlog

- [x] Core `ForecastProvider` interface
- [x] Preprocessing service
- [x] Diagnostics service
- [x] VAR / VARMAX / VECM providers
- [x] Forecast orchestration service
- [x] Narrative service (trend + anomaly, SMAPE-gated)
- [x] Report service
- [x] Data fetch script + CLI driver
- [ ] Real end-to-end run against the actual Kaggle dataset (requires Kaggle credentials, not exercised in CI)
- [ ] Real LLM narrative review against actual Ollama output (stubbed in tests)
```

- [ ] **Step 4: Fill in `pyproject.toml`'s description**

Change:
```toml
description = ""
```
to:
```toml
description = "Per-store sales forecasting (VAR/VARMAX/VECM) with GenAI narrative generation, built on sameer-forge."
```

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/ -v` (confirm nothing broke) and re-read the edited `README.md` once to confirm no broken code fences or headers.

```bash
git add README.md task.md pyproject.toml
git commit -m "docs: document TrendWhisperer usage and architecture"
```

---

## Self-Review Notes

- Spec coverage: `ForecastProvider` interface (Task 2), swappable VAR/VARMAX/VECM providers (Tasks 5-7), preprocessing/diagnostics extracted from the notebook (Tasks 3-4), forecast orchestration with per-store failure isolation (Task 8), narrative + SMAPE-gated (20%, configurable) anomaly explanation (Task 9), report assembly via `forge.report` (Task 10), Kaggle fetch script with loud failure on missing credentials (Task 11), CLI driver with all four flags from the design (`--stores`/`--provider`/`--llm-provider` plus `--test-size` added during planning to make the design's error-handling and testing sections actually testable) (Task 11), README/task.md consistency (Task 12) — all covered. Repo genesis, GitHub remote creation, and dev/main branch setup (Task 1) covers the "own GitHub repo" architecture decision.
- Type consistency checked: `StoreForecastResult` (Task 8) fields (`store_id`, `metrics`, `forecast_fig`, `decomposition_fig`) match usage in `narrative_service.py` (Task 9, uses `.store_id`/`.metrics`) and `report_service.py` (Task 10, uses all four). `NarrativeResult` fields (`trend_narrative`, `anomaly_narrative`) match `report_service.py`'s usage. `ForecastProvider.fit`/`forecast` signatures match across the interface (Task 2) and all three providers (Tasks 5-7) and `forecast_service.py`'s calls (Task 8). `PROVIDERS`/`LLM_PROVIDERS` dict keys (`var`/`varmax`/`vecm`, `ollama`/`claude`/`openai`) match argparse `choices` and the integration test's monkeypatch targets (Task 11).
- Fixed during planning: the design mentioned a `--test-size` implicitly (via `forecast_service.py`'s `test_size` parameter) but Task 11's original CLI sketch didn't expose it as a flag — added `--test-size` explicitly so the integration test (40 synthetic rows/store) can pass a small value instead of the production default of 100, which would leave zero training rows on synthetic test data.
