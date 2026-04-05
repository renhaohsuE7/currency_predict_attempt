# uv Package Manager (強制)

**本專案一律使用 `uv`，禁止 `pip`、`pip install`、`python -m pip`、`uv pip install`。**

## 開發環境

本專案使用 **docker compose + Dockerfile + uv** 作為開發環境。所有指令在容器內執行：

```bash
# 進入容器
docker exec -it <container> bash

# 在容器內使用 uv
uv add <package>
uv run pytest
```

## Dependency 管理

`pyproject.toml` 的 `[project.dependencies]`、`[project.optional-dependencies]` 和 `[dependency-groups]` **不可直接手動編輯**，
一律透過 uv 指令操作，讓 uv 同時更新 `pyproject.toml` 和 `uv.lock`：

```bash
# 新增 runtime dependency
uv add pandas

# 新增 optional dependency (到指定 extra group)
uv add --optional interactive plotly
uv add --optional pdf fpdf2
uv add --optional notebook nbstripout

# 新增 dev dependency
uv add --dev pytest

# 移除
uv remove <package>

# 安裝所有（含 optional）
uv sync --all-extras --all-groups
```

**絕對禁止：**
- `uv pip install <package>` — 只裝進 venv，不更新 uv.lock
- 手動編輯 pyproject.toml 的 dependency 區塊 — uv.lock 不會同步

例外：`[tool.pytest.ini_options]`、`[tool.black]`、`[tool.mypy]` 等工具設定區塊可直接編輯，因為這些不是 dependency。

## 常用指令

| 動作 | 指令 |
| --- | --- |
| 安裝所有依賴 | `uv sync --all-extras --all-groups` |
| 新增 runtime dep | `uv add <package>` |
| 新增 optional dep | `uv add --optional <group> <package>` |
| 新增 dev dep | `uv add --dev <package>` |
| 移除 dep | `uv remove <package>` |
| 驗證 lockfile | `uv lock --check` |
| 執行指令 | `uv run <cmd>` |
| 執行 Python | `uv run python` |
