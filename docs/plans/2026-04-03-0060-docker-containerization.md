# Docker Containerization with uv + CUDA GPU

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: project infrastructure
- **前置依賴**: plan #0050 (pretrained + fine-tune) ✓ completed

## 目標

Docker 容器化，支援 multi-stage build (dev + prod)、CUDA GPU、HuggingFace 模型快取存在專案目錄下。

## 背景

目前專案只能透過本地 `uv sync` + `uv run` 執行。HuggingFace 預訓練模型下載到 `~/.cache/huggingface/`，散落在使用者 home 目錄，不易管理。

需要：

1. Docker 容器化，確保環境可重現
2. Multi-stage build：dev image (含 pytest, black, jupyter) + prod image (最小 runtime)
3. CUDA GPU 支援
4. Docker volume 讓 HF 模型存在專案目錄下 `.cache/huggingface/`

## 關鍵發現

- `uv.lock` 中的 torch 2.8.0 PyPI wheel **已內含 CUDA 12.x** (nvidia-cublas-cu12, nvidia-cudnn-cu12 等)，不需要 `--extra-index-url`
- `HF_HOME` 環境變數控制 HuggingFace 快取位置，程式碼不需改動
- Pydantic Settings 支援 `CURRENCY_PRED_` 前綴的環境變數 + `.env` 檔案

## 設計決策

### Q1: Base image 選擇？

**最終採用 `python:3.11-slim`**（Debian trixie）。

原計畫使用 `nvidia/cuda:12.8.1-runtime-ubuntu22.04`，但建置時遭遇 Ubuntu/Debian apt mirror 持續性 Hash Sum mismatch 錯誤。由於：

- PyPI torch 2.8.0 wheel **已自帶 CUDA 12.x** 函式庫（nvidia-cublas-cu12, nvidia-cudnn-cu12 等）
- NVIDIA Container Toolkit 在 runtime 注入 `libcuda.so`
- 不需要 CUDA toolkit 或 nvcc

因此 `python:3.11-slim` + NVIDIA Container Toolkit 即可完整使用 GPU，且 image 更小、build 更穩定。

### Q2: uv 如何安裝到 Docker？

**`COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/`** — Astral 官方推薦方式。

### Q3: torch CUDA 版本處理？

**不需特殊處理**。`uv.lock` 已鎖定 torch 2.8.0 + nvidia-* CUDA 12.x wheels，`uv sync --frozen` 直接安裝。

### Q4: Volume mount 策略？

使用 bind mounts（非 named volumes），檔案可在 host 直接檢視/編輯。

## 修改範圍

| File | Action | 說明 |
| --- | --- | --- |
| `Dockerfile` | 新增 | Multi-stage: base (CUDA + Python + uv) → prod → dev |
| `docker-compose.yml` | 新增 | prod / dev / jupyter 三個 service |
| `.dockerignore` | 新增 | 排除 .venv, data/, models/, .cache/ 等 |
| `.env.example` | 新增 | 記錄可用環境變數 |
| `.gitignore` | 修改 | 新增 .cache/, .env, logs/ 等 |
| `README.md` | 修改 | 新增 Docker 使用說明 |

**計畫外額外修改**：`models/patchtst/huggingface/model.py`（見執行結果）

## 實作步驟

### Step 1: `.dockerignore`

排除不需 build 進 image 的檔案（volume mount 的目錄、.venv、.git 等）。

### Step 2: `Dockerfile` — Multi-Stage Build

三個 stage：

**Stage 1: `base`** — 共用基礎層

- Base image: `python:3.11-slim`（見 Q1 設計決策）
- apt retry 設定避免 CDN hash mismatch
- 安裝 uv: `COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/`
- 設定環境變數：`HF_HOME`, `TORCH_HOME`, `NVIDIA_VISIBLE_DEVICES`, `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`
- COPY `pyproject.toml`, `uv.lock`, `.python-version` (layer cache)

**Stage 2: `prod`** — 最小生產 image

- `uv sync --frozen --no-dev --no-install-project`
- COPY `src/`, `main.py`
- `uv sync --frozen --no-dev`
- CMD: `uv run main.py`

**Stage 3: `dev`** — 完整開發 image

- `uv sync --frozen --all-extras --all-groups --no-install-project`
- COPY `src/`, `main.py`, `tests/`, `examples/`, `notebooks/`
- EXPOSE 8888
- CMD: `uv run pytest -v`

### Step 3: `docker-compose.yml`

三個 service：prod, dev, jupyter

Volume mount 策略：

| Host | Container | 用途 |
| --- | --- | --- |
| `./data` | `/app/data` | 匯率資料 |
| `./models` | `/app/models` | 訓練後模型 |
| `./results` | `/app/results` | 預測結果 |
| `./logs` | `/app/logs` | 日誌 |
| `./.cache/huggingface` | `/app/.cache/huggingface` | HF 預訓練模型快取 |
| `./config.json` | `/app/config.json:ro` | 設定檔 (唯讀) |

GPU: `deploy.resources.reservations.devices` (nvidia, count: 1)

### Step 4: `.env.example` + `.gitignore` 更新 + README Docker 區塊

## 注意事項

- **Image 大小**: torch CUDA wheel ~888MB，prod image 約 8-10GB，無法避免
- **GPU 可選**: 沒有 NVIDIA GPU 的機器可用 `docker-compose.override.yml` 移除 deploy 區塊，程式碼已有 `cuda if available else cpu` fallback
- **首次執行**: 需下載 HF 模型，之後快取在 `.cache/huggingface/` volume 中
- **向後相容**: 完全 additive，既有 `uv run` 工作流不受影響

## 驗證

```bash
# Build
docker compose build prod dev

# 測試 prod
docker compose run --rm prod

# 測試 dev
docker compose run --rm dev uv run pytest -v

# HF cache 確認
ls .cache/huggingface/

# GPU 確認 (有 GPU 的環境)
docker compose run --rm dev uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 執行結果（2026-04-03）

### 建置過程

1. 首次嘗試 `nvidia/cuda:12.8.1-runtime-ubuntu22.04`：Ubuntu apt mirror Hash Sum mismatch，失敗
2. 切換至 `python:3.11-slim`：Debian apt mirror 同樣 Hash Sum mismatch，失敗
3. 加入 apt retry 設定（`Acquire::Retries "3"`, `Acquire::http::Pipeline-Depth "0"`）：build 成功
4. `uv sync --all-extras --all-groups` 缺少 `--no-install-project`：setuptools 找不到 `src/` 目錄，失敗
5. 加入 `--no-install-project` 後：dev/prod image 均 build 成功

### Bug fix: pretrained model context_length mismatch

Docker 內執行 258 tests 時發現 `test_full_finetune_fit_predict` 失敗。

**原因**：`fit()` 中 `_prepare_data()` 先於 `_setup_model()` 執行，使用預設 `context_length=64` 建立序列，但預訓練模型（granite-timeseries-patchtst）的 `context_length=512`。

**修復**：

- `fit()` 中將 `_setup_model()` 移至 `_prepare_data()` 之前
- 新增 `_infer_num_features()` helper，從輸入資料推斷特徵數量（不建立序列）

修改檔案：`src/currency_predictor/models/patchtst/huggingface/model.py`

### 驗證結果

| 項目 | 結果 |
| --- | --- |
| dev image build | 成功 |
| prod image build | 成功 |
| `pytest -v` (non-slow) | **258 passed, 0 failed** |
| `pytest -m slow` (pretrained integration) | **6/6 passed** |
| `nvidia-smi` | NVIDIA GeForce RTX 3090, Driver 560.35.03 |
| `torch.cuda.is_available()` | `True` |
| `torch.version.cuda` | 12.8 (PyPI wheel bundled) |
| HF cache on host | `.cache/huggingface/hub/models--ibm-granite--granite-timeseries-patchtst` |

## 完成標準

- [x] `docker compose build prod dev` 成功
- [x] `docker compose run --rm dev uv run pytest -v` — 258 passed
- [x] `docker compose run --rm dev uv run pytest -m slow` — 6 passed（HF pretrained 下載 + fine-tune）
- [x] `.cache/huggingface/` 在 host 存在且有模型快取
- [x] GPU 可用（nvidia-smi + torch.cuda.is_available()）
- [x] 既有 `uv run` 工作流不受影響（向後相容）
