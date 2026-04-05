# Docker Non-Root User — Fix Container File Ownership

- **Date**: 2026-04-05 17:01
- **Status**: completed
- **Module**: Dockerfile, docker-compose.yml

## 目標

修正 Docker 容器以 root 執行導致 volume-mounted 目錄（`results/`, `data/`, `models/`）內的檔案都是 root 所有，在 host 上執行 `./scripts/clean.sh` 或直接操作這些檔案時遇到 `Permission denied` 的問題。

## 問題

Host user: UID=1006, GID=1007 (`renhaohsu`)

容器以 root (UID 0) 執行 → 透過 bind mount 寫入的檔案在 host 上屬於 root → host user 無法刪除/修改這些檔案。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `Dockerfile` | `HOME=/app`、`chmod -R 777` writable dirs、`UV_NO_SYNC` (prod)、`UV_CACHE_DIR`、targeted chmod for editable install (dev) |
| `docker-compose.yml` | `user: "${APP_UID:-1000}:${APP_GID:-1000}"`、`.cache/matplotlib` volume mount |
| `.env` (new, git-ignored) | `APP_UID=1006`, `APP_GID=1007` — one-time setup |

## 方案選擇

評估了三種方案：

| 方案 | 優點 | 缺點 |
|------|------|------|
| Build-time `ARG UID` + `chown -R /app` | 容器內有真實 user | `chown -R` 耗時 322s+，需 per-user rebuild |
| **Runtime `user:` + `chmod 777`** | 快速 build、portable、不需 rebuild | writable dirs 是 777（可接受，因為是 container 內部） |
| `docker exec` 進容器操作 | 不改 Dockerfile | 不解決根本問題 |

選擇方案 2。

## 實作步驟

### Step 1: Dockerfile — base stage

- `HOME=/app`：非 root user 沒有 home dir，需要明確設定（否則 uv 無法建立 cache）
- `chmod -R 777` on `data`, `models`, `results`, `logs`, `.cache`：允許任何 UID 寫入

### Step 2: Dockerfile — prod stage

- `UV_NO_SYNC=true`：跳過 runtime sync（env 在 build 時已完整安裝，避免 Permission denied）
- `UV_CACHE_DIR=/tmp/uv-cache`：runtime uv cache 使用可寫的 `/tmp`（避免讀寫 build-time 的 root-owned cache）

### Step 3: Dockerfile — dev stage

- Targeted chmod on editable install files：
  ```dockerfile
  RUN chmod 777 /app/.venv/lib/python3.11/site-packages/ \
      && chmod -R 777 /app/.venv/lib/python3.11/site-packages/__editable__* \
                       /app/.venv/lib/python3.11/site-packages/currency_predict_attempt* \
                       /app/src/currency_predict_attempt.egg-info
  ```
- `UV_CACHE_DIR=/tmp/uv-cache`：同 prod
- 不設 `UV_NO_SYNC`：dev 需要 sync 以支援 editable install 更新

### Step 4: docker-compose.yml

- `user: "${APP_UID:-1000}:${APP_GID:-1000}"`：runtime 使用 host UID/GID
- 新增 `.cache/matplotlib` volume mount
- 移除 Jupyter 的 `--allow-root`（不再以 root 執行）

### Step 5: .env

One-time setup（git-ignored）：
```bash
echo -e "APP_UID=$(id -u)\nAPP_GID=$(id -g)" > .env
```

Docker Compose 自動讀取 `.env`，不需要在命令前加 prefix。

## 遇到的問題與解決

| 問題 | 解決方式 |
|------|----------|
| `uv` 無法建立 `/.cache/uv`（無 home dir） | `HOME=/app` |
| `chown -R /app` 耗時 322s（.venv 檔案太多） | 改用 `chmod 777` on writable dirs |
| `chmod -R 777 /app/.cache` 耗時 442s（uv cache 檔案太多） | `UV_CACHE_DIR=/tmp/uv-cache` 避開 build-time cache |
| `uv run` 嘗試 sync 並安裝 dev deps → Permission denied | Prod: `UV_NO_SYNC=true`；Dev: targeted chmod |
| `UID` 是 bash readonly variable | 改用 `APP_UID` 環境變數名稱 |
| 現有 root-owned files 無法被新容器刪除 | `docker compose run --user root` 一次性清理 |

## 風險評估

- **低風險**：不改動 production code，只改 Docker 設定
- `chmod 777` 限於容器內的 writable 目錄，不影響 host 安全性
- `.env` 已在 `.gitignore` 中，不會洩漏 UID/GID

## 驗證

```bash
# 容器 user
docker compose run --rm prod id
# → uid=1006 gid=1007

# uv run 正常
docker compose run --rm prod uv run python -c "from currency_predictor.config import ConfigManager; print('OK')"

# 檔案 ownership
docker compose run --rm prod uv run python -c "open('/app/results/test','w').write('ok')"
ls -la results/test  # → renhaohsu:renhaohsu
rm results/test      # → OK, no sudo needed

# Clean script
./scripts/clean.sh   # → no Permission denied

# 全部 tests
docker compose run --rm test  # → 539 passed
```

## 完成標準

- [x] 容器以 host user UID/GID 執行
- [x] Volume-mounted 檔案為 host user 所有
- [x] `./scripts/clean.sh` 在 host 上不需 sudo
- [x] Jupyter notebook 正常啟動（不需 `--allow-root`）
- [x] 539 tests 通過
