# Prod Docker Image Stale + Config Mount 問題

- **Date**: 2026-04-04 21:25
- **Status**: completed
- **Module**: docker-compose, visualization

## 問題

### Issue 1: prod service 使用 stale image

`docker compose run --rm prod` 跑的是 build 時 bake 進去的程式碼。`prod` service 不 mount `./src`（只有 `dev` 有），所以：

- Plan #0210 (Dashboard Volume=0 Fix) 修了 `visualizer.py`，但 prod image 沒 rebuild → forex dashboard **仍顯示空白「成交量」面板**而非「每日價格波幅」
- Plan #0260 加的 `--config` flag 改了 `main.py`，但 prod image 沒 rebuild → `--config` flag 不存在

**根本原因**：用戶跑 `docker compose run --rm prod` 時不會自動 rebuild，必須手動 `docker compose build prod`。

### Issue 2: prod 無法讀取自訂 config 檔

`prod` volumes 只 mount 了 `./config.json:/app/config.json:ro`。新增的 `tw2330_config.json` 在容器內不存在，`--config tw2330_config.json` 會 fallback 到 Pydantic 預設值。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `docker-compose.yml` | prod volumes 加 `tw2330_config.json` mount |
| `README.md` | 加「首次 / 程式碼更新後需 rebuild」提示 |

## 解法

### Issue 1: README 加提示 + build 指令

在 README Step 1 的 Docker 區塊前加提示：

```markdown
> **首次使用或程式碼更新後**，需先 rebuild image：`docker compose build prod`
```

目前 README 完整範例段落已有 `docker compose build prod`，但 Step 1 主要指令區塊沒有。

### Issue 2: mount 自訂 config

`docker-compose.yml` prod volumes 加：

```yaml
- ./tw2330_config.json:/app/tw2330_config.json:ro
```

或更通用的做法：mount 整個 project root 的 config files（但會增加 volume 數量）。

## 驗證

```bash
# Rebuild prod image
docker compose build prod

# 驗證 USDTWD dashboard Volume fallback
docker compose run --rm prod uv run main.py --full
# → results/latest/figures/USDTWD_dashboard.png 應顯示「每日價格波幅」而非「成交量」

# 驗證 tw2330 config
docker compose run --rm prod uv run main.py --config tw2330_config.json --full
# → results/latest/figures/2330.TW_dashboard.png 應存在且顯示成交量（股票有 Volume）
```

## 完成標準

- `docker compose build prod` 後，USDTWD dashboard 顯示「每日價格波幅」
- `--config tw2330_config.json` 在 prod 容器內可正常執行
- README 有明確的 rebuild 提示
