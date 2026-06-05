# DVC Data Versioning

- **Date**: 2026-04-05 20:40
- **Status**: draft
- **Module**: infra
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §8

## 目標

使用 DVC (Data Version Control) 版本化資料和模型 artifacts，確保完整可重現性。

## 背景

- 目前 `data/raw/` 和 `models/` 在 `.gitignore` 中，完全不受版本控制
- 無法重現過去的訓練結果（不知道用了哪份資料、哪個 scaler）
- DVC 與 Git 整合，對資料做 hash-based versioning
- 支援多種 remote storage（local, S3, GCS）

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `.dvc/` | DVC 配置目錄 |
| `data/raw/*.dvc` | 資料版本指標檔 |
| `models/*.dvc` | 模型 artifacts 版本指標檔 |
| `.gitignore` | 調整 DVC 管理的路徑 |
| `Dockerfile` | 安裝 DVC |

## 實作步驟

### Step 1: 初始化 DVC

```bash
uv add --dev dvc
dvc init
dvc add data/raw/
dvc add models/
```

### Step 2: 設定 remote storage

- 先用 local remote（`/tmp/dvc-storage`）做 POC
- 之後可切換到 S3/GCS

### Step 3: Pipeline definition

- `dvc.yaml` 定義 collect → process → train → predict pipeline
- `dvc repro` 自動重跑有變更的 stage

## 完成標準

- `data/raw/` 和 `models/` 由 DVC 管理
- `dvc repro` 可重現完整 pipeline
- `.dvc` 檔案 tracked by Git
- 文件化使用方式

## 備註

- 此 plan 優先度較低，適合在 #0310 (MLflow) 之後實作
- 若 team size 小，前期可用 RunManager 的 run ID 機制替代
