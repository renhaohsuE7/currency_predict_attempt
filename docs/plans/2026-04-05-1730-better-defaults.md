# Better Defaults — compare + visualize by default

- **Date**: 2026-04-05 17:30
- **Status**: completed
- **Module**: main.py, tests/test_e2e_cli.py, README.md

## 目標

零 flag 就能得到最完整的結果：多模型比較 + dashboard + comparison chart。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `main.py` | argparse 預設值 `compare=True, visualize=True`；新增 `--single`, `--no-viz` opt-out flags |
| `tests/test_e2e_cli.py` | 更新 parser defaults、新增 `--single`/`--no-viz` tests、新增 default routing test |
| `README.md` | 更新 quick start、examples、CLI 參數表 |

## 實作

### argparse 變更

- `--visualize` / `--compare`：`default=True`
- `--single`：`action='store_true'` → sets `compare=False`
- `--no-viz`：`action='store_true'` → sets `visualize=False`
- `--full`：保留向後相容（現為 no-op）

### 行為矩陣

| 指令 | Mode | Visualize |
|------|------|-----------|
| `main.py` | compare | yes |
| `main.py --single` | single | yes |
| `main.py --no-viz` | compare | no |
| `main.py --single --no-viz` | single | no |
| `main.py --full` | compare | yes (same as default) |

## 驗證

- All tests passed
