# Planning Rule

## 非 trivial 修改前必須先寫計畫

在實作任何非 trivial 的變更之前，先寫計畫文件到 `docs/plans/` 並取得 user 同意。

### 計畫文件命名

```
docs/plans/YYYY-MM-DD-HHmm-{plan-name-or-module-name}.md
```

- 使用建立時的日期時間（24 小時制，精確到分鐘）
- `{plan-name-or-module-name}` 用 kebab-case，描述功能或模組名稱
- 範例：
  - `docs/plans/2026-04-03-1430-patchtst-lightning-implementation.md`
  - `docs/plans/2026-04-03-0900-refactor-base-trainer.md`
  - `docs/plans/2026-04-04-1015-add-rsi-indicator.md`

### 計畫文件模板

```markdown
# {Plan Title}

- **Date**: YYYY-MM-DD HH:mm
- **Status**: draft | approved | in-progress | completed | cancelled
- **Module**: affected module(s)

## 目標

簡述這次變更要達成什麼。

## 影響範圍

列出會修改的檔案和模組。

## 實作步驟

1. Step 1
2. Step 2
3. ...

## 風險評估

可能的風險和對策。

## 完成標準

怎樣算做完（測試通過、功能驗證等）。
```

### 可以跳過計畫的情況

- Trivial 修改：typo、單行修正、明顯 bug fix
- 純文檔更新
- 測試補充（不改動 production code）

### 流程

1. 分析需求，撰寫計畫到 `docs/plans/`
2. 向 user 說明計畫內容
3. 取得同意後才開始實作
4. 實作過程中更新 Status 為 `in-progress`
5. 實作完成後更新 Status 為 `completed`
