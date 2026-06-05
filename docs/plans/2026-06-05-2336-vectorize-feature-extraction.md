# Plan: 向量化 _extract_features_from_data(加速 panel/訓練特徵抽取)

- **Date**: 2026-06-05 23:36
- **Status**: completed
- **結果**: 單股(2430 列×41 特徵)22.2s → 36ms = **618× 加速**,輸出逐元素相符;全測 665 passed
- **Module**: models/patchtst/sklearn/model.py
- **動機**: panel 實跑顯示 `_extract_features_from_data` 對 ~67k 序列花 ~10 分鐘(Python 雙層迴圈 + 每 patch 每 feature 一次 `np.polyfit`),是最大瓶頸。

## 目標
用 numpy 向量化 `_extract_features_from_data`,輸出**與現有迴圈版逐元素等價**(float allclose),大幅縮短時間。

## 不變式(必須完全保留)
- 視窗:`i in range(N - seq_len - pred_len + 1)`,每窗 `X[i:i+seq_len]`、target `y[i+seq_len:i+seq_len+pred_len]`。
- 每窗 patch:絕對起點 `i + p*stride`,`p=0..n_patches-1`(`n_patches=(seq_len-patch_len)//stride+1`,與 `_create_patches` 等價)。
- 每 patch 特徵順序:`[mean, std(ddof=0), min, max, median, trend_slope]`,各 `n_feat` 維,依此串接。
- 視窗特徵 = 各 patch 區塊依 p 串接;trend_slope = 最小平方斜率(= polyfit deg1 斜率)。
- 資料不足(視窗數 <= 0)→ `raise ValueError("資料不足以創建訓練序列")`(同原行為)。

## 作法
- `sliding_window_view(series, patch_len, axis=0)` 取得全序列所有 patch → 一次算 mean/std/min/max/median。
- 斜率向量化:`slope = ((pv - mean[...,None]) * dx).sum(-1) / sum(dx^2)`,`dx = arange(patch_len) - mean`(數學等同 polyfit)。
- 組 `PF=[P, 6*n_feat]`,以索引矩陣 `J[i,p]=i+p*stride` gather → `features=[n_windows, n_patches*6*n_feat]`。
- targets:`sliding_window_view(y, pred_len)[seq_len : seq_len+n_windows]`。
- **保留** `_create_patches` / `_extract_patch_features`(predict() 單窗仍用,不需向量化)。

## 安全網(最重要)
- **Characterization test**:新向量化輸出 vs 舊迴圈參考實作,在多組隨機資料/參數(不同 seq_len/patch_len/stride/n_feat/pred_len)上 `np.testing.assert_allclose`(features 與 targets 皆比對)。
- 全測維持綠(現有 sklearn 模型測試會間接驗證)。
- 量測加速(panel 實跑時間對比)。

## 風險
- float 末位差異(polyfit vs 解析斜率)→ 用 allclose 容忍;對模型訓練無實質影響(現有測試不比對精確預測值)。
- 邊界:patch_len>seq_len、視窗數<=0 → 保留原 raise。

## 完成標準
- [ ] characterization test 通過(逐元素 allclose)
- [ ] 全測綠;black/flake8/mypy clean
- [ ] panel 特徵抽取時間明顯下降
