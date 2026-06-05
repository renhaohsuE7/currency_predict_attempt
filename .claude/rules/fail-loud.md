---
paths:
  - "src/**/*.py"
---

# Fail-Loud:不可靜默吞掉評估/計算錯誤

## 原則

**評估、metrics 計算、資料準備等「會產生結果」的程式碼,失敗時必須 fail loud,不可靜默吞成空結果再回報成功。**

「靜默吞錯」= 用 `except: return {}` / `return None` / 塞空 dict,讓上層以為成功,實際上沒有任何有效輸出。這會讓「跑完了但沒結果」偽裝成「跑成功」,極難察覺。

**Why:** 2026-06-05 panel 訓練實跑時,`test_days=60 < seq_len=64` 使每檔評估丟出例外,被
`except Exception: per_symbol[sym] = {}` 吞掉 → 聚合 metrics 全空,但 `run()` 仍回 exit 0、寫出
`panel_results.json`,表面「成功」。同源問題也出現在 `_evaluate_model` 把例外吞成 `{}` 導致模型靜默被
排除於 ranking(見 code review #10)。

## 規則

1. **事前驗證(preferred)**:能在執行前檢查的不變量,就在入口處檢查並 `raise` 清楚的錯誤,
   不要等到深層才失敗。
   - 例:窗口模型評估前驗證 `test_days >= seq_len + pred_len`
     (見 `prepare_training_data`、`PanelTrainer.run`)。

2. **不可把例外吞成空結果當成功**:
   ```python
   # ❌ 禁止
   try:
       metrics = model.evaluate_rolling(...)
   except Exception:
       return {}            # 上層誤判成功,結果其實是空的

   # ✅ 要嘛讓它 raise
   metrics = model.evaluate_rolling(...)   # 前置條件已驗證,失敗就該炸

   # ✅ 要嘛 batch 中允許跳過單一項,但「失敗」必須反映在回傳/退出碼/聚合
   results, failures = {}, []
   for item in items:
       try:
           results[item] = evaluate(item)
       except Exception as e:
           logger.error(f"評估 {item} 失敗: {e}")   # ERROR,非 debug
           failures.append(item)
   if not results:                          # 全失敗 → 不可當成功
       raise RuntimeError(f"全部評估失敗: {failures}")
   ```

3. **空結果要大聲**:若聚合/輸出可能為空,明確 `logger.warning`/`raise` 並指出可能原因與修法,
   不要安靜寫出空檔案。

4. **log level 要對**:被吞掉而影響結果正確性的錯誤用 `logger.error`(或 `warning` 並反映在回傳),
   不要降級成 `logger.debug` 藏起來。

## 例外(可以 catch 不 raise 的情況)

- 真正的「可選 / best-effort」路徑,且**失敗不影響主結果正確性**(例:抽樣快取驗證的單一 spot-check
  網路錯誤、可選視覺化生成失敗)。這些 catch 後仍要 log,且主流程結果不受影響。

## 驗證

完成評估/metrics 相關功能後,用 `/verify` 或 `superpowers:verification-before-completion`
**實跑並確認輸出非空**,不要只靠 toy fixture 的單元測試就宣稱完成(單元測試的小 seq_len 抓不到
真實 config 的契約問題)。
