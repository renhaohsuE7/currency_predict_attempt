"""Forecast verifier — 比對 out-of-sample 預測 vs 實際資料。"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.collectors import YahooFinanceCollector

logger = logging.getLogger(__name__)


class ForecastVerifier:
    """驗證 out-of-sample forecast 的準確度。

    Usage::

        verifier = ForecastVerifier()
        # 直接指定 forecast JSON 路徑
        report = verifier.verify(forecast_path="results/runs/.../forecast_USDTWD.json")

        # 或透過 run_id 自動定位
        report = verifier.verify(run_id="20260413_022039")
    """

    def __init__(self, base_dir: str = "results"):
        self.collector = YahooFinanceCollector()
        self.base_dir = Path(base_dir)

    def verify(
        self,
        forecast_path: Optional[str | Path] = None,
        run_id: Optional[str] = None,
        op_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """載入 forecast JSON → 下載實際資料 → 計算準確度。

        Args:
            forecast_path: 直接指定 forecast JSON 路徑
            run_id: 透過 run ID 定位 (e.g., '20260413_022039')。None = 最新
            op_id: 透過 operation ID 定位。None = 該 run 的最新 op

        Returns:
            驗證結果 dict，包含各 symbol 的 per-model metrics
        """
        forecast_files = self._find_forecast_files(forecast_path, run_id, op_id)

        if not forecast_files:
            return {
                'success': False,
                'error': 'No forecast JSON files found',
                'symbols': {},
            }

        results: Dict[str, Any] = {
            'success': True,
            'verification_date': datetime.now().isoformat(timespec='seconds'),
            'symbols': {},
        }

        for fpath in forecast_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    forecast = json.load(f)

                symbol = forecast['symbol']
                symbol_result = self._verify_single(forecast)
                results['symbols'][symbol] = symbol_result
            except Exception as e:
                logger.error(f"驗證 {fpath} 失敗: {e}")
                results['symbols'][str(fpath)] = {'error': str(e)}

        return results

    def _verify_single(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """驗證單一 symbol 的 forecast。"""
        symbol = forecast['symbol']

        # 收集所有 prediction dates
        all_dates: set[str] = set()
        for model_data in forecast['models'].values():
            all_dates.update(model_data.get('prediction_dates', []))

        if not all_dates:
            return {'error': 'No prediction dates found'}

        sorted_dates = sorted(all_dates)
        min_date = sorted_dates[0]
        # yfinance end date is exclusive — extend by a few days
        max_date_dt = pd.Timestamp(sorted_dates[-1]) + timedelta(days=3)
        max_date = max_date_dt.strftime('%Y-%m-%d')

        # 下載實際資料
        actual_data = self.collector.get_currency_data_range(
            symbol, min_date, max_date
        )

        if actual_data is None or actual_data.empty:
            return {
                'error': f'No actual data available for {symbol} ({min_date} ~ {max_date})',
                'forecast_generated_at': forecast.get('forecast_generated_at', ''),
            }

        # 建立日期 → Close 映射（normalize 去掉 timezone）
        actual_map = self._build_actual_map(actual_data)

        # 各模型比對
        model_results: Dict[str, Any] = {}
        for model_name, model_data in forecast['models'].items():
            model_results[model_name] = self._verify_model(
                model_data, actual_map
            )

        total_dates = len(sorted_dates)
        matched_dates = len(actual_map.keys() & set(sorted_dates))

        return {
            'forecast_generated_at': forecast.get('forecast_generated_at', ''),
            'prediction_horizon': forecast.get('prediction_horizon', 0),
            'coverage': {
                'total': total_dates,
                'matched': matched_dates,
                'missing': total_dates - matched_dates,
            },
            'models': model_results,
        }

    def _verify_model(
        self, model_data: Dict[str, Any], actual_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """驗證單一模型的 forecast。"""
        pred_dates = model_data.get('prediction_dates', [])
        predictions = model_data.get('predictions', [])
        last_known_value = model_data.get('last_known_value', None)

        details: List[Dict[str, Any]] = []
        for i, date_str in enumerate(pred_dates):
            if i >= len(predictions):
                break
            actual_val = actual_map.get(date_str)
            if actual_val is None:
                continue
            pred_val = predictions[i]
            details.append({
                'date': date_str,
                'predicted': pred_val,
                'actual': actual_val,
                'error': pred_val - actual_val,
                'abs_error': abs(pred_val - actual_val),
                'horizon': i + 1,
            })

        if not details:
            return {
                'metrics': {},
                'details': [],
                'n_matched': 0,
            }

        # 計算 metrics
        errors = np.array([d['error'] for d in details])
        actuals = np.array([d['actual'] for d in details])
        preds = np.array([d['predicted'] for d in details])

        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # MAPE — 避免除以零
        safe_actuals = np.where(actuals == 0, np.nan, actuals)
        mape_values = np.abs(errors / safe_actuals) * 100
        mape = float(np.nanmean(mape_values))

        # Directional accuracy — 基於 previous actual (需要 last_known_value)
        direction_acc = self._compute_directional_accuracy(
            details, last_known_value
        )

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'directional_accuracy': direction_acc,
        }

        return {
            'metrics': metrics,
            'details': details,
            'n_matched': len(details),
        }

    @staticmethod
    def _compute_directional_accuracy(
        details: List[Dict[str, Any]], last_known_value: Optional[float]
    ) -> float:
        """計算方向準確度。

        看每個預測日期相對於「前一天 actual」的方向是否正確。
        """
        if len(details) < 1:
            return 0.0

        correct = 0
        total = 0
        prev_actual = last_known_value

        for d in details:
            if prev_actual is None:
                prev_actual = d['actual']
                continue

            actual_dir = d['actual'] - prev_actual
            pred_dir = d['predicted'] - prev_actual

            if actual_dir == 0:
                # 無方向 — 跳過
                prev_actual = d['actual']
                continue

            if (actual_dir > 0 and pred_dir > 0) or (actual_dir < 0 and pred_dir < 0):
                correct += 1
            total += 1
            prev_actual = d['actual']

        return float(correct / total) if total > 0 else 0.0

    @staticmethod
    def _build_actual_map(actual_data: pd.DataFrame) -> Dict[str, float]:
        """建立 date_str → Close price 映射。"""
        result: Dict[str, float] = {}
        for idx, row in actual_data.iterrows():
            # Normalize index to date string
            if hasattr(idx, 'tz_localize'):
                try:
                    idx = idx.tz_localize(None)
                except TypeError:
                    idx = idx.tz_convert(None)
            date_str = pd.Timestamp(idx).strftime('%Y-%m-%d')
            result[date_str] = float(row['Close'])
        return result

    def _find_forecast_files(
        self,
        forecast_path: Optional[str | Path],
        run_id: Optional[str],
        op_id: Optional[str],
    ) -> List[Path]:
        """定位 forecast JSON 檔案。"""
        if forecast_path:
            p = Path(forecast_path)
            return [p] if p.exists() else []

        runs_dir = self.base_dir / "runs"
        if not runs_dir.exists():
            logger.error(f"Runs directory not found: {runs_dir}")
            return []

        # 找到 run directory
        if run_id:
            run_dir = runs_dir / run_id
            if not run_dir.exists():
                logger.error(f"Run not found: {run_dir}")
                return []
        else:
            # 使用最新的 run
            existing_runs = sorted(
                d for d in runs_dir.iterdir()
                if d.is_dir() and not d.is_symlink()
            )
            if not existing_runs:
                logger.error("No runs found")
                return []
            run_dir = existing_runs[-1]

        # 找到 operation directory
        if op_id:
            op_dir = run_dir / op_id
            if not op_dir.exists():
                logger.error(f"Operation not found: {op_dir}")
                return []
            return list(op_dir.glob("forecast_*.json"))

        # 搜尋所有 op directories for forecast files
        # 優先用 manifest 找最新 completed operation
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # 找最新的 completed operation（有 forecast）
            for op in reversed(manifest.get('operations', [])):
                if op.get('status') == 'completed':
                    candidate = run_dir / op['op_id']
                    files = list(candidate.glob("forecast_*.json"))
                    if files:
                        return files

        # Fallback: 掃描所有子目錄
        all_files: List[Path] = []
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and not sub.is_symlink():
                all_files.extend(sub.glob("forecast_*.json"))

        return all_files

    def save_verification_report(
        self,
        report: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """儲存驗證報告為 JSON。"""
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Verification report saved: {output_path}")

    def format_report(self, report: Dict[str, Any]) -> str:
        """將驗證結果格式化為可讀文字。"""
        lines: List[str] = []
        lines.append("=" * 50)
        lines.append("Forecast Verification Report")
        lines.append(f"Verification date: {report.get('verification_date', 'N/A')}")
        lines.append("=" * 50)

        for symbol, sym_data in report.get('symbols', {}).items():
            if 'error' in sym_data:
                lines.append(f"\n{symbol}: ERROR — {sym_data['error']}")
                continue

            lines.append(f"\nSymbol: {symbol}")
            lines.append(f"Forecast date: {sym_data.get('forecast_generated_at', 'N/A')}")

            coverage = sym_data.get('coverage', {})
            lines.append(
                f"Coverage: {coverage.get('matched', 0)}/{coverage.get('total', 0)} "
                f"trading days matched"
            )

            # Model metrics table
            models = sym_data.get('models', {})
            if models:
                lines.append("")
                lines.append(
                    f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'Dir.Acc':>8} {'N':>4}"
                )
                lines.append("-" * 65)
                for model_name, model_result in models.items():
                    m = model_result.get('metrics', {})
                    n = model_result.get('n_matched', 0)
                    if not m:
                        lines.append(f"{model_name:<25} {'N/A':>8}")
                        continue
                    lines.append(
                        f"{model_name:<25} "
                        f"{m.get('rmse', 0):>8.4f} "
                        f"{m.get('mae', 0):>8.4f} "
                        f"{m.get('mape', 0):>7.3f}% "
                        f"{m.get('directional_accuracy', 0):>7.1%} "
                        f"{n:>4}"
                    )

                # Per-date details for best model
                best_model = min(
                    models.items(),
                    key=lambda x: x[1].get('metrics', {}).get('rmse', float('inf')),
                )
                best_name, best_data = best_model
                details = best_data.get('details', [])
                if details:
                    lines.append(f"\nBest model ({best_name}) — per-date details:")
                    lines.append(
                        f"  {'Date':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10}"
                    )
                    for d in details:
                        lines.append(
                            f"  {d['date']:<12} "
                            f"{d['predicted']:>10.4f} "
                            f"{d['actual']:>10.4f} "
                            f"{d['error']:>+10.4f}"
                        )

        lines.append("")
        return "\n".join(lines)
