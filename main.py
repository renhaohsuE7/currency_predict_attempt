"""
Currency Prediction Main Script

簡潔的主程式，使用模組化組件運行貨幣預測管道
"""

import logging
import argparse

from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline, ModelComparer, RunManager
from currency_predictor.backtesting import BacktestRunner
from currency_predictor.reporting.formatter import ResultFormatter
from currency_predictor.visualization import CurrencyVisualizer
from currency_predictor.utils import setup_logging, ensure_directories


def main(visualize=False, compare=False, models=None, symbols=None,
         fresh=False, train_only=False, predict_only=False,
         continue_run=False, use_op=None,
         backtest=False, backtest_strategy=None,
         config_path=None):
    """
    主函數 - 運行貨幣預測管道

    Args:
        visualize: 是否生成視覺化圖表
        compare: 是否執行多模型比較
        models: CLI 指定的模型名稱列表（覆蓋 config）
        symbols: CLI 指定的符號列表（覆蓋 config）
        fresh: 是否強制重新下載資料 + 重新訓練模型
        train_only: 是否只執行訓練
        predict_only: 是否只執行預測
        continue_run: 是否在最新 run 目錄內繼續
        use_op: predict_only 時指定使用的操作 ID
    """

    # 設置日誌
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting Currency Prediction Pipeline")

    # 確保必要目錄存在
    ensure_directories(['data', 'models', 'results', 'results/figures'])

    try:
        # 載入配置
        config_manager = ConfigManager(config_path=config_path)
        config = config_manager.get_config()

        # 建立 RunManager
        if continue_run:
            run_manager = RunManager.continue_latest(base_dir="results")
            run_manager.setup()  # idempotent
        else:
            run_manager = RunManager(base_dir="results")
            run_manager.setup(config=config)

        # CLI 覆蓋 symbols
        target_symbols = symbols or config_manager.get_symbols()
        prediction_horizon = config_manager.get_prediction_horizon()

        # 判斷執行模式
        model_names = models or config_manager.get_model_names()
        use_compare = compare or (model_names is not None)

        # Backtest 模式
        if backtest:
            op_type = "backtest"
            mode = "backtest"
        elif train_only:
            op_type = "train_only"
        elif predict_only:
            op_type = "predict_only"
        else:
            op_type = "full"

        if not backtest:
            mode = "compare" if use_compare else "single"

        # 開始操作 — 建立 op_dir 子目錄
        run_manager.start_operation(
            op_type=op_type,
            mode=mode,
            symbols=target_symbols,
            models=model_names or [],
            model_source_op_id=use_op if predict_only else None,
        )

        try:
            if backtest:
                exit_code = _run_backtest_mode(
                    config, target_symbols, model_names,
                    backtest_strategy, visualize, logger, run_manager,
                )
            elif use_compare:
                exit_code = _run_compare_mode(
                    config, model_names, target_symbols, prediction_horizon,
                    visualize, logger, run_manager, fresh,
                    op_type=op_type, use_op=use_op,
                )
            else:
                exit_code = _run_single_mode(
                    config, config_manager, target_symbols, prediction_horizon,
                    visualize, logger, run_manager, fresh,
                    op_type=op_type, use_op=use_op,
                )

            run_manager.complete_operation()
        except Exception as e:
            run_manager.fail_operation(str(e))
            raise

        # 更新 latest symlink（指向最新 op_dir）
        run_manager.update_latest_symlink()
        logger.info(f"Run results saved to: {run_manager.op_dir}")

        return exit_code

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


def _run_single_mode(config, config_manager, symbols, horizon, visualize,
                     logger, run_manager, fresh=False,
                     op_type="full", use_op=None):
    """單一模型 pipeline"""
    pipeline = PredictionPipeline(config, output_dir=str(run_manager.op_dir))

    if op_type == "train_only":
        results = pipeline.run_train_only(
            symbols=symbols,
            force_retrain=fresh,
        )
    elif op_type == "predict_only":
        # 從之前的操作載入模型
        model_paths = _resolve_model_paths(
            run_manager, symbols, config.get('model_name', 'patchtst_sklearn'),
            use_op,
        )
        results = pipeline.run_predict_only(
            symbols=symbols,
            prediction_horizon=horizon,
            model_paths=model_paths,
        )
    else:
        results = pipeline.run_full_pipeline(
            symbols=symbols,
            prediction_horizon=horizon,
            save_results=True,
            force_retrain=fresh,
        )

    formatter = ResultFormatter(use_logger=True)
    formatter.format_complete_results(results)

    if visualize and results.get('success', False) and op_type != "train_only":
        _generate_visualizations(symbols, logger, run_manager)

    return 0 if results.get('success', False) else 1


def _run_compare_mode(config, model_names, symbols, horizon, visualize,
                      logger, run_manager, fresh=False,
                      op_type="full", use_op=None):
    """多模型比較模式"""
    if model_names is None:
        from currency_predictor.models.factory import ModelFactory
        available = ModelFactory.get_available_models()
        model_names = [
            name for name, info in available.items()
            if info['available'] and not info.get('alias_for')
        ]

    logger.info(f"Multi-model comparison: models={model_names}, symbols={symbols}")

    comparer = ModelComparer(
        model_names=model_names,
        config=config,
        output_dir=str(run_manager.op_dir),
    )

    if op_type == "train_only":
        train_results = comparer.train_only(
            symbols=symbols,
            force_update=fresh,
        )
        # 顯示訓練結果
        formatter = ResultFormatter(use_logger=True)
        logger.info("Train-only completed for compare mode")
        has_results = any(
            any(m.get('training_completed') for m in sr.get('models', {}).values())
            for sr in train_results.get('symbols_results', {}).values()
        )
        return 0 if has_results else 1

    elif op_type == "predict_only":
        # 找到模型來源目錄
        source_model_dir = _resolve_compare_model_dir(run_manager, use_op)
        comparison_results = comparer.predict_only(
            symbols=symbols,
            prediction_horizon=horizon,
            model_dir=source_model_dir,
        )
    else:
        comparison_results = comparer.compare(
            symbols=symbols,
            prediction_horizon=horizon,
            force_update=fresh,
        )

    # 顯示比較結果
    formatter = ResultFormatter(use_logger=True)
    formatter.format_comparison_results(comparison_results)

    # 生成比較報告
    report = formatter.generate_comparison_report(comparison_results)
    report_path = run_manager.op_dir / "comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Comparison report saved to: {report_path}")

    # 視覺化
    if visualize:
        _generate_visualizations(symbols, logger, run_manager)
        _generate_comparison_charts(comparison_results, symbols, logger,
                                    run_manager)
        _generate_forecast_charts(comparison_results, symbols, logger,
                                  run_manager)

    has_results = len(comparison_results.get('overall_ranking', [])) > 0
    return 0 if has_results else 1


def _run_backtest_mode(config, symbols, model_names, strategy,
                      visualize, logger, run_manager):
    """Walk-forward backtesting 模式"""
    bt_config = config.get('backtest', {})
    effective_strategy = strategy or bt_config.get('strategy', 'rolling')

    # Resolve model list
    if model_names is None:
        model_names = [config.get('model_name', 'patchtst_sklearn')]

    logger.info(
        f"Backtest mode: strategy={effective_strategy}, "
        f"models={model_names}, symbols={symbols}"
    )

    runner = BacktestRunner(config=config, output_dir=str(run_manager.op_dir))
    results = runner.run(
        symbols=symbols,
        model_names=model_names,
        strategy=effective_strategy,
        initial_train_days=bt_config.get('initial_train_days', 252),
        test_step_days=bt_config.get('test_step_days', 30),
        test_window_days=bt_config.get('test_window_days', 30),
        data_period=bt_config.get('data_period', '3y'),
    )

    # Display results
    formatter = ResultFormatter(use_logger=True)
    formatter.format_backtest_results(results)

    # Visualizations
    if visualize:
        vis = CurrencyVisualizer(output_dir=str(run_manager.op_dir / "figures"))
        for symbol, model_results in results.items():
            for model_name, bt_result in model_results.items():
                import pandas as pd
                dates = pd.DatetimeIndex(bt_result.overall_dates)
                vis.plot_equity_curve(
                    bt_result.overall_equity_curve, dates,
                    symbol=symbol, model_name=model_name,
                )
                vis.plot_drawdown(
                    bt_result.overall_equity_curve, dates,
                    symbol=symbol, model_name=model_name,
                )

    has_results = any(len(mr) > 0 for mr in results.values())
    return 0 if has_results else 1


def _resolve_model_paths(run_manager, symbols, model_name, use_op):
    """從 manifest 查找各 symbol 的模型路徑"""
    from currency_predictor.prediction.predictor import _clean_symbol

    model_paths = {}
    for symbol in symbols:
        clean = _clean_symbol(symbol)
        path = run_manager.get_latest_model_path(
            symbol=clean,
            model_name=model_name,
            source_op_id=use_op,
        )
        if path:
            model_paths[symbol] = str(path)
    return model_paths


def _resolve_compare_model_dir(run_manager, use_op):
    """找到 compare 模式 predict_only 的模型目錄"""
    run_manager._load_manifest()
    ops = run_manager._manifest.get("operations", [])

    if use_op:
        return str(run_manager.run_dir / use_op)

    # 反向找最新的 full/train_only
    for op in reversed(ops):
        if op["type"] in ("full", "train_only") and op["status"] == "completed":
            return str(run_manager.run_dir / op["op_id"])

    return str(run_manager.op_dir)


def _generate_visualizations(symbols, logger, run_manager):
    """生成單一模型的視覺化圖表"""
    from currency_predictor.data.storage import DataStorage

    logger.info("Generating visualization charts...")
    try:
        storage = DataStorage("data")
        visualizer = CurrencyVisualizer(
            output_dir=str(run_manager.figures_dir),
        )

        for symbol in symbols:
            try:
                clean = symbol.replace('=X', '') if symbol.endswith('=X') else symbol
                df = storage.load_raw_data(clean)
                if df is None:
                    logger.warning(f"Data not found for {symbol}, skipping visualization")
                    continue
                visualizer.create_dashboard(
                    df=df,
                    symbol=symbol,
                    save_path=f"{clean}_dashboard.png",
                )
                logger.info(f"Dashboard created for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to create visualization for {symbol}: {e}")

        logger.info(f"Visualizations saved to: {run_manager.figures_dir}")

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")


def _generate_comparison_charts(comparison_results, symbols, logger,
                                run_manager):
    """生成多模型比較圖表"""
    import numpy as np
    import pandas as pd

    logger.info("Generating comparison charts...")
    try:
        visualizer = CurrencyVisualizer(
            output_dir=str(run_manager.figures_dir),
        )

        for symbol in symbols:
            sym_data = comparison_results.get('symbols_results', {}).get(symbol)
            if sym_data is None or 'error' in sym_data:
                continue

            models = sym_data.get('models', {})
            model_predictions = {}
            metrics = {}

            for mname, mresult in models.items():
                if mresult.get('predictions') is not None:
                    model_predictions[mname] = mresult['predictions']
                test_m = mresult.get('test_metrics', {})
                if test_m:
                    metrics[mname] = test_m

            if not model_predictions:
                continue

            # 使用真實 actual 資料（y_test from comparer）
            actual_series = sym_data.get('actual')
            if actual_series is not None:
                actual = actual_series
            else:
                # Fallback: placeholder
                n = max(len(p) for p in model_predictions.values())
                dummy_index = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
                actual = pd.Series(
                    np.zeros(n), index=dummy_index, name='actual'
                )

            clean_symbol = symbol.replace('=X', '') if symbol.endswith('=X') else symbol
            visualizer.plot_model_comparison(
                actual=actual,
                model_predictions=model_predictions,
                symbol=symbol,
                metrics=metrics if metrics else None,
                save_path=f"{clean_symbol}_model_comparison.png",
            )
            logger.info(f"Comparison chart created for {symbol}")

    except Exception as e:
        logger.error(f"Comparison chart generation failed: {e}")


def _generate_forecast_charts(comparison_results, symbols, logger,
                              run_manager):
    """生成 forecast chart：actual history + future predictions"""
    import numpy as np
    from currency_predictor.data.storage import DataStorage

    logger.info("Generating forecast charts...")
    try:
        storage = DataStorage("data")
        visualizer = CurrencyVisualizer(
            output_dir=str(run_manager.figures_dir),
        )

        lookback = 15

        for symbol in symbols:
            sym_data = comparison_results.get('symbols_results', {}).get(symbol)
            if sym_data is None or 'error' in sym_data:
                continue

            # 取得最近 N 天 actual close prices
            clean = symbol.replace('=X', '') if symbol.endswith('=X') else symbol
            raw_df = storage.load_raw_data(clean)
            if raw_df is None:
                logger.warning(f"Data not found for {symbol}, skipping forecast")
                continue
            historical = raw_df['Close'].iloc[-lookback:]

            # 收集各模型 predictions
            models = sym_data.get('models', {})
            model_predictions = {}
            last_known_date = None

            for mname, mresult in models.items():
                preds = mresult.get('predictions')
                dates = mresult.get('prediction_dates')
                if preds is not None and dates is not None:
                    model_predictions[mname] = (dates, np.asarray(preds))
                    if last_known_date is None:
                        last_known_date = mresult.get('last_known_date')

            if not model_predictions or last_known_date is None:
                continue

            visualizer.plot_forecast(
                historical=historical,
                model_predictions=model_predictions,
                symbol=symbol,
                last_known_date=last_known_date,
                save_path=f"{clean}_forecast.png",
            )
            logger.info(f"Forecast chart created for {symbol}")

    except Exception as e:
        logger.error(f"Forecast chart generation failed: {e}")


if __name__ == "__main__":
    """應用程式入口點"""
    parser = argparse.ArgumentParser(
        description="Currency Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                      # Multi-model comparison + visualization (default)
  python main.py --single                             # Single model only
  python main.py --no-viz                             # Multi-model comparison, skip charts
  python main.py --single --no-viz                    # Single model, no charts
  python main.py --fresh                              # Force re-download + re-train
  python main.py --train-only                         # Only collect data + train models
  python main.py --predict-only --continue            # Only predict using latest models
  python main.py --predict-only --use-op a3f7c1e2     # Predict using specific op's models
  python main.py --models sklearn,huggingface         # Compare specific models
  python main.py --symbols AAPL,TSLA                  # Predict stock tickers
  python main.py --config tw2330_config.json          # Use custom config file
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to config JSON file (default: config.json)'
    )
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        default=True,
        help='Generate visualization charts (default: enabled)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        default=True,
        help='Run multi-model comparison mode (default: enabled)'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='Run single model only (disable multi-model comparison)'
    )
    parser.add_argument(
        '--no-viz',
        dest='no_viz',
        action='store_true',
        help='Skip visualization chart generation'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full pipeline: all models + comparison + visualization (same as default, kept for compatibility)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force re-download data and re-train models (ignore cache)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Run data collection + model training only (skip prediction)'
    )
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Run prediction only using pre-trained models'
    )
    parser.add_argument(
        '--continue',
        dest='continue_run',
        action='store_true',
        help='Continue in the latest existing run directory'
    )
    parser.add_argument(
        '--use-op',
        type=str,
        default=None,
        help='Operation ID to load models from (for --predict-only)'
    )
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated model names (e.g. sklearn,huggingface)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated symbols to predict (e.g. USDTWD=X,AAPL)'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run walk-forward backtesting instead of prediction'
    )
    parser.add_argument(
        '--backtest-strategy',
        type=str,
        default=None,
        choices=['rolling', 'expanding'],
        help='Walk-forward strategy (default: rolling)'
    )

    args = parser.parse_args()

    # Opt-out flags override defaults
    if args.single:
        args.compare = False
    if args.no_viz:
        args.visualize = False

    # --full is kept for backward compatibility (now a no-op since defaults match)
    if args.full:
        args.compare = True
        args.visualize = True

    # Validation
    if args.backtest and (args.train_only or args.predict_only):
        parser.error("--backtest is mutually exclusive with --train-only and --predict-only")

    if args.train_only and args.predict_only:
        parser.error("--train-only and --predict-only are mutually exclusive")

    if args.predict_only and not args.continue_run:
        args.continue_run = True  # predict-only implies --continue

    if args.use_op and not args.predict_only:
        parser.error("--use-op requires --predict-only")

    # Parse comma-separated lists
    models_list = args.models.split(',') if args.models else None
    symbols_list = args.symbols.split(',') if args.symbols else None

    exit_code = main(
        visualize=args.visualize,
        compare=args.compare,
        models=models_list,
        symbols=symbols_list,
        fresh=args.fresh,
        train_only=args.train_only,
        predict_only=args.predict_only,
        continue_run=args.continue_run,
        use_op=args.use_op,
        backtest=args.backtest,
        backtest_strategy=args.backtest_strategy,
        config_path=args.config,
    )
    exit(exit_code)
