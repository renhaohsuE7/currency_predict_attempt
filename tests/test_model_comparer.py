"""
測試多模型比較器 (ModelComparer)
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from currency_predictor.prediction.comparer import (
    ModelComparer,
    resolve_model_name,
    MODEL_SHORTNAMES,
)


class TestResolveModelName(unittest.TestCase):
    """測試 resolve_model_name 函數"""

    def test_full_name_passthrough(self):
        """完整模型名稱直接回傳"""
        self.assertEqual(resolve_model_name('patchtst_sklearn'), 'patchtst_sklearn')

    def test_shortname_resolution(self):
        """CLI 短名稱正確解析"""
        self.assertEqual(resolve_model_name('sklearn'), 'patchtst_sklearn')
        self.assertEqual(resolve_model_name('huggingface'), 'patchtst_huggingface')
        self.assertEqual(resolve_model_name('hf'), 'patchtst_huggingface')
        self.assertEqual(resolve_model_name('transformer'), 'patchtst_huggingface')

    def test_case_insensitive(self):
        """短名稱大小寫不敏感"""
        self.assertEqual(resolve_model_name('SKLEARN'), 'patchtst_sklearn')
        self.assertEqual(resolve_model_name('HuggingFace'), 'patchtst_huggingface')

    def test_unknown_name_raises(self):
        """未知名稱應拋出 ValueError"""
        with self.assertRaises(ValueError):
            resolve_model_name('nonexistent_model')


class TestDirectionAccuracy(unittest.TestCase):
    """測試 direction_accuracy 靜態方法"""

    def test_perfect_accuracy(self):
        """方向完全一致 → 1.0"""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 1.5, 2.5, 3.5])
        self.assertAlmostEqual(ModelComparer.direction_accuracy(actual, predicted), 1.0)

    def test_zero_accuracy(self):
        """方向完全相反 → 0.0"""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(ModelComparer.direction_accuracy(actual, predicted), 0.0)

    def test_partial_accuracy(self):
        """部分正確"""
        actual = np.array([1.0, 2.0, 1.5, 2.5])
        predicted = np.array([1.0, 1.5, 2.0, 2.5])
        # actual diffs: [+1, -0.5, +1] → [+, -, +]
        # pred diffs:   [+0.5, +0.5, +0.5] → [+, +, +]
        # match:        [T, F, T] → 2/3
        acc = ModelComparer.direction_accuracy(actual, predicted)
        self.assertAlmostEqual(acc, 2 / 3, places=5)

    def test_too_short_returns_zero(self):
        """序列長度不足應回傳 0.0"""
        self.assertEqual(ModelComparer.direction_accuracy(np.array([1.0]), np.array([2.0])), 0.0)
        self.assertEqual(ModelComparer.direction_accuracy(np.array([]), np.array([])), 0.0)


class TestComputeUnifiedMetrics(unittest.TestCase):
    """測試 compute_unified_metrics 靜態方法"""

    def test_perfect_prediction(self):
        """完全預測正確 → 所有 error 指標為 0"""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        metrics = ModelComparer.compute_unified_metrics(actual, predicted)
        self.assertAlmostEqual(metrics['rmse'], 0.0)
        self.assertAlmostEqual(metrics['mae'], 0.0)
        self.assertAlmostEqual(metrics['mape'], 0.0)

    def test_known_errors(self):
        """已知誤差的計算"""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        metrics = ModelComparer.compute_unified_metrics(actual, predicted)
        # errors: [-10, 10]
        self.assertAlmostEqual(metrics['mae'], 10.0)
        self.assertAlmostEqual(metrics['rmse'], 10.0)
        # MAPE: mean(|10/100|, |10/200|) * 100 = mean(0.1, 0.05) * 100 = 7.5
        self.assertAlmostEqual(metrics['mape'], 7.5)

    def test_different_lengths(self):
        """不同長度應截斷到較短者"""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0])
        metrics = ModelComparer.compute_unified_metrics(actual, predicted)
        self.assertAlmostEqual(metrics['rmse'], 0.0)

    def test_zero_actual_mape(self):
        """actual 為 0 時 MAPE 跳過該值"""
        actual = np.array([0.0, 0.0])
        predicted = np.array([1.0, 1.0])
        metrics = ModelComparer.compute_unified_metrics(actual, predicted)
        self.assertEqual(metrics['mape'], float('inf'))

    def test_metrics_keys(self):
        """回傳的字典應包含所有指標"""
        metrics = ModelComparer.compute_unified_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
        )
        for key in ('rmse', 'mae', 'mape', 'direction_accuracy'):
            self.assertIn(key, metrics)


class TestModelComparerInit(unittest.TestCase):
    """測試 ModelComparer 初始化"""

    def test_init_with_sklearn(self):
        """使用 sklearn 模型初始化"""
        comparer = ModelComparer(
            model_names=['sklearn'],
            config={'data_storage_path': '/tmp/test_comparer'},
        )
        self.assertIn('patchtst_sklearn', comparer.model_names)
        self.assertIn('patchtst_sklearn', comparer.predictors)

    def test_init_deduplicates(self):
        """重複名稱應去重"""
        comparer = ModelComparer(
            model_names=['sklearn', 'patchtst_sklearn', 'sklearn'],
            config={'data_storage_path': '/tmp/test_comparer'},
        )
        self.assertEqual(len(comparer.model_names), 1)

    def test_init_unknown_model_raises(self):
        """未知模型名稱在初始化時拋出"""
        with self.assertRaises(ValueError):
            ModelComparer(
                model_names=['nonexistent'],
                config={},
            )

    def test_init_multiple_models(self):
        """多模型初始化（sklearn + 可能不可用的 huggingface）"""
        # sklearn 一定可用
        comparer = ModelComparer(
            model_names=['sklearn'],
            config={'data_storage_path': '/tmp/test_comparer'},
        )
        self.assertEqual(len(comparer.predictors), 1)


class TestModelComparerCompare(unittest.TestCase):
    """測試 ModelComparer.compare 方法"""

    @patch.object(ModelComparer, '_init_predictors')
    def test_compare_structure(self, mock_init):
        """compare 回傳結構正確"""
        # 建立 mock predictor
        mock_predictor = MagicMock()
        mock_predictor.collect_and_store_data.return_value = {'SYM=X': True}
        mock_predictor.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_predictor.train_model.return_value = {
            'training_completed': True,
            'train_metrics': {'rmse': 0.1},
            'test_metrics': {'rmse': 0.2},
        }
        mock_predictor.predict.return_value = {
            'predictions': np.array([1.0, 2.0, 3.0]),
        }

        mock_init.return_value = {'patchtst_sklearn': mock_predictor}

        comparer = ModelComparer.__new__(ModelComparer)
        comparer.config = {}
        comparer.output_dir = MagicMock()
        comparer.model_names = ['patchtst_sklearn']
        comparer.predictors = mock_init.return_value

        result = comparer.compare(['SYM=X'], prediction_horizon=3)

        self.assertIn('symbols_results', result)
        self.assertIn('overall_ranking', result)
        self.assertIn('model_names', result)
        self.assertIn('SYM=X', result['symbols_results'])

    @patch.object(ModelComparer, '_init_predictors')
    def test_compare_best_model_selection(self, mock_init):
        """compare 正確選出最佳模型"""
        # 兩個 mock predictor，第二個 RMSE 更低
        mock_pred_a = MagicMock()
        mock_pred_a.collect_and_store_data.return_value = {'SYM=X': True}
        mock_pred_a.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_pred_a.train_model.return_value = {
            'training_completed': True,
            'train_metrics': {'rmse': 0.5},
            'test_metrics': {'rmse': 0.5},
        }
        mock_pred_a.predict.return_value = {'predictions': np.array([1.0])}

        mock_pred_b = MagicMock()
        mock_pred_b.collect_and_store_data.return_value = {'SYM=X': True}
        mock_pred_b.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_pred_b.train_model.return_value = {
            'training_completed': True,
            'train_metrics': {'rmse': 0.1},
            'test_metrics': {'rmse': 0.1},
        }
        mock_pred_b.predict.return_value = {'predictions': np.array([1.0])}

        mock_init.return_value = {
            'model_a': mock_pred_a,
            'model_b': mock_pred_b,
        }

        comparer = ModelComparer.__new__(ModelComparer)
        comparer.config = {}
        comparer.output_dir = MagicMock()
        comparer.model_names = ['model_a', 'model_b']
        comparer.predictors = mock_init.return_value

        result = comparer.compare(['SYM=X'])

        sym_result = result['symbols_results']['SYM=X']
        self.assertEqual(sym_result['best_model'], 'model_b')
        # overall ranking: model_b (0.1) < model_a (0.5)
        self.assertEqual(result['overall_ranking'][0][0], 'model_b')


class TestModelComparerActual(unittest.TestCase):
    """測試 compare 結果包含真實 actual 資料"""

    @patch.object(ModelComparer, '_init_predictors')
    def test_compare_result_contains_actual(self, mock_init):
        """compare 結果應包含 actual (y_test) 資料"""
        import pandas as pd

        y_test = pd.Series([31.5, 31.6, 31.7], name='Close')

        mock_predictor = MagicMock()
        mock_predictor.collect_and_store_data.return_value = {'SYM=X': True}
        mock_predictor.prepare_training_data.return_value = (
            MagicMock(), MagicMock(), MagicMock(), y_test
        )
        mock_predictor.train_model.return_value = {
            'training_completed': True,
            'train_metrics': {'rmse': 0.1},
            'test_metrics': {'rmse': 0.2},
        }
        mock_predictor.predict.return_value = {
            'predictions': np.array([31.5, 31.6, 31.7]),
        }

        mock_init.return_value = {'patchtst_sklearn': mock_predictor}

        comparer = ModelComparer.__new__(ModelComparer)
        comparer.config = {}
        comparer.output_dir = MagicMock()
        comparer.model_names = ['patchtst_sklearn']
        comparer.predictors = mock_init.return_value

        result = comparer.compare(['SYM=X'], prediction_horizon=3)

        sym_result = result['symbols_results']['SYM=X']
        self.assertIn('actual', sym_result)
        pd.testing.assert_series_equal(sym_result['actual'], y_test)


class TestModelComparerModelSaving(unittest.TestCase):
    """測試 compare() 訓練後儲存模型"""

    @patch.object(ModelComparer, '_init_predictors')
    def test_compare_saves_models(self, mock_init):
        """compare 訓練成功後應儲存模型"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {'SYM=X': True}
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            mock_predictor.train_model.return_value = {
                'training_completed': True,
                'train_metrics': {'rmse': 0.1},
                'test_metrics': {'rmse': 0.2},
            }
            mock_predictor.predict.return_value = {
                'predictions': np.array([1.0, 2.0]),
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            comparer.compare(['SYM=X'], prediction_horizon=3)

            mock_predictor.save_model.assert_called_once()
            call_path = mock_predictor.save_model.call_args[0][0]
            self.assertIn('models', call_path)
            self.assertIn('patchtst_sklearn', call_path)

    @patch.object(ModelComparer, '_init_predictors')
    def test_compare_model_path_in_result(self, mock_init):
        """compare 結果應包含 model_path"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {'SYM=X': True}
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            mock_predictor.train_model.return_value = {
                'training_completed': True,
                'train_metrics': {},
                'test_metrics': {'rmse': 0.1},
            }
            mock_predictor.predict.return_value = {
                'predictions': np.array([1.0]),
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            result = comparer.compare(['SYM=X'])

            model_result = result['symbols_results']['SYM=X']['models']['patchtst_sklearn']
            self.assertIn('model_path', model_result)

    @patch.object(ModelComparer, "_init_predictors")
    def test_compare_save_failure_records_save_error(self, mock_init):
        """save_model 回傳 False 時應記錄 save_error，且不可謊報 model_path。"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {"SYM=X": True}
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            mock_predictor.train_model.return_value = {
                "training_completed": True,
                "train_metrics": {"rmse": 0.1},
                "test_metrics": {"rmse": 0.2},
            }
            # 模擬儲存失敗（fail-loud 路徑）
            mock_predictor.save_model.return_value = False
            mock_predictor.predict.return_value = {
                "predictions": np.array([1.0, 2.0]),
            }

            mock_init.return_value = {"patchtst_sklearn": mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ["patchtst_sklearn"]
            comparer.predictors = mock_init.return_value

            result = comparer.compare(["SYM=X"], prediction_horizon=3)

            model_result = result["symbols_results"]["SYM=X"]["models"][
                "patchtst_sklearn"
            ]
            # save_model 確實被呼叫但失敗
            mock_predictor.save_model.assert_called_once()
            # 必須記錄 save_error，且不可塞 bogus model_path
            self.assertIn("save_error", model_result)
            self.assertNotIn("model_path", model_result)


class TestModelComparerTrainOnly(unittest.TestCase):
    """測試 ModelComparer.train_only"""

    @patch.object(ModelComparer, '_init_predictors')
    def test_train_only_structure(self, mock_init):
        """train_only 回傳結構正確"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {'SYM=X': True}
            mock_predictor.train_model.return_value = {
                'training_completed': True,
                'train_metrics': {'rmse': 0.1},
                'test_metrics': {'rmse': 0.2},
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            result = comparer.train_only(['SYM=X'])

            self.assertEqual(result['operation_type'], 'train')
            self.assertIn('symbols_results', result)
            self.assertIn('SYM=X', result['symbols_results'])

            sym = result['symbols_results']['SYM=X']
            self.assertIn('patchtst_sklearn', sym['models'])
            self.assertTrue(sym['models']['patchtst_sklearn']['training_completed'])

    @patch.object(ModelComparer, '_init_predictors')
    def test_train_only_saves_model(self, mock_init):
        """train_only 儲存模型"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {}
            mock_predictor.train_model.return_value = {
                'training_completed': True,
                'train_metrics': {},
                'test_metrics': {},
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            comparer.train_only(['SYM=X'])

            mock_predictor.save_model.assert_called_once()

    @patch.object(ModelComparer, '_init_predictors')
    def test_train_only_no_prediction(self, mock_init):
        """train_only 不執行預測"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.collect_and_store_data.return_value = {}
            mock_predictor.train_model.return_value = {
                'training_completed': True,
                'train_metrics': {},
                'test_metrics': {},
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            comparer.train_only(['SYM=X'])

            mock_predictor.predict.assert_not_called()


class TestModelComparerPredictOnly(unittest.TestCase):
    """測試 ModelComparer.predict_only"""

    @patch.object(ModelComparer, '_init_predictors')
    def test_predict_only_structure(self, mock_init):
        """predict_only 回傳結構正確"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # 先建立模型檔案
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "SYM_patchtst_sklearn.joblib").write_text("fake")

            mock_predictor = MagicMock()
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            mock_predictor.load_model.return_value = True
            mock_predictor.predict.return_value = {
                'predictions': np.array([1.0, 2.0, 3.0]),
            }
            mock_predictor.model.evaluate.return_value = {
                'rmse': 0.1, 'mae': 0.08, 'mse': 0.01,
            }

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            result = comparer.predict(
                ['SYM=X'],
                prediction_horizon=3,
                model_dir=tmpdir,
            )

            self.assertEqual(result['operation_type'], 'predict')
            self.assertIn('symbols_results', result)

            sym = result['symbols_results']['SYM=X']
            model_result = sym['models']['patchtst_sklearn']
            self.assertIn('predictions', model_result)

    @patch.object(ModelComparer, '_init_predictors')
    def test_predict_only_model_not_found(self, mock_init):
        """predict_only 模型不存在時記錄 error"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            result = comparer.predict_only(
                ['SYM=X'],
                prediction_horizon=3,
                model_dir=tmpdir,
            )

            model_result = result['symbols_results']['SYM=X']['models']['patchtst_sklearn']
            self.assertIn('error', model_result)

    @patch.object(ModelComparer, '_init_predictors')
    def test_predict_only_no_training(self, mock_init):
        """predict_only 不執行訓練"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_predictor = MagicMock()
            mock_predictor.prepare_training_data.return_value = (
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )

            mock_init.return_value = {'patchtst_sklearn': mock_predictor}

            comparer = ModelComparer.__new__(ModelComparer)
            comparer.config = {}
            comparer.output_dir = Path(tmpdir)
            comparer.model_names = ['patchtst_sklearn']
            comparer.predictors = mock_init.return_value

            comparer.predict_only(['SYM=X'], model_dir=tmpdir)

            mock_predictor.train_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
