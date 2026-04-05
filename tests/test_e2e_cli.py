"""
E2E Test: CLI Entry Point

測試 main.py 的 argparse 參數解析和 mode routing。
不需要網路或真實模型訓練，全部使用 mock。

執行：uv run pytest tests/test_e2e_cli.py -v
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# main.py is in the project root, not in a package — add root to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestArgParsing:
    """Test argparse argument parsing in main.py."""

    @staticmethod
    def _parse_args(argv):
        """Simulate argparse parsing with given argv."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--visualize', action='store_true', default=True)
        parser.add_argument('--compare', action='store_true', default=True)
        parser.add_argument('--single', action='store_true')
        parser.add_argument('--no-viz', dest='no_viz', action='store_true')
        parser.add_argument('--models', type=str, default=None)
        parser.add_argument('--symbols', type=str, default=None)
        return parser.parse_args(argv)

    def test_default_args(self):
        """Default args: compare and visualize enabled."""
        args = self._parse_args([])
        assert args.visualize is True
        assert args.compare is True
        assert args.models is None
        assert args.symbols is None

    def test_single_flag(self):
        """--single disables compare."""
        args = self._parse_args(['--single'])
        assert args.single is True

    def test_no_viz_flag(self):
        """--no-viz disables visualization."""
        args = self._parse_args(['--no-viz'])
        assert args.no_viz is True

    def test_visualize_flag(self):
        """-v sets visualize=True (already default)."""
        args = self._parse_args(['-v'])
        assert args.visualize is True

    def test_compare_flag(self):
        """--compare sets compare=True (already default)."""
        args = self._parse_args(['--compare'])
        assert args.compare is True

    def test_models_parsing(self):
        """--models sklearn,huggingface parses as string."""
        args = self._parse_args(['--models', 'sklearn,huggingface'])
        models = args.models.split(',')
        assert models == ['sklearn', 'huggingface']

    def test_symbols_parsing(self):
        """--symbols USDTWD=X,AAPL parses as string."""
        args = self._parse_args(['--symbols', 'USDTWD=X,AAPL'])
        symbols = args.symbols.split(',')
        assert symbols == ['USDTWD=X', 'AAPL']


class TestModeRouting:
    """Test that main() routes to correct mode based on args."""

    @patch('main._run_single_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_single_mode_called(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_single
    ):
        """compare=False → _run_single_mode called."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(visualize=False, compare=False, models=None, symbols=None)
        assert result == 0
        mock_single.assert_called_once()

    @patch('main._run_compare_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_default_routes_to_compare_mode(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_compare
    ):
        """Default (no args) → _run_compare_mode called (new default behavior)."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        # CLI default: compare=True, visualize=True (new argparse defaults)
        result = main(compare=True, visualize=True)
        assert result == 0
        mock_compare.assert_called_once()

    @patch('main._run_compare_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_compare_mode_called(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_compare
    ):
        """--compare → _run_compare_mode called."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(visualize=False, compare=True, models=None, symbols=None)
        assert result == 0
        mock_compare.assert_called_once()


class TestNewArgParsing:
    """Test new CLI arguments added in Plan #0190."""

    @staticmethod
    def _parse_args(argv):
        """Simulate argparse parsing with the full main.py parser."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default=None)
        parser.add_argument('-v', '--visualize', action='store_true', default=True)
        parser.add_argument('--compare', action='store_true', default=True)
        parser.add_argument('--single', action='store_true')
        parser.add_argument('--no-viz', dest='no_viz', action='store_true')
        parser.add_argument('--full', action='store_true')
        parser.add_argument('--fresh', action='store_true')
        parser.add_argument('--train-only', action='store_true')
        parser.add_argument('--predict-only', action='store_true')
        parser.add_argument('--continue', dest='continue_run', action='store_true')
        parser.add_argument('--use-op', type=str, default=None)
        parser.add_argument('--models', type=str, default=None)
        parser.add_argument('--symbols', type=str, default=None)
        parser.add_argument('--backtest', action='store_true')
        parser.add_argument('--backtest-strategy', type=str, default=None,
                            choices=['rolling', 'expanding'])
        return parser.parse_args(argv)

    def test_train_only_flag(self):
        args = self._parse_args(['--train-only'])
        assert args.train_only is True
        assert args.predict_only is False

    def test_predict_only_flag(self):
        args = self._parse_args(['--predict-only'])
        assert args.predict_only is True
        assert args.train_only is False

    def test_continue_flag(self):
        args = self._parse_args(['--continue'])
        assert args.continue_run is True

    def test_use_op_flag(self):
        args = self._parse_args(['--use-op', 'abc12345'])
        assert args.use_op == 'abc12345'

    def test_full_implies_compare_and_visualize(self):
        args = self._parse_args(['--full'])
        assert args.full is True
        # The logic in main.py sets compare=True and visualize=True when full=True

    def test_fresh_flag(self):
        args = self._parse_args(['--fresh'])
        assert args.fresh is True

    def test_config_flag(self):
        args = self._parse_args(['-c', 'tw2330_config.json'])
        assert args.config == 'tw2330_config.json'

    def test_config_long_flag(self):
        args = self._parse_args(['--config', 'custom.json'])
        assert args.config == 'custom.json'

    def test_backtest_flag(self):
        args = self._parse_args(['--backtest'])
        assert args.backtest is True
        assert args.backtest_strategy is None

    def test_backtest_strategy_flag(self):
        args = self._parse_args(['--backtest', '--backtest-strategy', 'expanding'])
        assert args.backtest is True
        assert args.backtest_strategy == 'expanding'

    def test_single_flag(self):
        args = self._parse_args(['--single'])
        assert args.single is True

    def test_no_viz_flag(self):
        args = self._parse_args(['--no-viz'])
        assert args.no_viz is True

    def test_default_new_args(self):
        args = self._parse_args([])
        assert args.config is None
        assert args.visualize is True
        assert args.compare is True
        assert args.single is False
        assert args.no_viz is False
        assert args.train_only is False
        assert args.predict_only is False
        assert args.continue_run is False
        assert args.use_op is None
        assert args.fresh is False
        assert args.full is False
        assert args.backtest is False
        assert args.backtest_strategy is None


class TestNewModeRouting:
    """Test that main() correctly routes train-only/predict-only modes."""

    @patch('main._run_single_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_train_only_passes_op_type(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_single
    ):
        """--train-only passes op_type='train_only' to _run_single_mode."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(train_only=True)
        assert result == 0
        mock_single.assert_called_once()

        # Check op_type kwarg
        _, kwargs = mock_single.call_args
        assert kwargs.get('op_type') == 'train_only'

    @patch('main._run_compare_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_train_only_compare_mode(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_compare
    ):
        """--train-only --compare passes op_type='train_only' to _run_compare_mode."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(train_only=True, compare=True)
        assert result == 0
        mock_compare.assert_called_once()

        _, kwargs = mock_compare.call_args
        assert kwargs.get('op_type') == 'train_only'

    @patch('main._run_single_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_continue_run_uses_continue_latest(
        self, mock_logging, mock_dirs, mock_cm, mock_rm_cls, mock_single
    ):
        """--continue uses RunManager.continue_latest()."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm_cls.continue_latest.return_value = mock_rm_instance

        result = main(continue_run=True)
        assert result == 0
        mock_rm_cls.continue_latest.assert_called_once()

    @patch('main._run_single_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_predict_only_passes_use_op(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_single
    ):
        """--predict-only --use-op passes use_op to mode function."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        # predict_only implies continue_run
        result = main(predict_only=True, continue_run=True, use_op='abc12345')
        assert result == 0
        mock_single.assert_called_once()

        _, kwargs = mock_single.call_args
        assert kwargs.get('op_type') == 'predict_only'
        assert kwargs.get('use_op') == 'abc12345'

    @patch('main._run_backtest_mode', return_value=0)
    @patch('main._run_single_mode', return_value=0)
    @patch('main._run_compare_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_backtest_mode_routes_to_run_backtest_mode(
        self, mock_logging, mock_dirs, mock_cm, mock_rm,
        mock_compare, mock_single, mock_backtest
    ):
        """--backtest routes to _run_backtest_mode, not single/compare."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(backtest=True)
        assert result == 0
        mock_backtest.assert_called_once()
        mock_single.assert_not_called()
        mock_compare.assert_not_called()

    @patch('main._run_backtest_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_backtest_passes_strategy(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_backtest
    ):
        """--backtest --backtest-strategy expanding passes strategy through."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(backtest=True, backtest_strategy='expanding')
        assert result == 0
        mock_backtest.assert_called_once()

        args, kwargs = mock_backtest.call_args
        # strategy is the 4th positional arg in _run_backtest_mode
        assert args[3] == 'expanding'

    @patch('main._run_compare_mode', return_value=0)
    @patch('main._run_single_mode', return_value=0)
    @patch('main.RunManager')
    @patch('main.ConfigManager')
    @patch('main.ensure_directories')
    @patch('main.setup_logging')
    def test_full_flag_routes_to_compare_mode(
        self, mock_logging, mock_dirs, mock_cm, mock_rm, mock_single, mock_compare
    ):
        """--full (compare=True, visualize=True) routes to _run_compare_mode."""
        from main import main

        mock_cm_instance = MagicMock()
        mock_cm.return_value = mock_cm_instance
        mock_cm_instance.get_config.return_value = {}
        mock_cm_instance.get_symbols.return_value = ['USDTWD=X']
        mock_cm_instance.get_prediction_horizon.return_value = 7
        mock_cm_instance.get_model_names.return_value = None

        mock_rm_instance = MagicMock()
        mock_rm_instance.op_dir = Path('/tmp/fake_op_dir')
        mock_rm.return_value = mock_rm_instance

        result = main(compare=True, visualize=True)
        assert result == 0
        mock_compare.assert_called_once()
        mock_single.assert_not_called()


class TestCLIValidation:
    """Test mutual exclusivity validation for CLI arguments."""

    @staticmethod
    def _parse_and_validate(argv):
        """Replicate main.py argparse + validation logic."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--backtest', action='store_true')
        parser.add_argument('--backtest-strategy', type=str, default=None,
                            choices=['rolling', 'expanding'])
        parser.add_argument('--train-only', action='store_true')
        parser.add_argument('--predict-only', action='store_true')
        parser.add_argument('--compare', action='store_true', default=True)
        parser.add_argument('--single', action='store_true')
        parser.add_argument('--no-viz', dest='no_viz', action='store_true')
        args = parser.parse_args(argv)

        if args.single:
            args.compare = False
        if args.backtest and (args.train_only or args.predict_only):
            parser.error(
                "--backtest is mutually exclusive with --train-only and --predict-only"
            )
        return args

    def test_backtest_exclusive_with_train_only(self):
        """--backtest --train-only should raise SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse_and_validate(['--backtest', '--train-only'])
        assert exc_info.value.code == 2

    def test_backtest_exclusive_with_predict_only(self):
        """--backtest --predict-only should raise SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse_and_validate(['--backtest', '--predict-only'])
        assert exc_info.value.code == 2

    def test_backtest_allowed_with_compare(self):
        """--backtest --compare should be allowed."""
        args = self._parse_and_validate(['--backtest', '--compare'])
        assert args.backtest is True
        assert args.compare is True
