"""
Tests for PDF report generator.

Covers ReportGenerator with pipeline results, comparison results, and chart embedding.
"""

import pytest
from pathlib import Path

fpdf = pytest.importorskip("fpdf")
from currency_predictor.reporting.pdf_generator import ReportGenerator  # noqa: E402


@pytest.fixture
def generator(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path / "reports"))


@pytest.fixture
def pipeline_results():
    """Minimal pipeline execution results."""
    return {
        "success": True,
        "symbols": ["USDTWD=X"],
        "pipeline_status": {
            "data_collection": True,
            "model_training": True,
            "prediction": True,
            "results_saved": True,
        },
        "predictions": [
            {
                "symbol": "USDTWD=X",
                "last_known_value": 30.5,
                "predictions": [30.6, 30.7, 30.65],
            }
        ],
    }


@pytest.fixture
def comparison_results():
    """Multi-model comparison results."""
    return {
        "model_names": ["sklearn", "huggingface"],
        "prediction_horizon": 5,
        "symbols_results": {
            "USDTWD=X": {
                "models": {
                    "sklearn": {
                        "test_metrics": {"rmse": 0.05, "mae": 0.03},
                        "training_time": 1.2,
                    },
                    "huggingface": {
                        "test_metrics": {"rmse": 0.04, "mae": 0.025},
                        "training_time": 12.5,
                    },
                },
                "best_model": "huggingface",
            },
        },
        "overall_ranking": [
            ("huggingface", 0.04),
            ("sklearn", 0.05),
        ],
    }


@pytest.fixture
def failed_prediction_results():
    """Pipeline results with a failed prediction."""
    return {
        "success": False,
        "pipeline_status": {"data_collection": True, "model_training": False},
        "predictions": [
            {"symbol": "EURUSD=X", "error": "Insufficient data"},
        ],
    }


# ------------------------------------------------------------------
# Basic generation
# ------------------------------------------------------------------


class TestGeneratePdf:
    def test_pipeline_report(self, generator, pipeline_results, tmp_path):
        path = generator.generate_pdf(pipeline_results, output_path="test.pdf")
        assert path.exists()
        assert path.suffix == ".pdf"
        # PDF should start with %PDF
        assert path.read_bytes()[:5] == b"%PDF-"

    def test_default_filename(self, generator, pipeline_results):
        path = generator.generate_pdf(pipeline_results)
        assert path.exists()
        assert "prediction_report_" in path.name

    def test_adds_suffix(self, generator, pipeline_results):
        path = generator.generate_pdf(pipeline_results, output_path="no_suffix")
        assert path.suffix == ".pdf"

    def test_creates_output_dir(self, tmp_path, pipeline_results):
        out = tmp_path / "deep" / "nested" / "reports"
        gen = ReportGenerator(output_dir=str(out))
        path = gen.generate_pdf(pipeline_results, output_path="r.pdf")
        assert path.exists()

    def test_failed_prediction(self, generator, failed_prediction_results):
        path = generator.generate_pdf(failed_prediction_results, output_path="fail.pdf")
        assert path.exists()


# ------------------------------------------------------------------
# Comparison results
# ------------------------------------------------------------------


class TestComparisonReport:
    def test_comparison_report(self, generator, comparison_results):
        path = generator.generate_pdf(comparison_results, output_path="compare.pdf")
        assert path.exists()
        assert path.stat().st_size > 100  # non-trivial PDF

    def test_comparison_with_error_symbol(self, generator):
        results = {
            "symbols_results": {
                "BAD": {"error": "No data"},
            },
            "overall_ranking": [],
        }
        path = generator.generate_pdf(results, output_path="err.pdf")
        assert path.exists()

    def test_comparison_with_failed_model(self, generator):
        results = {
            "symbols_results": {
                "USDTWD=X": {
                    "models": {
                        "good": {"test_metrics": {"rmse": 0.01, "mae": 0.01}, "training_time": 1},
                        "bad": {"error": "OOM"},
                    },
                },
            },
            "overall_ranking": [],
        }
        path = generator.generate_pdf(results, output_path="mixed.pdf")
        assert path.exists()


# ------------------------------------------------------------------
# Chart embedding
# ------------------------------------------------------------------


class TestChartEmbedding:
    def test_embed_png(self, generator, pipeline_results, tmp_path):
        # Create a minimal valid PNG (1x1 white pixel)
        import struct, zlib
        def _minimal_png():
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            raw = zlib.compress(b"\x00\xff\xff\xff")
            idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
            idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return sig + ihdr + idat + iend

        chart = tmp_path / "chart.png"
        chart.write_bytes(_minimal_png())

        path = generator.generate_pdf(
            pipeline_results, charts=[str(chart)], output_path="with_chart.pdf"
        )
        assert path.exists()
        assert path.stat().st_size > 200

    def test_missing_chart_skipped(self, generator, pipeline_results):
        path = generator.generate_pdf(
            pipeline_results, charts=["/nonexistent/chart.png"], output_path="skip.pdf"
        )
        assert path.exists()

    def test_unsupported_format_skipped(self, generator, pipeline_results, tmp_path):
        svg = tmp_path / "chart.svg"
        svg.write_text("<svg></svg>")
        path = generator.generate_pdf(
            pipeline_results, charts=[str(svg)], output_path="svg.pdf"
        )
        assert path.exists()


# ------------------------------------------------------------------
# Empty / minimal inputs
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results(self, generator):
        path = generator.generate_pdf({}, output_path="empty.pdf")
        assert path.exists()

    def test_no_predictions(self, generator):
        results = {"success": True, "pipeline_status": {}}
        path = generator.generate_pdf(results, output_path="no_pred.pdf")
        assert path.exists()
