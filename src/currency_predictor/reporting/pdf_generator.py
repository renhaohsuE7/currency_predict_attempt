"""
PDF Report Generator

Generates PDF prediction reports with text summaries, metrics tables,
and embedded chart images. Requires ``fpdf2`` (install via ``uv sync --extra pdf``).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

logger = logging.getLogger(__name__)


def _require_fpdf() -> None:
    if not HAS_FPDF:
        raise ImportError(
            "fpdf2 is required for PDF report generation. "
            "Install with: uv sync --extra pdf"
        )


class ReportGenerator:
    """Generate PDF reports for currency prediction results."""

    def __init__(self, output_dir: str = "results/reports"):
        _require_fpdf()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf(
        self,
        results: Dict[str, Any],
        charts: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Generate a PDF report from pipeline results.

        Args:
            results: Pipeline execution results dict (from CurrencyPredictor
                     or ModelComparer).
            charts: List of image file paths (PNG/JPG) to embed.
            output_path: Output filename (relative to output_dir). Defaults to
                         ``prediction_report_<timestamp>.pdf``.

        Returns:
            Path to the generated PDF file.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Title page ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 40, "Currency Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", "", 12)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)

        # Overall status
        success = results.get("success", False)
        status_text = "SUCCESS" if success else "FAILED"
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Overall Status: {status_text}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # --- Pipeline stages ---
        pipeline_status = results.get("pipeline_status", {})
        if pipeline_status:
            self._add_section_header(pdf, "Pipeline Stages")
            for stage, ok in pipeline_status.items():
                label = stage.replace("_", " ").title()
                mark = "[OK]" if ok else "[FAIL]"
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 7, f"  {label}: {mark}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

        # --- Predictions summary ---
        predictions = results.get("predictions", [])
        if predictions:
            self._add_section_header(pdf, "Predictions")
            self._add_predictions_table(pdf, predictions)

        # --- Comparison results ---
        symbols_results = results.get("symbols_results", {})
        if symbols_results:
            self._add_section_header(pdf, "Model Comparison")
            self._add_comparison_tables(pdf, results)

        # --- Charts ---
        if charts:
            self._add_section_header(pdf, "Charts")
            for chart_path in charts:
                self._embed_chart(pdf, chart_path)

        # Save
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"prediction_report_{ts}.pdf"

        dest = self.output_dir / output_path
        if not str(dest).endswith(".pdf"):
            dest = dest.with_suffix(".pdf")
        dest.parent.mkdir(parents=True, exist_ok=True)

        pdf.output(str(dest))
        logger.info(f"PDF report saved to {dest}")
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_section_header(pdf: "FPDF", title: str) -> None:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    @staticmethod
    def _add_predictions_table(pdf: "FPDF", predictions: List[Dict[str, Any]]) -> None:
        pdf.set_font("Helvetica", "B", 10)
        col_w = [50, 45, 45, 50]
        headers = ["Symbol", "Last Price", "Predicted", "Change (%)"]
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 8, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        for pred in predictions:
            symbol = str(pred.get("symbol", ""))
            if pred.get("error"):
                pdf.cell(col_w[0], 8, symbol, border=1)
                pdf.cell(sum(col_w[1:]), 8, f"FAILED: {pred['error']}", border=1)
                pdf.ln()
                continue

            last_val = pred.get("last_known_value", 0)
            preds_arr = pred.get("predictions", [])
            first_pred = preds_arr[0] if preds_arr else 0
            change = ((first_pred - last_val) / last_val * 100) if last_val else 0

            pdf.cell(col_w[0], 8, symbol, border=1)
            pdf.cell(col_w[1], 8, f"{last_val:.4f}" if last_val else "-", border=1)
            pdf.cell(col_w[2], 8, f"{first_pred:.4f}" if first_pred else "-", border=1)
            pdf.cell(col_w[3], 8, f"{change:+.2f}%", border=1)
            pdf.ln()

        pdf.ln(5)

    @staticmethod
    def _add_comparison_tables(pdf: "FPDF", results: Dict[str, Any]) -> None:
        symbols_results = results.get("symbols_results", {})

        for symbol, sym_data in symbols_results.items():
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, symbol, new_x="LMARGIN", new_y="NEXT")

            if "error" in sym_data:
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, f"Error: {sym_data['error']}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
                continue

            # Table header
            col_w = [50, 40, 40, 40]
            pdf.set_font("Helvetica", "B", 10)
            for i, h in enumerate(["Model", "RMSE", "MAE", "Time (s)"]):
                pdf.cell(col_w[i], 8, h, border=1)
            pdf.ln()

            pdf.set_font("Helvetica", "", 10)
            models = sym_data.get("models", {})
            for mname, mresult in models.items():
                pdf.cell(col_w[0], 8, mname, border=1)
                if mresult.get("error"):
                    pdf.cell(sum(col_w[1:]), 8, "FAILED", border=1)
                    pdf.ln()
                    continue

                tm = mresult.get("test_metrics", {})
                rmse = tm.get("rmse")
                mae = tm.get("mae")
                t_time = mresult.get("training_time", 0)
                pdf.cell(col_w[1], 8, f"{rmse:.6f}" if isinstance(rmse, (int, float)) else "-", border=1)
                pdf.cell(col_w[2], 8, f"{mae:.6f}" if isinstance(mae, (int, float)) else "-", border=1)
                pdf.cell(col_w[3], 8, f"{t_time:.1f}", border=1)
                pdf.ln()

            best = sym_data.get("best_model")
            if best:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 7, f"Best model: {best}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

        # Overall ranking
        overall_ranking = results.get("overall_ranking", [])
        if overall_ranking:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Overall Ranking", new_x="LMARGIN", new_y="NEXT")

            col_w = [30, 60, 50]
            pdf.set_font("Helvetica", "B", 10)
            for i, h in enumerate(["Rank", "Model", "Avg RMSE"]):
                pdf.cell(col_w[i], 8, h, border=1)
            pdf.ln()

            pdf.set_font("Helvetica", "", 10)
            for rank, (mname, avg_rmse) in enumerate(overall_ranking, 1):
                pdf.cell(col_w[0], 8, f"#{rank}", border=1)
                pdf.cell(col_w[1], 8, mname, border=1)
                pdf.cell(col_w[2], 8, f"{avg_rmse:.6f}", border=1)
                pdf.ln()
            pdf.ln(5)

    @staticmethod
    def _embed_chart(pdf: "FPDF", chart_path: str) -> None:
        path = Path(chart_path)
        if not path.exists():
            logger.warning(f"Chart file not found: {chart_path}")
            return

        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg"):
            logger.warning(f"Unsupported image format: {suffix}")
            return

        # Check if we need a new page (leave 80mm for image)
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()

        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, path.name, new_x="LMARGIN", new_y="NEXT")

        # Fit image to page width (with margins)
        img_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(str(path), w=img_width)
        pdf.ln(5)
