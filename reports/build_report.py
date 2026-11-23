"""Build the final assessment PDF from measured results and the supplied chart.

Run after evaluate.py, train_predict.py, and score.py. The builder does not fit
models, predict rates, modify inputs, or substitute earlier benchmark scores for
the production backtest. Requires reportlab (declared in requirements-solution.txt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#132D46")
TEAL = colors.HexColor("#087E8B")
INK = colors.HexColor("#24394B")
MUTED = colors.HexColor("#52677A")
PALE = colors.HexColor("#EEF5F7")
LINE = colors.HexColor("#D7E1E8")
WIDTH, HEIGHT = 595.276, 841.89
CONTENT = WIDTH - 92


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def money(value):
    return f"${float(value):,.2f}"


def metric_for(metrics, split, model):
    return next(row for row in metrics if row["split"] == split and row["model"] == model)


def pooled(metrics, model):
    rows = [metric_for(metrics, s, model) for s in
            ("chronological_jul_aug", "chronological_sep_oct")]
    n = sum(row["n"] for row in rows)
    return sum(row["mae"] * row["n"] for row in rows) / n


def install_fonts():
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportBody", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            pdfmetrics.registerFontFamily("ReportBody", normal="ReportBody", bold="ReportBold")
            return "ReportBody", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


def build(root: Path, output: Path):
    evaluation_path = root / "evaluation/production_results.json"
    chart_path = root / "scorer_results/candidate_december.png"
    for path in (evaluation_path, chart_path, root / "validation_predictions.csv",
                 root / "december_predictions.csv", root / "models/metadata.json",
                 root / "evaluation/production_error_slices.csv"):
        if not path.exists():
            raise FileNotFoundError(f"Generate the required artifact first: {path}")
    evaluation = read_json(evaluation_path)
    metrics = evaluation["metrics"]
    metadata = read_json(root / "models/metadata.json")
    for branch, feature_count in (("main", 25), ("chart", 31)):
        saved = metadata[branch]
        if len(saved["feature_columns"]) != feature_count:
            raise ValueError(f"Unexpected saved {branch} feature contract")
        training = saved["training"]
        if training["rows"] != 48000 or training["validation_used_for_training"]:
            raise ValueError(f"Unexpected saved {branch} training scope")
    main_rows = read_csv(root / "validation_predictions.csv")
    chart_rows = read_csv(root / "december_predictions.csv")
    main_rates = [float(row["predicted_rate"]) for row in main_rows]
    chart_rates = [float(row["predicted_rate"]) for row in chart_rows]
    if len(main_rows) != 12000 or len(chart_rows) != 31:
        raise ValueError("The report requires the complete 12,000 and 31 row outputs")
    if not all(math.isfinite(x) and x > 0 for x in main_rates + chart_rates):
        raise ValueError("The report requires finite positive completed predictions")

    source_hashes = read_json(root / "analysis/source_manifest.json")["sha256"]
    originals_ok = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                       for name, digest in source_hashes.items())
    if not originals_ok:
        raise ValueError("An original assessment file differs from the recorded source manifest")
    body_font, bold_font = install_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyReport", fontName=body_font, fontSize=9.2,
                              leading=13.2, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("SmallReport", fontName=body_font, fontSize=8,
                              leading=11.6, textColor=MUTED, spaceAfter=7))
    styles.add(ParagraphStyle("TableReport", fontName=body_font, fontSize=8.2,
                              leading=10.8, textColor=INK))
    styles.add(ParagraphStyle("HeadReport", fontName=bold_font, fontSize=22,
                              leading=27, textColor=NAVY, spaceAfter=15))
    styles.add(ParagraphStyle("SubReport", fontName=bold_font, fontSize=11.6,
                              leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=7))
    styles.add(ParagraphStyle("TitleReport", fontName=bold_font, fontSize=31,
                              leading=37, textColor=NAVY, spaceAfter=16))
    styles.add(ParagraphStyle("KickerReport", fontName=bold_font, fontSize=9,
                              leading=13, textColor=TEAL, spaceAfter=14))

    def p(text, kind="BodyReport"):
        return Paragraph(text, styles[kind])

    def table(headers, rows, widths=None, size=8.2):
        cell_style = ParagraphStyle("cell", parent=styles["TableReport"], fontSize=size)
        head_style = ParagraphStyle("head", parent=cell_style, fontName=bold_font,
                                   textColor=colors.white)
        data = [[Paragraph(escape(str(x)), head_style) for x in headers]]
        data += [[Paragraph(escape(str(x)), cell_style) for x in row] for row in rows]
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
        ]))
        return result

    def page_title(number, title):
        return [p(f"{number:02d}  /  FREIGHT RATE PREDICTION", "KickerReport"), p(title, "HeadReport")]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(46, 42, WIDTH - 46, 42)
        canvas.setFont(body_font, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(46, 29, "Freight Rate Prediction Assessment | Development evidence and final pipeline")
        canvas.drawRightString(WIDTH - 46, 29, f"{document.page}")
        canvas.restoreState()

    document = BaseDocTemplate(str(output), pagesize=(WIDTH, HEIGHT),
                               leftMargin=46, rightMargin=46, topMargin=40,
                               bottomMargin=57, title="Freight Rate Prediction Assessment",
                               author="", subject="Validation, models, and fixed December chart")
    document.addPageTemplates(PageTemplate(
        id="report", frames=[Frame(46, 57, CONTENT, HEIGHT - 97,
                                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
        onPage=footer))
    story = []

    # 1. Decision and assessment contract.
    story += [p("MACHINE LEARNING ENGINEER ASSESSMENT", "KickerReport"),
              p("Freight rate<br/>prediction", "TitleReport"),
              p("Models, validation, and December forecast", "SubReport"),
              p("The goal is to predict total prices in dollars for 12,000 loads in November and "
                "December 2025. The main model uses the supplied market index. The December chart "
                "uses a separate model because its input file has fewer columns.")]
    story += [table(["Prediction path", "Selected estimator", "Two-month pooled MAE"], [
        ["12,000 hidden loads", "Histogram gradient boosting; observed market; no quote",
         money(pooled(metrics, "hgb_market_only"))],
        ["31-row December scenario", "CatBoost; no market or quote",
         money(pooled(metrics, "catboost_universal"))],
    ], [123, 271, 109]), Spacer(1, 12)]
    story += [p("The table reports average dollar error on historical holdouts. The November and "
                "December prices are hidden. The supplied scorer checks the output files and draws "
                "the chart."),
              p("Input files", "SubReport"),
              table(["Information set", "Rows", "Dates / purpose"], [
                  ["train-test.csv", "48,000", "Jan 1-Oct 31, 2025; all supervised fitting"],
                  ["validation.csv", "12,000", "Nov 1-Dec 31, 2025; unlabeled inference only"],
                  ["december-chart-inputs.csv", "31", "Dec 1-31, 2025; fixed daily scenario"],
              ], [161, 54, 288]), Spacer(1, 9),
              p("The output target is posted_rate, exported as predicted_rate. load_id is used "
                "only to align output rows; it is excluded from both estimators.", "SmallReport")]
    story += [PageBreak()]

    # 2. Exploration and quality.
    story += page_title(2, "Data exploration and cleaning")
    story += [p("All 12,000 final loads take place after the last labeled date. They also include "
                "eight cities absent from training. Of those loads, 1,447 (12.06%) have a new pickup "
                "or delivery city. The model can still use the coordinates for those cities."),
              p("The quote signal changes meaning", "SubReport"),
              table(["Labeled months", "Measured relationship to price per mile"], [
                  ["Jan, Feb, Mar, Jun, Sep", "RPM is approximately quote_signal"],
                  ["Apr, May, Jul, Oct", "RPM is approximately 4.15 - quote_signal"],
                  ["August", "No useful relationship of those forms"],
                  ["November-December inputs", "Marginal fingerprints resemble August; target relationship is unverified"],
              ], [167, 336]), Spacer(1, 8),
              p("The correlation between quote_signal and distance was +0.017 in November and "
                "+0.006 in December. In labeled months it was about -0.70 when the signal tracked "
                "price per mile and +0.72 when it was inverted. The near-zero future correlations "
                "look more like August, so both models leave out quote_signal."),
              p("How each model handles data issues", "SubReport"),
              table(["Issue", "Measured evidence", "Implemented treatment"], [
                  ["Missing weight", "300 training / 165 hidden", "Main: native missing values + flag. Chart: training-fold median."],
                  ["Negative weight", "Frequency approximately doubles in hidden inputs", "Main: absolute magnitude + negative flag. Chart: raw weight, matching its tested pipeline."],
                  ["Missing market", "374 training / 249 hidden", "Main: native missing values. Chart path omits this field entirely."],
                  ["Extreme prices", "609 of 43,241 reconstructable rows show large multiplier deviations", "Retain every target in scoring; use absolute-error training loss."],
              ], [96, 145, 262]), Spacer(1, 8),
              p("The price and signal patterns are consistent with synthetic construction or "
                "multiplicative contamination, but this does not prove provenance or causal order. "
                "The market index has strong seasonal and level structure; unobserved future shocks "
                "remain uncertain. No rows are relabeled from a reconstructed formula.", "SmallReport")]
    story += [PageBreak()]

    # 3. Model contracts.
    story += page_title(3, "Models and features")
    story += [p("The main prediction input includes the market index and coordinates for both "
                "cities. The December chart input includes pickup, delivery, distance, equipment, "
                "weight, and date. Its model was trained for those inputs."),
              p("Main loads: histogram gradient boosting", "SubReport"),
              p("HistGradientBoostingRegressor uses absolute_error loss, 150 iterations, learning "
                "rate 0.08, 15 leaves, minimum 40 samples per leaf, L2 regularization 10, no early "
                "stopping, and seed 1729. Its 25 features cover city and equipment categories; "
                "four coordinates; distance, absolute weight, observed market; calendar terms; "
                "weight flags; and selected freight interactions."),
              p("Categories are learned from each training partition. An unseen city becomes a "
                "native missing category while supplied numeric geography remains intact. Numeric "
                "NaNs remain native missing values. ID, quote, route identity, target statistics, "
                "and additional geographic/count features are excluded."),
              p("December chart: CatBoost", "SubReport"),
              p("CatBoostRegressor uses MAE loss, 500 iterations, depth 7, learning rate 0.07, "
                "L2 regularization 8, seed 1729, and no holdout-based early stopping. Its "
                "31-feature universal contract includes city/equipment/ordered-route strings, coordinates, "
                "distance and raw weight, calendar cycles, geographic geometry, and freight "
                "interactions. Market, quote, and every derived pricing-signal product are absent."),
              p("Numeric medians are fitted only on the training partition; categorical values "
                "remain strings. The fixed chart's coordinates are recovered from a training-only "
                "city dictionary with uniqueness assertions. Lexington and Fort Wayne each have "
                "one consistent supplied coordinate pair. No external geocoding is used."),
              p("Why these models", "SubReport"),
              table(["Selection consideration", "Decision"], [
                  ["Future periods and new cities", "HGB performed well on both the forward splits and the held-out city test."],
                  ["Reduced chart schema", "Use the measured universal CatBoost branch; preserve its own raw-weight policy."],
                  ["Complexity versus incremental gain", "Avoid route/count additions to the main HGB for a small and inconsistent gain."],
                  ["Unknown hidden metric", "Report MAE, RMSE, R-squared, and relative errors; do not claim one provably optimal loss."],
              ], [166, 337]), Spacer(1, 8),
              p("Both estimators are finally fitted on all 48,000 January-October labels. "
                "Serialized models and metadata retain feature order, preprocessing state, "
                "configuration, dependency versions, and the static city dictionary.", "SmallReport")]
    story += [PageBreak()]

    # 4. Validation with live production results.
    story += page_title(4, "Training and validation splits")
    story += [p("The main tests train on earlier months and score the next two months, matching "
                "the forecast horizon. Each split fits preprocessing and models on its own "
                "training rows. The random split shows how much easier it is to predict loads "
                "from the same months, and the city holdout tests performance on unfamiliar cities."),
              table(["Split", "Training window", "Evaluation window / rows"], [
                  ["Forward A", "Jan-Jun 2025", "Jul-Aug 2025; 9,671 loads"],
                  ["Forward B", "Jan-Aug 2025", "Sep-Oct 2025; 9,523 loads"],
                  ["Random diagnostic", "38,400 randomly selected rows", "9,600 rows across Jan-Oct; seed 1729"],
                  ["Future city stress", "Jan-Aug; all rows touching four held cities removed", "Sep-Oct loads touching held cities; 1,271 rows"],
              ], [99, 179, 225]), Spacer(1, 10),
              p("Backtest results", "SubReport")]
    split_labels = {
        "chronological_jul_aug": "Jul-Aug",
        "chronological_sep_oct": "Sep-Oct",
        "random_80_20": "Random",
        "unseen_cities_temporal": "Future city",
    }
    metric_rows = []
    for split in split_labels:
        for model, label in (("hgb_market_only", "Main HGB"),
                             ("catboost_universal", "Chart CatBoost")):
            matches = [r for r in metrics if r["split"] == split and r["model"] == model]
            for row in matches:
                metric_rows.append([split_labels[split], label, money(row["mae"]),
                                    money(row["rmse"]), f"{row['r2']:.4f}",
                                    f"{row['mape_pct']:.2f}%"])
    story += [table(["Split", "Model", "MAE", "RMSE", "R-sq.", "MAPE"], metric_rows,
                    [74, 112, 82, 84, 70, 81]), Spacer(1, 8),
              p("The pooled forward MAE weights each holdout by its row count: "
                f"{money(pooled(metrics, 'hgb_market_only'))} for the main model and "
                f"{money(pooled(metrics, 'catboost_universal'))} for the chart model. "
                "This pool describes two nonoverlapping development holdouts; it is not a "
                "confidence interval or an estimate verified against November-December labels."),
              p("How to interpret these results", "SubReport"),
              p("Development experiments compared baselines, ridge regression, and boosted trees, "
                "along with different feature sets, target transformations, and loss functions. "
                "The same historical splits informed model selection and the reported scores. "
                "Actual November and December prices remain hidden."),
              p("The four fixed held cities are Albuquerque, Buffalo, Philadelphia, and Grand "
                "Rapids. Its 1,271 rows include 33 with both endpoints held out; actual hidden "
                "inputs have exactly one new endpoint when a new city occurs. Geographic stress "
                "therefore remains a proxy for the actual eight new cities.", "SmallReport"),
              p("Source: evaluation/production_results.json. All errors include extreme targets; "
                "there is no target-residual filtering or holdout early stopping.", "SmallReport")]
    story += [PageBreak()]

    # 5. Errors and limitations.
    story += page_title(5, "Error analysis and practical limits")
    story += [p("MAE matches the selected robust training loss, while RMSE exposes the effect of "
                "rare extreme labels. Reporting both avoids presenting ordinary-load accuracy as "
                "accuracy on every supplied price. MAPE can be large for very small prices; "
                "dollar error and sample count remain necessary context."),
              p("Main-model September-October diagnostic slices", "SubReport")]
    slices = read_csv(root / "evaluation/production_error_slices.csv")
    selected = []
    wanted = {("month", "2025-09"), ("month", "2025-10"),
              ("equipment", "Dry Van"), ("equipment", "Reefer"),
              ("equipment", "Flatbed"), ("target", "[8000.0, inf)")}
    for row in slices:
        model = row.get("model", row.get("pipeline", ""))
        slice_name = row.get("slice", row.get("slice_type", ""))
        group = row.get("group", row.get("slice_value", ""))
        if row["split"] == "chronological_sep_oct" and "market" in model and (slice_name, group) in wanted:
            selected.append([slice_name, group, f"{int(row['n']):,}",
                             money(row["mae"]), money(row["rmse"])])
    if selected:
        story += [table(["Slice", "Group", "Loads", "MAE", "RMSE"], selected,
                        [78, 135, 67, 111, 112]), Spacer(1, 8),
                  p("These are the production HGB implementation's measured development "
                    "slices from evaluation/production_error_slices.csv. They are not slices "
                    "of the unlabeled final outputs. Target-price buckets are post-outcome "
                    "diagnostics and never model features.", "SmallReport")]
    story += [p("Remaining uncertainty", "SubReport"),
              table(["Limit", "Implication"], [
                  ["One partial year", "January-October cannot identify unseen winter effects or establish repeated annual seasonality."],
                  ["Market shifts", "Observed market helps the main branch; future market shocks are not known for the chart."],
                  ["New city effects", "Numeric coordinates support transfer, but unknown city-specific price effects are unmeasured."],
                  ["Suspected label contamination", "Absolute-error loss favors typical prices; very large true rates can remain underpredicted."],
                  ["Unknown hidden objective", "A hidden squared-error ranking could prefer a different tradeoff from the selected MAE loss."],
                  ["No calibrated intervals", "The required chart shows point predictions. Its decorative fill is not an uncertainty band."],
              ], [133, 370]), Spacer(1, 9),
              p("If more labeled data were available, the next evaluation would reserve a fresh "
                "future period, measure winter and new-city errors separately, and examine market "
                "missingness and extreme-price frequency. This submission uses only the supplied "
                "data and keeps that uncertainty explicit.")]
    story += [PageBreak()]

    # 6. Mandatory actual scorer chart.
    story += page_title(6, "Fixed December prediction chart")
    story += [p("Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb<br/>"
                "December 1-31, 2025; all inputs fixed except the date.")]
    image = Image(str(chart_path), width=CONTENT, height=362, kind="proportional", hAlign="CENTER")
    story += [image, Spacer(1, 9),
              p("The supplied score.py generated this chart from december_predictions.csv.", "SmallReport"),
              table(["Output check", "Measured value"], [
                  ["Rows / distinct dates", f"{len(chart_rows)} / {len({r['date'] for r in chart_rows})}"],
                  ["Minimum / maximum predicted rate", f"{money(min(chart_rates))} / {money(max(chart_rates))}"],
                  ["Mean predicted rate", money(sum(chart_rates) / len(chart_rates))],
                  ["Prediction path", "Universal CatBoost; pricing signals omitted"],
              ], [213, 290]), Spacer(1, 9),
              p("Only the date changes across the 31 rows. The line shows how the model's "
                "calendar features affect its price estimate for this fixed load. Each point "
                "is a single price estimate, and the scorer shades the area beneath the line."),
              p("Main hidden-load output", "SubReport"),
              p(f"The main file contains {len(main_rows):,} unique load IDs in template order, "
                f"with finite positive predictions ranging from {money(min(main_rates))} to "
                f"{money(max(main_rates))}. Both files keep total-load dollar units. "
                "The scorer's successful checks establish schema conformance only.")]

    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    print(f"Created {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports/Freight_Rate_Assessment_Report.pdf"
    build(root, output.resolve())


if __name__ == "__main__":
    main()
