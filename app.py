"""
Streamlit web entry for RetainPro.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_preprocessing import DataPreprocessor
from micro_analysis import MicroAnalysis
from rfm_analysis import RFMAnalyzer
from single_project_analysis import run_single_project_analysis
from visualization import Visualization


SAMPLE_DATA_PATH = BASE_DIR / "data" / "bank_data.csv"
TARGET_COLUMN = "Exited"


def read_input_bytes(uploaded_file) -> tuple[bytes, str]:
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    return SAMPLE_DATA_PATH.read_bytes(), SAMPLE_DATA_PATH.name


def build_zip(artifacts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in artifacts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def collect_artifacts(work_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    artifacts: dict[str, bytes] = {}
    images: dict[str, bytes] = {}

    for path in sorted((work_dir / "outputs").glob("*")):
        if path.is_file():
            rel_name = f"outputs/{path.name}"
            content = path.read_bytes()
            artifacts[rel_name] = content
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                images[path.name] = content

    for path in sorted((work_dir / "reports").glob("*")):
        if path.is_file():
            artifacts[f"reports/{path.name}"] = path.read_bytes()

    single_report = work_dir / "singleProjectData.txt"
    if single_report.exists():
        artifacts[single_report.name] = single_report.read_bytes()

    return artifacts, images


def prepare_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError("数据中缺少 Exited 列，无法计算 ROC-AUC 和 SHAP。")

    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN]).copy()

    identifier_keywords = ["id", "row", "surname", "name"]
    drop_cols = [
        col for col in x.columns
        if any(keyword in col.lower().replace("_", "") for keyword in identifier_keywords)
    ]
    x = x.drop(columns=drop_cols, errors="ignore")
    x = x.dropna(axis=1, how="all")

    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].fillna(x[col].median())
        else:
            mode = x[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            x[col] = x[col].fillna(fill_value)

    x = pd.get_dummies(x, drop_first=True)
    return x, y


def make_shap_summary_plot(model, x_sample: pd.DataFrame, output_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, x_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_churn_model(df: pd.DataFrame, output_dir: Path) -> dict:
    x, y = prepare_model_features(df)
    if y.nunique() < 2:
        raise ValueError("Exited 只有一个类别，无法计算 ROC-AUC。")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_score = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    roc_path = output_dir / "roc_curve.png"
    plt.figure(figsize=(7.5, 5.4))
    plt.plot(fpr, tpr, color="#0f766e", linewidth=2.4, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn Model ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=220, bbox_inches="tight")
    plt.close()

    importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    importance_path = output_dir / "feature_importance.png"
    top_importances = importances.head(15).sort_values()
    plt.figure(figsize=(8.5, 6))
    top_importances.plot(kind="barh", color="#14b8a6")
    plt.xlabel("Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=220, bbox_inches="tight")
    plt.close()

    shap_path = output_dir / "shap_summary.png"
    shap_sample = x_test.sample(min(len(x_test), 200), random_state=42)
    make_shap_summary_plot(model, shap_sample, shap_path)

    model_report = output_dir / "model_report.txt"
    model_report.write_text(
        "\n".join([
            "流失预测模型评估",
            "=" * 40,
            f"训练样本数: {len(x_train):,}",
            f"测试样本数: {len(x_test):,}",
            f"特征数: {x.shape[1]:,}",
            f"ROC-AUC: {roc_auc:.4f}",
            "",
            "Top 15 特征重要性:",
            *[f"{name}: {value:.4f}" for name, value in importances.head(15).items()],
        ]),
        encoding="utf-8",
    )

    return {
        "roc_auc": roc_auc,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": x.shape[1],
        "top_features": importances.head(15),
    }


def analyze_dataset(data_bytes: bytes, file_name: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="retainpro_") as temp_dir:
        work_dir = Path(temp_dir)
        output_dir = work_dir / "outputs"
        report_dir = work_dir / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        input_path = work_dir / file_name
        input_path.write_bytes(data_bytes)

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            df = pd.read_csv(io.BytesIO(data_bytes))
            model_metrics = run_churn_model(df, output_dir)

            preprocessor = DataPreprocessor()
            preprocessor.original_shape = df.shape
            df_clean = preprocessor.clean_data(df)
            df_final = preprocessor.feature_engineering(df_clean)
            quality_report = preprocessor.get_data_quality_report(df_final)

            rfm_analyzer = RFMAnalyzer(output_dir=str(output_dir))
            df_rfm = rfm_analyzer.create_adaptive_rfm(df_final)
            segment_stats = rfm_analyzer.analyze_segment_performance(df_rfm)
            strategies = rfm_analyzer.get_segment_strategies()

            micro_analyzer = MicroAnalysis()
            micro_results = micro_analyzer.run_all_analysis(df_rfm)

            visualizer = Visualization(output_dir=str(output_dir))
            visualizer.create_all_visualizations(
                df_rfm,
                segment_stats,
                strategies,
                clean_output=False,
            )

            processed_csv = output_dir / "processed_customer_data.csv"
            df_rfm.to_csv(processed_csv, index=False)

            analysis_report = output_dir / "analysis_report.txt"
            if analysis_report.exists():
                (report_dir / "analysis_report.txt").write_text(
                    analysis_report.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            detailed_report = report_dir / "detailed_micro_analysis.txt"
            detailed_report.write_text(
                micro_analyzer.generate_micro_analysis_report(),
                encoding="utf-8",
            )

            run_single_project_analysis(str(processed_csv), str(work_dir / "singleProjectData.txt"))

        artifacts, images = collect_artifacts(work_dir)
        return {
            "data": df_rfm.copy(),
            "quality_report": quality_report,
            "segment_stats": segment_stats.copy(),
            "strategies": strategies,
            "micro_results": micro_results,
            "model_metrics": model_metrics,
            "artifacts": artifacts,
            "images": images,
            "zip": build_zip(artifacts),
            "log": log_buffer.getvalue(),
        }


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def render_metrics(result: dict) -> None:
    df = result["data"]
    segment_rates = df.groupby("Customer_Segment")["Exited"].mean()
    segment_counts = df["Customer_Segment"].value_counts()
    high_risk_segment = segment_rates.idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("客户数", f"{len(df):,}")
    col2.metric("整体流失率", format_percent(df["Exited"].mean()))
    col3.metric("最高风险分群", high_risk_segment)
    col4.metric("最大分群", segment_counts.idxmax())


def render_downloads(result: dict) -> None:
    artifacts = result["artifacts"]
    cols = st.columns(4)
    cols[0].download_button(
        "下载全部",
        data=result["zip"],
        file_name="retainpro_analysis_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )

    if "outputs/processed_customer_data.csv" in artifacts:
        cols[1].download_button(
            "处理数据 CSV",
            data=artifacts["outputs/processed_customer_data.csv"],
            file_name="processed_customer_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if "reports/analysis_report.txt" in artifacts:
        cols[2].download_button(
            "综合报告",
            data=artifacts["reports/analysis_report.txt"],
            file_name="analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if "reports/detailed_micro_analysis.txt" in artifacts:
        cols[3].download_button(
            "微观报告",
            data=artifacts["reports/detailed_micro_analysis.txt"],
            file_name="detailed_micro_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_images(result: dict) -> None:
    preferred_order = [
        "overview_dashboard.png",
        "roc_curve.png",
        "shap_summary.png",
        "feature_importance.png",
        "rfm_analysis.png",
        "correlation_heatmap.png",
        "radar_chart.png",
        "strategy_cards.png",
        "k_selection_plot.png",
    ]
    images = result["images"]
    ordered_names = [name for name in preferred_order if name in images]
    ordered_names += [name for name in images if name not in ordered_names]

    for index in range(0, len(ordered_names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, ordered_names[index:index + 2]):
            col.image(images[name], caption=name, use_container_width=True)


def render_app() -> None:
    st.set_page_config(
        page_title="RetainPro",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        [data-testid="stMetricLabel"] { color: #334155; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 14px;
            background: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            border-color: #0f766e;
            color: #0f766e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RetainPro 客户留存分析")

    with st.sidebar:
        st.subheader("数据")
        uploaded = st.file_uploader("CSV 文件", type=["csv"])
        file_bytes, file_name = read_input_bytes(uploaded)

        if uploaded is None:
            st.caption(f"当前数据：{SAMPLE_DATA_PATH.name}")
        else:
            st.caption(f"当前数据：{file_name}")

        run_clicked = st.button("开始分析", type="primary", use_container_width=True)

        st.divider()
        st.caption("部署入口：`streamlit run app.py`")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if run_clicked:
        try:
            with st.spinner("正在生成分析结果"):
                st.session_state.analysis_result = analyze_dataset(file_bytes, file_name)
        except Exception as exc:
            st.error(f"分析失败：{exc}")
            st.stop()

    result = st.session_state.analysis_result

    if result is None:
        preview_df = pd.read_csv(io.BytesIO(file_bytes)).head(20)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        st.info("点击左侧按钮后生成分群、图表和报告。")
    else:
        render_metrics(result)
        render_downloads(result)

        overview_tab, model_tab, chart_tab, report_tab, data_tab, log_tab = st.tabs(
            ["总览", "模型", "图表", "报告", "数据", "运行日志"]
        )

        with overview_tab:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("客户分群")
                segment_counts = result["data"]["Customer_Segment"].value_counts().rename("客户数")
                st.dataframe(segment_counts, use_container_width=True)
            with col2:
                st.subheader("分群表现")
                st.dataframe(result["segment_stats"], use_container_width=True)

            if "overview_dashboard.png" in result["images"]:
                st.image(
                    result["images"]["overview_dashboard.png"],
                    caption="overview_dashboard.png",
                    use_container_width=True,
                )

        with model_tab:
            metrics = result["model_metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            col2.metric("测试样本", f"{metrics['test_rows']:,}")
            col3.metric("模型特征数", f"{metrics['feature_count']:,}")

            if "roc_curve.png" in result["images"]:
                st.image(
                    result["images"]["roc_curve.png"],
                    caption="ROC 曲线",
                    use_container_width=True,
                )

            left, right = st.columns(2)
            if "shap_summary.png" in result["images"]:
                left.image(
                    result["images"]["shap_summary.png"],
                    caption="SHAP Summary",
                    use_container_width=True,
                )
            if "feature_importance.png" in result["images"]:
                right.image(
                    result["images"]["feature_importance.png"],
                    caption="Feature Importance",
                    use_container_width=True,
                )

            st.subheader("Top 特征")
            st.dataframe(
                metrics["top_features"].rename("importance"),
                use_container_width=True,
            )

        with chart_tab:
            render_images(result)

        with report_tab:
            report_text = result["artifacts"].get("reports/analysis_report.txt", b"").decode(
                "utf-8",
                errors="ignore",
            )
            micro_text = result["artifacts"].get("reports/detailed_micro_analysis.txt", b"").decode(
                "utf-8",
                errors="ignore",
            )
            report_col, micro_col = st.columns(2)
            report_col.text_area("综合报告", report_text, height=520)
            micro_col.text_area("微观报告", micro_text, height=520)

        with data_tab:
            st.subheader("处理后数据")
            st.dataframe(result["data"].head(200), use_container_width=True, hide_index=True)

        with log_tab:
            st.code(result["log"], language="text")


if __name__ == "__main__":
    render_app()
