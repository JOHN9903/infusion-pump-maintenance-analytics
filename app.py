%%writefile app.py
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==================================================
# 1. DATA AND MODEL INITIALIZATION
# ==================================================

# Set random seed for reproducibility
np.random.seed(42)

# Generate model training dataset (1500 pumps)
n_train = 1500
age_years_train = np.random.uniform(0.5, 10, n_train)
monthly_hours_train = np.random.randint(40, 301, n_train)
previous_faults_train = np.random.poisson(1.2, n_train)
battery_pct_train = np.random.uniform(35, 100, n_train)
months_since_service_train = np.random.randint(0, 19, n_train)

risk_score_train = (
    -3.5
    + 0.18 * age_years_train
    + 0.004 * monthly_hours_train
    + 0.35 * previous_faults_train
    - 0.025 * battery_pct_train
    + 0.07 * months_since_service_train
)
probability_train = 1 / (1 + np.exp(-risk_score_train))
failed_next_30_days = np.random.binomial(1, probability_train)

train_df = pd.DataFrame({
    "age_years": age_years_train.round(1),
    "monthly_hours": monthly_hours_train,
    "previous_faults_90d": previous_faults_train,
    "battery_capacity_pct": battery_pct_train.round(1),
    "months_since_service": months_since_service_train,
    "failed_next_30_days": failed_next_30_days
})

FEATURE_COLUMNS = [
    "age_years",
    "monthly_hours",
    "previous_faults_90d",
    "battery_capacity_pct",
    "months_since_service"
]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["failed_next_30_days"]

# Build and train baseline model
baseline_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
)
baseline_model.fit(X_train, y_train)

# Generate simulated hospital inventory (100 pumps)
np.random.seed(123)
n_pumps = 100
pump_inventory = pd.DataFrame({
    "pump_id": [f"IP-{i:03d}" for i in range(1, n_pumps + 1)],
    "age_years": np.round(np.random.uniform(0.5, 10, n_pumps), 1),
    "monthly_hours": np.random.randint(40, 301, n_pumps),
    "previous_faults_90d": np.random.poisson(1.2, n_pumps),
    "battery_capacity_pct": np.round(
        np.random.uniform(35, 100, n_pumps), 1
    ),
    "months_since_service": np.random.randint(0, 19, n_pumps)
})

pump_ids = sorted(pump_inventory["pump_id"].tolist())
DEFAULT_THRESHOLD = 0.30

# ==================================================
# 2. INVENTORY ANALYTICS
# ==================================================

def get_ranked_inventory(threshold):
    results = pump_inventory.copy()

    results["failure_probability_pct"] = (
        baseline_model.predict_proba(
            results[FEATURE_COLUMNS]
        )[:, 1] * 100
    ).round(1)

    results["alert"] = np.where(
        results["failure_probability_pct"] >= threshold * 100,
        "Review flag",
        "Below threshold"
    )

    results = results.sort_values(
        "failure_probability_pct",
        ascending=False
    ).reset_index(drop=True)

    results.insert(0, "risk_rank", range(1, len(results) + 1))

    return results


def visual_dashboard(threshold, view):
    results = get_ranked_inventory(threshold)

    if view == "Flagged pumps only":
        displayed = results[
            results["alert"] == "Review flag"
        ].copy()
    else:
        displayed = results.copy()

    total = len(results)
    flagged = int((results["alert"] == "Review flag").sum())
    average_risk = results["failure_probability_pct"].mean()

    summary = (
        f"### Inventory Summary\n\n"
        f"- **Total simulated pumps:** {total}\n"
        f"- **Review flags:** {flagged}\n"
        f"- **Below threshold:** {total - flagged}\n"
        f"- **Average estimated probability:** {average_risk:.1f}%\n"
        f"- **Current threshold:** {threshold:.0%}\n\n"
        f"> Synthetic-data research demonstration only. "
        f"Scores are not validated predictions of real pump failures."
    )

    top10 = results.head(10).sort_values(
        "failure_probability_pct"
    )

    bar_chart = px.bar(
        top10,
        x="failure_probability_pct",
        y="pump_id",
        orientation="h",
        title="10 Highest Estimated Failure Probabilities",
        labels={
            "failure_probability_pct": "Estimated probability (%)",
            "pump_id": "Pump ID"
        }
    )

    bar_chart.update_layout(
        xaxis_range=[0, 100],
        height=400
    )

    risk_chart = px.histogram(
        results,
        x="failure_probability_pct",
        nbins=10,
        title="Distribution of Estimated Risk",
        labels={
            "failure_probability_pct": "Estimated probability (%)",
            "count": "Number of pumps"
        }
    )

    risk_chart.add_vline(
        x=threshold * 100,
        line_dash="dash",
        annotation_text="Selected threshold"
    )

    risk_chart.update_layout(height=350)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="pump_risk_report_",
        delete=False,
        encoding="utf-8",
        newline=""
    ) as report:
        displayed.to_csv(report.name, index=False)
        report_path = report.name

    return summary, displayed, bar_chart, risk_chart, report_path


# ==================================================
# 3. INDIVIDUAL PUMP INVESTIGATION
# ==================================================

def investigate_pump(pump_id, threshold):

    if not pump_id:
        return (
            "Select a pump to begin.",
            pd.DataFrame(columns=["Measurement", "Recorded value"]),
            go.Figure(),
            pd.DataFrame(),
            None
        )

    results = get_ranked_inventory(threshold)

    selected_rows = results[
        results["pump_id"] == pump_id
    ]

    if selected_rows.empty:
        return (
            "Pump not found in the inventory.",
            pd.DataFrame(columns=["Measurement", "Recorded value"]),
            go.Figure(),
            pd.DataFrame(),
            None
        )

    pump = selected_rows.iloc[0]
    probability = float(pump["failure_probability_pct"]) / 100
    rank = int(pump["risk_rank"])

    flagged_count = int(
        (results["failure_probability_pct"] >= threshold * 100).sum()
    )

    if probability >= threshold:
        status = "⚠️ FLAGGED FOR MAINTENANCE REVIEW"
        short_status = "Flag for review"
    else:
        status = "✅ BELOW THE SELECTED REVIEW THRESHOLD"
        short_status = "Below threshold"

    summary = f"""
## Pump Investigation: {pump_id}

### {status}

- **Estimated failure probability:** {probability:.1%}
- **Selected review threshold:** {threshold:.0%}
- **Risk ranking:** #{rank} of {len(results)} simulated pumps
- **Pumps at or above threshold:** {flagged_count} of {len(results)}

**Research limitation:** This model uses synthetic data. Its score is
not a verified probability of real equipment failure.
"""

    feature_labels = {
        "age_years": "Pump age (years)",
        "monthly_hours": "Operating hours per month",
        "previous_faults_90d": "Previous faults (90 days)",
        "battery_capacity_pct": "Battery capacity (%)",
        "months_since_service": "Months since last service"
    }

    profile = pd.DataFrame({
        "Measurement": [
            feature_labels[col] for col in FEATURE_COLUMNS
        ],
        "Recorded value": [
            pump[col] for col in FEATURE_COLUMNS
        ]
    })

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Model-estimated risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {
                        "range": [0, threshold * 100],
                        "color": "#dbeafe"
                    },
                    {
                        "range": [threshold * 100, 100],
                        "color": "#fee2e2"
                    }
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100
                }
            }
        )
    )

    gauge.update_layout(
        height=320,
        margin=dict(l=25, r=25, t=65, b=20)
    )

    # Compare the selected pump with nearby-ranked pumps
    start = max(0, rank - 3)
    end = min(len(results), rank + 2)

    comparison = results.iloc[start:end][
        ["risk_rank", "pump_id", "failure_probability_pct", "alert"]
    ].copy()

    comparison = comparison.rename(columns={
        "risk_rank": "Rank",
        "pump_id": "Pump ID",
        "failure_probability_pct": "Estimated risk (%)",
        "alert": "Review status"
    })

    # Create the individual pump report
    report_data = {
        "pump_id": pump_id,
        **{col: pump[col] for col in FEATURE_COLUMNS},
        "estimated_risk_probability": round(probability, 4),
        "selected_threshold": threshold,
        "risk_rank": rank,
        "inventory_size": len(results),
        "review_status": short_status,
        "data_type": "Simulated research data"
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=f"{pump_id}_investigation_",
        delete=False,
        encoding="utf-8",
        newline=""
    ) as report:
        pd.DataFrame([report_data]).to_csv(
            report.name,
            index=False
        )
        report_path = report.name

    return summary, profile, gauge, comparison, report_path


# ==================================================
# 4. CONNECT THE SELECTED PUMP TO THE INVESTIGATION
# ==================================================

def load_selected_pump(pump_id):
    if not pump_id:
        return (
            gr.update(),
            "Please select a pump first."
        )
    return (
        gr.update(value=pump_id),
        f"**{pump_id} selected.** Open the Pump Investigation tab."
    )


def investigate_from_selector(pump_id, threshold):
    return investigate_pump(pump_id, threshold)


def refresh_both_tabs(threshold, view, pump_id):
    analytics = visual_dashboard(threshold, view)
    investigation = investigate_pump(pump_id, threshold)
    return (*analytics, *investigation)


# ==================================================
# 5. BUILD THE INTEGRATED APPLICATION
# ==================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🏥 Infusion Pump Maintenance & Analytics Hub

    Explore the simulated pump inventory and investigate individual
    pumps using a machine-learning research prototype.
    """)

    gr.Markdown(
        "**Research demonstration only:** Synthetic records and model "
        "scores are not validated for real equipment-safety decisions."
    )

    # Shared threshold applies to both tabs
    threshold_control = gr.Slider(
        minimum=0.05,
        maximum=0.80,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        label="System-wide review threshold"
    )

    with gr.Tabs():

        # ------------------------------------------
        # TAB 1: INVENTORY ANALYTICS
        # ------------------------------------------

        with gr.Tab("📊 Inventory Analytics"):

            gr.Markdown("## System-Wide Risk Tracking")

            inventory_filter = gr.Radio(
                choices=["All pumps", "Flagged pumps only"],
                value="All pumps",
                label="Inventory filter"
            )

            refresh_button = gr.Button(
                "Refresh Analytics",
                variant="primary"
            )

            analytics_summary = gr.Markdown()

            inventory_table = gr.Dataframe(
                label="Ranked Pump Inventory",
                interactive=False
            )

            with gr.Row():
                highest_risk_chart = gr.Plot(
                    label="Highest-Risk Pumps"
                )
                distribution_chart = gr.Plot(
                    label="Risk Distribution"
                )

            inventory_download = gr.File(
                label="Download CSV Report"
            )

            gr.Markdown("## Investigate a Pump")

            pump_to_investigate = gr.Dropdown(
                choices=pump_ids,
                value=pump_ids[0] if pump_ids else None,
                label="Choose a pump from the inventory",
                interactive=True
            )

            open_investigation_button = gr.Button(
                "Load Pump into Investigation",
                variant="primary"
            )

            navigation_message = gr.Markdown()

        # ------------------------------------------
        # TAB 2: PUMP INVESTIGATION
        # ------------------------------------------

        with gr.Tab("🔍 Pump Investigation"):

            gr.Markdown("""
            ## Individual Pump Investigation

            Inspect technical measurements, risk score, inventory
            ranking, and an individual CSV report.
            """)

            investigation_selector = gr.Dropdown(
                choices=pump_ids,
                value=pump_ids[0] if pump_ids else None,
                label="Selected pump ID",
                interactive=True
            )

            investigation_summary = gr.Markdown()

            with gr.Row():
                technical_profile = gr.Dataframe(
                    label="Pump Technical Profile",
                    interactive=False
                )

                risk_gauge = gr.Plot(
                    label="Estimated Risk Score"
                )

            nearby_pumps = gr.Dataframe(
                label="Nearby Pumps in Risk Ranking",
                interactive=False
            )

            individual_download = gr.File(
                label="Download Individual Pump Report"
            )

    # ------------------------------------------
    # EVENT CONNECTIONS
    # ------------------------------------------

    analytics_outputs = [
        analytics_summary,
        inventory_table,
        highest_risk_chart,
        distribution_chart,
        inventory_download
    ]

    investigation_outputs = [
        investigation_summary,
        technical_profile,
        risk_gauge,
        nearby_pumps,
        individual_download
    ]

    # Initial display loads
    app.load(
        fn=visual_dashboard,
        inputs=[threshold_control, inventory_filter],
        outputs=analytics_outputs
    )

    app.load(
        fn=investigate_from_selector,
        inputs=[investigation_selector, threshold_control],
        outputs=investigation_outputs
    )

    # Refresh or filter the inventory
    refresh_button.click(
        fn=visual_dashboard,
        inputs=[threshold_control, inventory_filter],
        outputs=analytics_outputs
    )

    inventory_filter.change(
        fn=visual_dashboard,
        inputs=[threshold_control, inventory_filter],
        outputs=analytics_outputs
    )

    # Select a pump in analytics and load its investigation
    open_investigation_button.click(
        fn=load_selected_pump,
        inputs=[pump_to_investigate],
        outputs=[
            investigation_selector,
            navigation_message
        ]
    )

    # Also allow direct selection inside the investigation tab
    investigation_selector.change(
        fn=investigate_from_selector,
        inputs=[investigation_selector, threshold_control],
        outputs=investigation_outputs
    )

    # Single threshold event updating both views
    threshold_control.change(
        fn=refresh_both_tabs,
        inputs=[
            threshold_control,
            inventory_filter,
            investigation_selector
        ],
        outputs=analytics_outputs + investigation_outputs
    )

if __name__ == "__main__":
    app.launch()
