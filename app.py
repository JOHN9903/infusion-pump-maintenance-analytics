
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# INFUSION PUMP PREDICTIVE MAINTENANCE DEMONSTRATION
# Research prototype using synthetic data only.
# Not validated for real medical equipment or clinical use.
# ============================================================

APP_TITLE = "Infusion Pump Maintenance Analytics"
RANDOM_SEED = 42

FEATURES = [
    "age_years",
    "monthly_hours",
    "previous_faults_90d",
    "battery_capacity_pct",
    "months_since_service",
]

FEATURE_LABELS = {
    "age_years": "Pump age (years)",
    "monthly_hours": "Monthly operating hours",
    "previous_faults_90d": "Faults in the last 90 days",
    "battery_capacity_pct": "Battery capacity (%)",
    "months_since_service": "Months since last service",
}

DISCLAIMER = (
    "**Research prototype:** All records and predictions in this application "
    "are generated from synthetic data. This model has not been validated "
    "against real infusion-pump failure records. Do not use its outputs to "
    "make clinical decisions, determine equipment safety, or replace "
    "manufacturer instructions and qualified biomedical engineering checks."
)


# ============================================================
# 1. GENERATE SYNTHETIC TRAINING DATA
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

n_training_records = 1500

training_data = pd.DataFrame({
    "age_years": rng.integers(1, 16, n_training_records),
    "monthly_hours": rng.integers(20, 301, n_training_records),
    "previous_faults_90d": rng.integers(0, 6, n_training_records),
    "battery_capacity_pct": rng.integers(50, 101, n_training_records),
    "months_since_service": rng.integers(0, 25, n_training_records),
})

# Illustrative relationship between equipment characteristics and
# a synthetic failure label. This is not a real-world failure equation.
risk_score_train = (
    -3.5
    + 0.18 * training_data["age_years"]
    + 0.004 * training_data["monthly_hours"]
    + 0.35 * training_data["previous_faults_90d"]
    - 0.025 * training_data["battery_capacity_pct"]
    + 0.07 * training_data["months_since_service"]
)

probability_train = 1 / (1 + np.exp(-risk_score_train))

training_data["failed_next_30_days"] = rng.binomial(
    1, probability_train
)

X_train = training_data[FEATURES]
y_train = training_data["failed_next_30_days"]

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
    ),
])

model.fit(X_train, y_train)


# ============================================================
# 2. GENERATE A SYNTHETIC PUMP INVENTORY
# ============================================================

inventory_rng = np.random.default_rng(123)
N_PUMPS = 100

inventory = pd.DataFrame({
    "pump_id": [f"IP-{i:03d}" for i in range(1, N_PUMPS + 1)],
    "age_years": inventory_rng.integers(1, 16, N_PUMPS),
    "monthly_hours": inventory_rng.integers(20, 301, N_PUMPS),
    "previous_faults_90d": inventory_rng.integers(0, 6, N_PUMPS),
    "battery_capacity_pct": inventory_rng.integers(50, 101, N_PUMPS),
    "months_since_service": inventory_rng.integers(0, 25, N_PUMPS),
})

inventory["failure_probability"] = model.predict_proba(
    inventory[FEATURES]
)[:, 1]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def risk_category(probability, threshold=0.30):
    """Assign a descriptive category to a model probability."""
    if probability >= threshold:
        return "High"
    elif probability >= threshold * 0.5:
        return "Medium"
    return "Low"


def prepare_inventory(threshold=0.30, view="All pumps"):
    """Return a sorted inventory with risk categories."""
    df = inventory.copy()

    df["failure_risk_pct"] = (
        df["failure_probability"] * 100
    ).round(1)

    df["risk_category"] = df["failure_probability"].apply(
        lambda p: risk_category(p, threshold)
    )

    df["threshold_flag"] = np.where(
        df["failure_probability"] >= threshold,
        "Above threshold",
        "Below threshold",
    )

    df = df.sort_values(
        "failure_probability", ascending=False
    ).reset_index(drop=True)

    if view == "High risk":
        df = df[df["risk_category"] == "High"]
    elif view == "Medium risk":
        df = df[df["risk_category"] == "Medium"]
    elif view == "Low risk":
        df = df[df["risk_category"] == "Low"]

    return df


def inventory_chart(df):
    """Create a chart of the highest predicted risk scores."""
    fig = px.bar(
        df.head(15).sort_values("failure_risk_pct"),
        x="failure_risk_pct",
        y="pump_id",
        orientation="h",
        title="15 Highest Predicted Risk Scores",
        labels={
            "failure_risk_pct": "Predicted probability (%)",
            "pump_id": "Pump ID",
        },
        hover_data=[
            "age_years",
            "previous_faults_90d",
            "battery_capacity_pct",
            "months_since_service",
        ],
    )

    fig.update_layout(
        height=500,
        xaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=60, b=40),
    )

    return fig


def create_csv_download(df, prefix="pump_inventory"):
    """Write a dataframe to a temporary CSV file for download."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=f"{prefix}_",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name


# ============================================================
# 4. INVENTORY ANALYTICS
# ============================================================

def visual_dashboard(threshold, view):
    df = prepare_inventory(threshold, view)

    all_df = prepare_inventory(threshold, "All pumps")
    total = len(inventory)
    high_count = int((all_df["risk_category"] == "High").sum())
    medium_count = int((all_df["risk_category"] == "Medium").sum())
    low_count = int((all_df["risk_category"] == "Low").sum())
    above_threshold = int(
        (all_df["failure_probability"] >= threshold).sum()
    )

    summary = f"""
### Inventory overview

- **Total synthetic pumps:** {total}
- **High-risk category:** {high_count}
- **Medium-risk category:** {medium_count}
- **Low-risk category:** {low_count}
- **At or above selected threshold ({threshold:.0%}):** {above_threshold}

Showing **{len(df)}** pumps for the selected filter.

{DISCLAIMER}
"""

    display_columns = [
        "pump_id",
        "risk_category",
        "failure_risk_pct",
        "age_years",
        "monthly_hours",
        "previous_faults_90d",
        "battery_capacity_pct",
        "months_since_service",
        "threshold_flag",
    ]

    display_df = df[display_columns].copy()

    fig = inventory_chart(all_df)

    csv_path = create_csv_download(
        display_df, prefix="infusion_pump_inventory"
    )

    return summary, display_df, fig, csv_path


def export_inventory(threshold, view):
    df = prepare_inventory(threshold, view)

    export_columns = [
        "pump_id",
        "risk_category",
        "failure_probability",
        "failure_risk_pct",
        "age_years",
        "monthly_hours",
        "previous_faults_90d",
        "battery_capacity_pct",
        "months_since_service",
        "threshold_flag",
    ]

    return create_csv_download(df[export_columns], prefix="pump_risk_report")


# ============================================================
# 5. INDIVIDUAL PUMP INVESTIGATION
# ============================================================

def investigate_pump(pump_id, threshold):
    if not pump_id:
        return (
            "Select a pump to view its synthetic risk assessment.",
            go.Figure(),
        )

    row = inventory.loc[inventory["pump_id"] == pump_id]

    if row.empty:
        return "Pump ID not found.", go.Figure()

    pump = row.iloc[0]
    probability = float(pump["failure_probability"])
    category = risk_category(probability, threshold)

    features_df = pd.DataFrame(
        [pump[FEATURES].to_dict()]
    )

    # Logistic-regression feature contributions are shown as an
    # illustrative explanation, not as a causal diagnosis.
    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]

    scaled_values = scaler.transform(features_df)
    contributions = scaled_values[0] * classifier.coef_[0]

    explanation_df = pd.DataFrame({
        "Feature": [
            FEATURE_LABELS[feature] for feature in FEATURES
        ],
        "Illustrative model contribution": contributions,
    })

    explanation_df = explanation_df.sort_values(
        "Illustrative model contribution"
    )

    fig = px.bar(
        explanation_df,
        x="Illustrative model contribution",
        y="Feature",
        orientation="h",
        title="Illustrative Feature Contributions",
        labels={
            "Illustrative model contribution": "Scaled model contribution",
            "Feature": "Input feature",
        },
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
    )

    details = f"""
### Pump investigation: {pump_id}

**Predicted probability:** {probability:.1%}

**Category at the selected threshold:** {category}

**Threshold:** {threshold:.0%}

#### Recorded synthetic characteristics

| Characteristic | Value |
|---|---:|
| Pump age | {int(pump["age_years"])} years |
| Monthly operating hours | {int(pump["monthly_hours"])} |
| Faults in previous 90 days | {int(pump["previous_faults_90d"])} |
| Battery capacity | {int(pump["battery_capacity_pct"])}% |
| Months since last service | {int(pump["months_since_service"])} |

#### Suggested research workflow

1. Review the pump's actual service and maintenance history.
2. Check the manufacturer's maintenance instructions and service intervals.
3. Have qualified biomedical engineering personnel perform appropriate inspections and tests.
4. Document verified findings before making maintenance decisions.

The probability is produced by a model trained on synthetic data. It is **not a verified probability of real pump failure**, and the chart does not establish that any feature caused a failure.

{DISCLAIMER}
"""

    return details, fig


def load_selected_pump(pump_id):
    """Return a selected pump ID for the investigation tab."""
    return pump_id


# ============================================================
# 6. GRADIO WEB APPLICATION
# ============================================================

pump_choices = inventory["pump_id"].tolist()

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(),
) as app:

    gr.Markdown(
        f"""
# 🏥 {APP_TITLE}

**Machine learning portfolio prototype for predictive maintenance**

Explore a synthetic infusion-pump inventory, adjust the risk threshold,
investigate individual pumps, and export research reports.

{DISCLAIMER}
"""
    )

    with gr.Row():
        threshold = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.30,
            step=0.05,
            label="Risk threshold",
            info="Used to classify a pump as high risk when its predicted score meets or exceeds this value.",
        )

    with gr.Tabs():

        # ----------------------------------------------------
        # TAB 1: INVENTORY ANALYTICS
        # ----------------------------------------------------
        with gr.Tab("📊 Inventory Analytics"):

            view = gr.Dropdown(
                choices=[
                    "All pumps",
                    "High risk",
                    "Medium risk",
                    "Low risk",
                ],
                value="All pumps",
                label="Inventory filter",
            )

            refresh_button = gr.Button(
                "Refresh dashboard",
                variant="primary",
            )

            summary_output = gr.Markdown()

            inventory_table = gr.Dataframe(
                label="Pump risk inventory",
                interactive=False,
                wrap=True,
            )

            inventory_plot = gr.Plot(
                label="Risk ranking chart"
            )

            with gr.Row():
                export_button = gr.Button(
                    "Export filtered inventory CSV"
                )
                export_file = gr.File(
                    label="Download inventory report"
                )

            selected_inventory_pump = gr.Dropdown(
                choices=pump_choices,
                value=pump_choices[0],
                label="Choose a pump to investigate",
            )

            investigate_button = gr.Button(
                "Open pump investigation",
                variant="secondary",
            )

        # ----------------------------------------------------
        # TAB 2: INDIVIDUAL PUMP INVESTIGATION
        # ----------------------------------------------------
        with gr.Tab("🔍 Pump Investigation"):

            selected_pump = gr.Dropdown(
                choices=pump_choices,
                value=pump_choices[0],
                label="Select pump ID",
            )

            investigate_one_button = gr.Button(
                "Assess selected pump",
                variant="primary",
            )

            investigation_output = gr.Markdown()

            investigation_plot = gr.Plot(
                label="Illustrative feature contributions"
            )

    gr.Markdown(
        """
---
**Project note:** This application demonstrates a machine-learning workflow.
It does not connect to real infusion pumps, hospital records, or monitoring
systems. All generated data are synthetic.
"""
    )

    # Initial dashboard and investigation display
    app.load(
        fn=visual_dashboard,
        inputs=[threshold, view],
        outputs=[
            summary_output,
            inventory_table,
            inventory_plot,
            export_file,
        ],
    )

    app.load(
        fn=investigate_pump,
        inputs=[selected_pump, threshold],
        outputs=[investigation_output, investigation_plot],
    )

    # Refresh the inventory dashboard
    refresh_button.click(
        fn=visual_dashboard,
        inputs=[threshold, view],
        outputs=[
            summary_output,
            inventory_table,
            inventory_plot,
            export_file,
        ],
    )

    # Changing the filter refreshes the inventory view
    view.change(
        fn=visual_dashboard,
        inputs=[threshold, view],
        outputs=[
            summary_output,
            inventory_table,
            inventory_plot,
            export_file,
        ],
    )

    # Changing the threshold refreshes both tabs
    threshold.change(
        fn=visual_dashboard,
        inputs=[threshold, view],
        outputs=[
            summary_output,
            inventory_table,
            inventory_plot,
            export_file,
        ],
    )

    threshold.change(
        fn=investigate_pump,
        inputs=[selected_pump, threshold],
        outputs=[investigation_output, investigation_plot],
    )

    # Export the filtered inventory as a CSV
    export_button.click(
        fn=export_inventory,
        inputs=[threshold, view],
        outputs=[export_file],
    )

    # Update the investigation selector when a pump is chosen
    investigate_button.click(
        fn=load_selected_pump,
        inputs=[selected_inventory_pump],
        outputs=[selected_pump],
    ).then(
        fn=investigate_pump,
        inputs=[selected_pump, threshold],
        outputs=[investigation_output, investigation_plot],
    )

    # Recalculate the individual assessment
    investigate_one_button.click(
        fn=investigate_pump,
        inputs=[selected_pump, threshold],
        outputs=[investigation_output, investigation_plot],
    )

    selected_pump.change(
        fn=investigate_pump,
        inputs=[selected_pump, threshold],
        outputs=[investigation_output, investigation_plot],
    )


if __name__ == "__main__":
    app.launch()
