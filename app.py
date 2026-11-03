"""Loan Agent - Intelligent Loan Approval Prediction Dashboard (Streamlit)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    CAT_COLS,
    LABELS,
    TARGET,
    load_model_and_artifacts,
    load_raw_data,
    predict_applicant,
)

APPROVED_COLOR = "#0F9D8A"
REJECTED_COLOR = "#E4572E"
INK = "#12263F"
COLOR_MAP = {"Approved": APPROVED_COLOR, "Rejected": REJECTED_COLOR}

st.set_page_config(
    page_title="Loan Agent | Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .hero {{background: linear-gradient(120deg, {INK} 0%, #1D4E6B 100%);
            padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;}}
    .hero h1 {{color: #fff; margin: 0; font-size: 1.9rem;}}
    .hero p {{color: #C9D8E6; margin: .3rem 0 0 0;}}
    div[data-testid="stMetric"] {{background: rgba(18,38,63,.05); border: 1px solid rgba(18,38,63,.12);
            padding: .9rem 1rem; border-radius: 12px;}}
    .result-approved {{border-left: 8px solid {APPROVED_COLOR}; background: rgba(15,157,138,.10);
            padding: 1rem 1.3rem; border-radius: 10px;}}
    .result-rejected {{border-left: 8px solid {REJECTED_COLOR}; background: rgba(228,87,46,.10);
            padding: 1rem 1.3rem; border-radius: 10px;}}
    .result-approved h2, .result-rejected h2 {{margin: 0;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model (trains automatically on first run)...")
def get_model():
    return load_model_and_artifacts()


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = load_raw_data()
    df["Status"] = df[TARGET].map(LABELS)
    return df


model, art = get_model()
data = get_data()
m = art["metrics"]

if "history" not in st.session_state:
    st.session_state.history = []


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🏦 Loan Agent")
    st.caption("Intelligent Loan Approval Prediction Dashboard")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Loan Prediction", "Model Performance", "Feature Importance", "Prediction History"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"Random Forest · {len(art['feature_columns'])} features · "
        f"{art['n_train']:,} train / {art['n_test']:,} test rows"
    )
    st.caption(f"Predictions this session: {len(st.session_state.history)}")


# =========================================================================== #
# PAGE 1 - DASHBOARD
# =========================================================================== #
def page_dashboard() -> None:
    hero("Loan Agent – Intelligent Loan Approval Prediction Dashboard",
         "Overview of the loan dataset and the Random Forest model trained on it.")

    approved = int((data[TARGET] == 1).sum())
    rejected = int((data[TARGET] == 0).sum())

    c = st.columns(5)
    c[0].metric("Records analysed", f"{len(data):,}")
    c[1].metric("Model accuracy", f"{m['accuracy']:.2%}")
    c[2].metric("Precision (Approved)", f"{m['precision_approved']:.2f}")
    c[3].metric("Recall (Approved)", f"{m['recall_approved']:.2f}")
    c[4].metric("F1 (Approved)", f"{m['f1_approved']:.2f}")

    left, right = st.columns([1, 1.4])
    with left:
        fig = px.pie(
            names=["Approved", "Rejected"], values=[approved, rejected], hole=0.55,
            color=["Approved", "Rejected"], color_discrete_map=COLOR_MAP,
            title="Approved vs rejected loans (whole dataset)",
        )
        fig.update_traces(textinfo="label+percent+value")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        score_bins = pd.cut(
            data["Credit Score"], bins=[0, 579, 669, 739, 799, 850],
            labels=["<580", "580-669", "670-739", "740-799", "800+"],
        )
        by_score = data.groupby(score_bins, observed=True)[TARGET].mean().reset_index()
        by_score.columns = ["Credit score band", "Approval rate"]
        fig = px.bar(by_score, x="Credit score band", y="Approval rate",
                     title="Approval rate by credit score band", text_auto=".1%",
                     color_discrete_sequence=[APPROVED_COLOR])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig), width="stretch")

    tab_overview, tab_charts, tab_insights = st.tabs(["Dataset overview", "More charts", "Key insights"])

    with tab_overview:
        st.write(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1] - 1} columns "
                 f"(incl. target `{TARGET}`) · **Missing values:** {int(data.drop(columns='Status').isnull().sum().sum())}")
        st.dataframe(data.drop(columns="Status").head(50), width="stretch", height=260)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Column types**")
            st.dataframe(
                pd.DataFrame({"Column": data.drop(columns="Status").columns,
                              "Type": data.drop(columns="Status").dtypes.astype(str).values}),
                width="stretch", hide_index=True,
            )
        with cols[1]:
            st.markdown("**Numeric summary**")
            st.dataframe(data.drop(columns=["Status"]).describe().T.round(2), width="stretch")

    with tab_charts:
        a, b = st.columns(2)
        with a:
            cat = st.selectbox("Approval rate by…", CAT_COLS, index=2,
                               format_func=lambda x: "Home Ownership" if x == "Home Onwership" else x)
            g = data.groupby(cat)[TARGET].agg(["mean", "size"]).reset_index()
            g.columns = [cat, "Approval rate", "Applicants"]
            fig = px.bar(g, x=cat, y="Approval rate", hover_data=["Applicants"], text_auto=".1%",
                         color_discrete_sequence=[APPROVED_COLOR])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 340), width="stretch")
        with b:
            num = st.selectbox("Distribution by status…",
                               ["Loan interest Rate", "Loan Amount", "Credit Score", "Person Income", "Age", "Credit History"])
            clip = data[num].quantile(0.99)
            d = data[data[num] <= clip]
            fig = px.histogram(d, x=num, color="Status", barmode="overlay", opacity=0.65, nbins=40,
                               color_discrete_map=COLOR_MAP)
            fig.update_layout(title=f"{num} (values above the 99th percentile hidden for readability)")
            st.plotly_chart(style(fig, 340), width="stretch")
        corr_cols = ["Age", "Person Income", "Loan Amount", "Loan interest Rate", "Credit History", "Credit Score", TARGET]
        fig = px.imshow(data[corr_cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation of numeric columns with each other and the target")
        st.plotly_chart(style(fig, 460), width="stretch")

    with tab_insights:
        overall = data[TARGET].mean()
        prev = data.groupby("Previous Loan")[TARGET].mean()
        score = data.groupby("Status")["Credit Score"].median()
        rate = data.groupby("Status")["Loan interest Rate"].median()
        inc = data.groupby("Status")["Person Income"].median()
        intent = data.groupby("Loan Intent")[TARGET].mean().sort_values()
        st.markdown(
            f"""
- **Class balance:** {overall:.1%} of the {len(data):,} records are labelled approved, so the data is imbalanced.
- **Previous Loan:** approval rate is **{prev.get('No', float('nan')):.1%}** where `Previous Loan = No` and **{prev.get('Yes', float('nan')):.1%}** where it is `Yes`.
- **Credit score:** median is {score.get('Approved', float('nan')):.0f} for approved vs {score.get('Rejected', float('nan')):.0f} for rejected records.
- **Interest rate:** median is {rate.get('Approved', float('nan')):.2f}% for approved vs {rate.get('Rejected', float('nan')):.2f}% for rejected records.
- **Income:** median is {inc.get('Approved', float('nan')):,.0f} for approved vs {inc.get('Rejected', float('nan')):,.0f} for rejected records.
- **Loan intent:** highest approval rate is `{intent.index[-1]}` ({intent.iloc[-1]:.1%}); lowest is `{intent.index[0]}` ({intent.iloc[0]:.1%}).
- **Data quality note:** the dataset has extreme values (for example Age up to {data['Age'].max()} and Person Income up to {data['Person Income'].max():,}).
            """
        )
        st.caption("These are descriptive patterns in this dataset, not proof that any one factor causes approval or rejection.")


# =========================================================================== #
# PAGE 2 - LOAN PREDICTION
# =========================================================================== #
def page_prediction() -> None:
    hero("Loan Prediction", "Enter an applicant's details to get the model's estimate.")
    st.caption("Fields start at the dataset's median / most common values so you always begin from real data. Change them to describe your applicant.")

    ns = art["numeric_stats"]
    cats = art["categories"]
    modes = art["modes"]

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 20, 100, int(ns["Age"]["median"]), step=1,
                                  help="Training data ranges from 20 to 144; the form is capped at 100.")
            gender = st.selectbox("Gender", cats["Gender"], index=cats["Gender"].index(modes["Gender"]))
            education = st.selectbox("Education", cats["Education"], index=cats["Education"].index(modes["Education"]))
            income = st.number_input("Person Income", int(ns["Person Income"]["min"]), int(ns["Person Income"]["max"]),
                                     int(ns["Person Income"]["median"]), step=1000)
        with c2:
            home = st.selectbox("Home Ownership", cats["Home Onwership"],
                                index=cats["Home Onwership"].index(modes["Home Onwership"]))
            amount = st.number_input("Loan Amount", int(ns["Loan Amount"]["min"]), int(ns["Loan Amount"]["max"]),
                                     int(ns["Loan Amount"]["median"]), step=500)
            intent = st.selectbox("Loan Intent", cats["Loan Intent"],
                                  index=cats["Loan Intent"].index(modes["Loan Intent"]))
            rate = st.number_input("Loan Interest Rate (%)", float(ns["Loan interest Rate"]["min"]),
                                   float(ns["Loan interest Rate"]["max"]), round(ns["Loan interest Rate"]["median"], 2), step=0.1)
        with c3:
            hist = st.number_input("Credit History (years)", int(ns["Credit History"]["min"]),
                                   int(ns["Credit History"]["max"]), int(ns["Credit History"]["median"]), step=1)
            score = st.number_input("Credit Score", int(ns["Credit Score"]["min"]), int(ns["Credit Score"]["max"]),
                                    int(ns["Credit Score"]["median"]), step=1)
            prev = st.selectbox("Previous Loan", cats["Previous Loan"],
                                index=cats["Previous Loan"].index(modes["Previous Loan"]))
        submitted = st.form_submit_button("Predict loan outcome", type="primary")

    if not submitted:
        return

    applicant = {
        "Age": age, "Gender": gender, "Education": education, "Person Income": income,
        "Home Onwership": home, "Loan Amount": amount, "Loan Intent": intent,
        "Loan interest Rate": rate, "Credit History": hist, "Credit Score": score, "Previous Loan": prev,
    }
    result = predict_applicant(model, art, applicant)

    # Warn when an input sits outside the range the model mostly saw in training
    outside = [k for k in ns if not ns[k]["q01"] <= applicant[k] <= ns[k]["q99"]]
    if outside:
        st.warning("These values are unusual compared with the training data (outside its 1st–99th percentile), "
                   "so the model is extrapolating and the estimate is less reliable: " + ", ".join(outside))

    css = "result-approved" if result["label"] == "Approved" else "result-rejected"
    st.markdown(
        f'<div class="{css}"><h2>Predicted outcome: {result["label"]}</h2>'
        f'<p style="margin:.3rem 0 0 0">Approved: {result["p_approved"]:.1%} · Rejected: {result["p_rejected"]:.1%}</p></div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    for lab, p in (("Approved", result["p_approved"]), ("Rejected", result["p_rejected"])):
        fig.add_bar(x=[p], y=["Probability"], orientation="h", name=lab, marker_color=COLOR_MAP[lab],
                    text=[f"{lab} {p:.1%}"], textposition="inside")
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
    st.plotly_chart(style(fig, 160), width="stretch")
    st.info(
        "**This is a machine-learning estimate, not an actual lending decision.** "
        "The probability reflects how similar applicants were labelled in the training dataset. "
        "It should not replace a lender's assessment, credit checks or regulatory requirements."
    )

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **applicant,
        "Prediction": result["label"],
        "P(Approved)": round(result["p_approved"], 4),
        "P(Rejected)": round(result["p_rejected"], 4),
    })
    st.success(f"Saved to Prediction History (session total: {len(st.session_state.history)}).")


# =========================================================================== #
# PAGE 3 - MODEL PERFORMANCE
# =========================================================================== #
def page_performance() -> None:
    hero("Model Performance", f"Evaluated on the held-out 20% test set ({art['n_test']:,} records).")

    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.2%}")
    c[1].metric("Precision (Approved)", f"{m['precision_approved']:.3f}")
    c[2].metric("Recall (Approved)", f"{m['recall_approved']:.3f}")
    c[3].metric("F1-score (Approved)", f"{m['f1_approved']:.3f}")
    c = st.columns(4)
    c[0].metric("Macro precision", f"{m['precision_macro']:.3f}")
    c[1].metric("Macro recall", f"{m['recall_macro']:.3f}")
    c[2].metric("Macro F1", f"{m['f1_macro']:.3f}")
    c[3].metric("ROC AUC", f"{m['roc_auc']:.3f}")
    st.caption(
        f"Precision, recall and F1 are shown for the *Approved* class (class 1). Because most records are rejected, "
        f"always predicting “Rejected” would already score {m['majority_baseline']:.1%} accuracy, so accuracy alone "
        "is not the whole story; recall on approved loans is the weaker number."
    )

    a, b = st.columns(2)
    with a:
        cm = art["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Pred Rejected", "Pred Approved"], y=["Actual Rejected", "Actual Approved"],
                        color_continuous_scale=["#EAF4F2", APPROVED_COLOR], title="Confusion matrix")
        st.plotly_chart(style(fig), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_scatter(x=art["roc"]["fpr"], y=art["roc"]["tpr"], mode="lines", name="Random Forest",
                        line=dict(color=APPROVED_COLOR, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey"))
        fig.update_layout(title=f"ROC curve (AUC = {m['roc_auc']:.3f})",
                          xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(fig), width="stretch")

    a, b = st.columns(2)
    with a:
        rep = art["report"]
        rows = [{"Class": LABELS[int(k)], "Metric": mt, "Score": rep[k][mt.lower()]}
                for k in ("0", "1") for mt in ("Precision", "Recall", "F1-score")]
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Metric", y="Score", color="Class", barmode="group", text_auto=".2f",
                     color_discrete_map=COLOR_MAP, title="Precision / recall / F1 per class")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style(fig), width="stretch")
    with b:
        dist = pd.DataFrame({"P(Approved)": art["y_proba"],
                             "Actual": pd.Series(art["y_test"]).map(LABELS).values})
        fig = px.histogram(dist, x="P(Approved)", color="Actual", nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map=COLOR_MAP, title="Predicted probability of approval, by actual outcome")
        st.plotly_chart(style(fig), width="stretch")

    st.markdown("**Classification report**")
    rep = art["report"]
    table = pd.DataFrame(rep).T
    table.index = [LABELS.get(int(i), i) if str(i).isdigit() else i for i in table.index]
    st.dataframe(table.round(3), width="stretch")
    with st.expander("Plain-text report (same format as the notebook)"):
        st.code(art["report_text"])
    st.caption("Results come from a single 80/20 split with random_state=42, as in the notebook. "
               "No cross-validation was run, so real-world performance may differ.")


# =========================================================================== #
# PAGE 4 - FEATURE IMPORTANCE
# =========================================================================== #
def page_importance() -> None:
    hero("Feature Importance", "Which inputs the Random Forest relied on most.")
    imp = art["importance"].copy()

    c1, c2 = st.columns([2, 1])
    with c1:
        view = st.radio("View", ["Individual features", "Grouped by original column"], horizontal=True)
        kind = st.radio("Method", ["Impurity-based (as in notebook)", "Permutation (test set)"], horizontal=True)
    col = "Importance" if kind.startswith("Impurity") else "Permutation Importance"

    if view.startswith("Grouped"):
        plot_df = imp.groupby("Source Column", as_index=False)[col].sum().rename(columns={"Source Column": "Feature"})
    else:
        plot_df = imp[["Feature", col]]
    plot_df = plot_df.sort_values(col, ascending=False)
    with c2:
        top_n = st.slider("Top N", 3, len(plot_df), min(10, len(plot_df)))

    top = plot_df.head(top_n).iloc[::-1]
    fig = px.bar(top, x=col, y="Feature", orientation="h", color=col,
                 color_continuous_scale=["#CFE9E4", APPROVED_COLOR, INK], title=f"Top {top_n} – {kind}")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style(fig, max(360, 34 * top_n + 90)), width="stretch")

    with st.expander("Full table"):
        st.dataframe(imp.drop(columns=["Source Column"]).round(4), width="stretch", hide_index=True)

    st.markdown(
        """
**What feature importance means.** It summarises how much the trained model used each input when making its
predictions. The impurity-based score (the one your notebook printed) measures how much each feature reduced
node impurity across all trees; permutation importance measures how much test accuracy drops when a feature's
values are shuffled. Larger means the model leaned on it more.

**What it does not mean.**
- It does **not** prove causation: a high-importance feature is associated with the label in this dataset, not shown to cause approval or rejection.
- Correlated features can share or hide importance, and impurity-based scores can favour features with many distinct values.
- One-hot categories (for example `Education_Master`) are measured against a dropped baseline level, so each dummy is only a part of its original column; use the grouped view to see the whole column.
- It describes this model on this data, not how real lenders decide.
        """
    )


# =========================================================================== #
# PAGE 5 - PREDICTION HISTORY
# =========================================================================== #
def page_history() -> None:
    hero("Prediction History", "Predictions made in this browser session (cleared when the session ends).")
    hist = st.session_state.history
    if not hist:
        st.info("No predictions yet. Make one on the **Loan Prediction** page.")
        return

    df = pd.DataFrame(hist)
    c = st.columns(3)
    c[0].metric("Predictions", len(df))
    c[1].metric("Approved", int((df["Prediction"] == "Approved").sum()))
    c[2].metric("Rejected", int((df["Prediction"] == "Rejected").sum()))
    st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)

    a, b, _ = st.columns([1, 1, 4])
    a.download_button("Download CSV", df.to_csv(index=False).encode(), "prediction_history.csv", "text/csv")
    if b.button("Clear history"):
        st.session_state.history = []
        st.rerun()


{
    "Dashboard": page_dashboard,
    "Loan Prediction": page_prediction,
    "Model Performance": page_performance,
    "Feature Importance": page_importance,
    "Prediction History": page_history,
}[page]()
