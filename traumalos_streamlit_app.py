from pathlib import Path
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TRAUMALOS Dynamic Nomogram", layout="wide", initial_sidebar_state="collapsed")

APP_DIR = Path(__file__).resolve().parent

MODEL_CONFIG = {
    "Prolonged ICU LOS": {
        "path": APP_DIR / "model_outputs_DDS_rea_logistic" / "traumalos_dds_rea_logistic_bundle.joblib",
        "short_name": "ICU LOS",
    },
    "Prolonged total hospital LOS": {
        "path": APP_DIR / "model_outputs_DDS_tot_logistic" / "traumalos_dds_tot_logistic_bundle.joblib",
        "short_name": "Total hospital LOS",
    },
}

ISS_COL = "ISS score"
CGS_COL = "CGS score"
CENTER_COL = "Trauma center"
SAPS_COL = "SAPS II"
AIS_HEAD_COL = "AIS head neck"
AIS_CHEST_COL = "AIS chest"
SHOCK_COL = "hemorragic shock"
PROCEDURE_COL = "Surgical and/or hemostatic radiological procedures within first 24 hours"

PROCEDURE_DISPLAY_OPTIONS = [
    "No",
    "Orthopedics",
    "Neurosurgery",
    "Maxillofacial surgery",
    "Spine surgery",
    "Thoracic surgery",
    "Visceral surgery",
    "Urology",
    "Interventional radiology",
    "Vascular surgery",
    "Cardiac surgery",
    "Other operating room procedure",
]

PROCEDURE_KEYWORDS = {
    "No": ["no"],
    "Orthopedics": ["orthoped"],
    "Neurosurgery": ["neurosurg"],
    "Maxillofacial surgery": ["maxillofacial"],
    "Spine surgery": ["spine"],
    "Thoracic surgery": ["thoracic"],
    "Visceral surgery": ["visceral"],
    "Urology": ["urolog"],
    "Interventional radiology": ["interventional radiology"],
    "Vascular surgery": ["vascular"],
    "Cardiac surgery": ["cardiac"],
    "Other operating room procedure": ["other operating room", "other operating"],
}

st.markdown(
    """
    <style>
      .block-container {padding-top:1rem; padding-bottom:2rem; max-width:1800px;}
      h1 {font-size:1.75rem !important; font-weight:500 !important; margin-bottom:.4rem !important;}
      div[data-testid="stMetric"] {background:#fafafa;border:1px solid #e6e6e6;border-radius:6px;padding:.7rem .9rem;}
      div[data-testid="stVerticalBlockBorderWrapper"] {background:#fafafa;}
      .prediction-box {border:1px solid #d9d9d9;border-radius:6px;padding:.9rem 1rem;background:white;margin-bottom:.8rem;}
      .prediction-main {font-size:1.65rem;font-weight:600;margin:0;}
      .prediction-ci {font-size:1rem;margin-top:.25rem;color:#444;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_bundle(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found:\n{path}")
    return joblib.load(path)

def normalize_text(value):
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)

def get_training_categories(bundle):
    base_model = bundle["base_model"]
    preprocessor = base_model.named_steps["preprocess"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["onehot"]
    categorical_cols = list(bundle["categorical_cols"])
    return {
        col: list(categories)
        for col, categories in zip(categorical_cols, encoder.categories_)
    }

def resolve_trauma_center(bundle, selected_center):
    categories = get_training_categories(bundle)
    if CENTER_COL not in categories:
        return selected_center
    target = normalize_text(selected_center)
    for raw in categories[CENTER_COL]:
        if normalize_text(raw) == target:
            return raw
    try:
        target_float = float(selected_center)
        for raw in categories[CENTER_COL]:
            try:
                if float(raw) == target_float:
                    return raw
            except (TypeError, ValueError):
                pass
    except (TypeError, ValueError):
        pass
    raise ValueError(
        f"Trauma center {selected_center} was not found in training categories: "
        f"{categories[CENTER_COL]}"
    )

def resolve_procedure_category(bundle, display_value):
    categories = get_training_categories(bundle)
    if PROCEDURE_COL not in categories:
        raise KeyError(f"{PROCEDURE_COL!r} is not present among model categorical variables.")

    raw_categories = categories[PROCEDURE_COL]
    display_norm = normalize_text(display_value)

    for raw in raw_categories:
        if normalize_text(raw) == display_norm:
            return raw

    candidates = []
    for raw in raw_categories:
        raw_norm = normalize_text(raw)
        if any(keyword in raw_norm for keyword in PROCEDURE_KEYWORDS[display_value]):
            candidates.append(raw)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=lambda x: len(str(x)))[0]

    raise ValueError(
        f"Unable to match '{display_value}' to a model category. "
        f"Training categories: {raw_categories}"
    )

def procedure_mapping_table(bundle):
    rows = []
    for display_value in PROCEDURE_DISPLAY_OPTIONS:
        try:
            raw = resolve_procedure_category(bundle, display_value)
        except Exception as exc:
            raw = f"NOT MATCHED — {exc}"
        rows.append({"Interface label": display_value, "Model category": raw})
    return pd.DataFrame(rows)

def predict_with_bootstrap_ci(bundle, patient_df, ci_level=0.95):
    features = bundle["features"]
    missing = [c for c in features if c not in patient_df.columns]
    if missing:
        raise KeyError(f"Missing variables: {missing}")

    X_new = patient_df[features].copy()

    # IMPORTANT:
    # During model training, categorical variables were explicitly cast
    # to pandas dtype "object". Streamlit otherwise creates Trauma center
    # as an integer, which can trigger a dtype mismatch in OneHotEncoder
    # (notably with recent scikit-learn versions).
    for col in bundle.get("categorical_cols", [CENTER_COL, PROCEDURE_COL]):
        if col in X_new.columns:
            X_new[col] = X_new[col].astype("object")

    # Binary variable must stay numeric (0/1).
    if SHOCK_COL in X_new.columns:
        X_new[SHOCK_COL] = pd.to_numeric(
            X_new[SHOCK_COL],
            errors="raise"
        ).astype("float64")

    # Continuous/ordinal numerical variables are also made explicit.
    for col in [ISS_COL, CGS_COL, SAPS_COL, AIS_HEAD_COL, AIS_CHEST_COL]:
        if col in X_new.columns:
            X_new[col] = pd.to_numeric(
                X_new[col],
                errors="raise"
            ).astype("float64")

    point = bundle["final_calibrated_model"].predict_proba(X_new)[:, 1]

    bootstrap_models = bundle["bootstrap_models"]
    if not bootstrap_models:
        raise ValueError("No bootstrap models were found in the bundle.")

    bootstrap_probabilities = np.array(
        [m.predict_proba(X_new)[:, 1][0] for m in bootstrap_models],
        dtype=float,
    )

    alpha = 1.0 - ci_level
    lower = np.quantile(bootstrap_probabilities, alpha / 2.0)
    upper = np.quantile(bootstrap_probabilities, 1.0 - alpha / 2.0)

    return {
        "probability": float(point[0]),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_probabilities": bootstrap_probabilities,
    }

def clean_feature_name(name):
    name = str(name)
    for prefix in ("num__", "binary__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

def extract_logistic_coefficients(bundle):
    pipeline = bundle["base_model"]
    preprocessor = pipeline.named_steps["preprocess"]
    logistic = pipeline.named_steps["model"]

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"Feature {i+1}" for i in range(logistic.coef_.shape[1])]

    coef = logistic.coef_.ravel()
    table = pd.DataFrame(
        {
            "Predictor": [clean_feature_name(x) for x in feature_names],
            "Coefficient": coef,
            "Odds ratio": np.exp(np.clip(coef, -709, 709)),
        }
    )

    intercept = pd.DataFrame(
        {
            "Predictor": ["Intercept"],
            "Coefficient": [float(logistic.intercept_[0])],
            "Odds ratio": [float(np.exp(np.clip(logistic.intercept_[0], -709, 709)))],
        }
    )
    return pd.concat([intercept, table], ignore_index=True)

def outcome_definition(bundle):
    label = bundle.get("outcome_label", "")
    threshold = bundle.get("threshold")
    operator = bundle.get("operator", ">=")
    return label if threshold is None else f"{label} ({operator} {threshold} days)"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = {key: [] for key in MODEL_CONFIG}

st.title("TRAUMALOS Dynamic Nomogram")

input_panel, results_panel = st.columns([0.34, 0.66], gap="medium")

with input_panel:
    with st.container(border=True):
        st.markdown("#### Outcome")

        outcome_options = list(MODEL_CONFIG.keys())
        if hasattr(st, "segmented_control"):
            selected_outcome = st.segmented_control(
                "Outcome selection",
                options=outcome_options,
                default=outcome_options[0],
                label_visibility="collapsed",
            )
        else:
            selected_outcome = st.radio(
                "Outcome selection",
                options=outcome_options,
                horizontal=True,
                label_visibility="collapsed",
            )
        if selected_outcome is None:
            selected_outcome = outcome_options[0]

        try:
            bundle = load_bundle(str(MODEL_CONFIG[selected_outcome]["path"]))
        except Exception as exc:
            st.error(f"The selected model could not be loaded.\n\n{exc}")
            st.stop()

        st.markdown("---")

        iss = st.slider("ISS", 1, 75, 16, 1)
        cgs = st.slider("CGS", 3, 15, 15, 1)
        trauma_center_display = st.selectbox("Trauma center", list(range(1, 33)), index=0)
        saps = st.slider("SAPS II", 0, 120, 30, 1)
        ais_head = st.slider("AIS head/neck", 0, 6, 0, 1)
        ais_chest = st.slider("AIS chest", 0, 6, 0, 1)

        hemorrhagic_shock_display = st.selectbox(
            "Hemorrhagic shock", ["No", "Yes"], index=0
        )

        procedure_display = st.selectbox(
            "Surgical/radiological procedure within first 24 hours",
            PROCEDURE_DISPLAY_OPTIONS,
            index=0,
        )

        custom_axis = st.checkbox("Set x-axis ranges", value=False)
        if custom_axis:
            a, b = st.columns(2)
            with a:
                x_min = st.number_input(
                    "Minimum", 0.0, 0.99, 0.0, 0.05, format="%.2f"
                )
            with b:
                x_max = st.number_input(
                    "Maximum", 0.01, 1.0, 1.0, 0.05, format="%.2f"
                )
            if x_min >= x_max:
                st.warning("The maximum must be greater than the minimum.")
        else:
            x_min, x_max = 0.0, 1.0

        predict_clicked = st.button("Predict", type="primary")
        clear_clicked = st.button("Clear prediction history")

        if clear_clicked:
            st.session_state.prediction_history[selected_outcome] = []
            st.rerun()

if predict_clicked:
    try:
        patient = pd.DataFrame(
            [{
                ISS_COL: iss,
                CGS_COL: cgs,
                CENTER_COL: resolve_trauma_center(bundle, trauma_center_display),
                SAPS_COL: saps,
                AIS_HEAD_COL: ais_head,
                AIS_CHEST_COL: ais_chest,
                SHOCK_COL: 1 if hemorrhagic_shock_display == "Yes" else 0,
                PROCEDURE_COL: resolve_procedure_category(bundle, procedure_display),
            }]
        )

        with st.spinner("Computing calibrated probability and bootstrap confidence interval..."):
            pred = predict_with_bootstrap_ci(bundle, patient, ci_level=0.95)

        hist = st.session_state.prediction_history[selected_outcome]
        hist.append(
            {
                "Prediction": len(hist) + 1,
                "ISS": iss,
                "CGS": cgs,
                "Trauma center": trauma_center_display,
                "SAPS II": saps,
                "AIS head/neck": ais_head,
                "AIS chest": ais_chest,
                "Hemorrhagic shock": hemorrhagic_shock_display,
                "Procedure within first 24h": procedure_display,
                "Predicted probability": pred["probability"],
                "95% CI lower": pred["ci_lower"],
                "95% CI upper": pred["ci_upper"],
            }
        )
    except Exception as exc:
        st.error(f"Prediction could not be calculated.\n\n{type(exc).__name__}: {exc}")

with results_panel:
    graphical_tab, numerical_tab, model_tab = st.tabs(
        ["Graphical Summary", "Numerical Summary", "Model Summary"]
    )

    history = st.session_state.prediction_history[selected_outcome]

    with graphical_tab:
        st.markdown("### 95% Confidence Interval for Predicted Probability")

        if not history:
            st.info("Enter the patient characteristics and click **Predict**.")
        else:
            latest = history[-1]
            p = latest["Predicted probability"]
            lo = latest["95% CI lower"]
            hi = latest["95% CI upper"]

            st.markdown(
                f"""
                <div class="prediction-box">
                  <div class="prediction-main">Predicted probability: {100*p:.1f}%</div>
                  <div class="prediction-ci">95% bootstrap confidence interval: {100*lo:.1f}% – {100*hi:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            hdf = pd.DataFrame(history)
            probs = hdf["Predicted probability"].to_numpy()
            lows = hdf["95% CI lower"].to_numpy()
            highs = hdf["95% CI upper"].to_numpy()
            y = np.arange(len(hdf))

            fig, ax = plt.subplots(figsize=(10.5, max(3.2, 0.55 * len(hdf) + 1.8)))
            ax.errorbar(
                probs,
                y,
                xerr=np.vstack([probs - lows, highs - probs]),
                fmt="s",
                markersize=6,
                capsize=4,
                linewidth=1.6,
            )
            ax.set_xlim(float(x_min), float(x_max))
            ax.set_yticks(y)
            ax.set_yticklabels([f"Prediction {int(i)}" for i in hdf["Prediction"]])
            ax.set_xlabel("Probability")
            ax.grid(axis="x", alpha=0.20)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Point estimate: recalibrated logistic regression. "
                "95% CI: percentile interval from the calibrated bootstrap models."
            )

    with numerical_tab:
        st.markdown("### Numerical Summary")

        if not history:
            st.info("No prediction has been generated yet.")
        else:
            display_df = pd.DataFrame(history).copy()
            for col in ["Predicted probability", "95% CI lower", "95% CI upper"]:
                display_df[col] = (100 * display_df[col]).map(lambda x: f"{x:.1f}%")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            latest = history[-1]
            st.markdown("#### Latest prediction")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Item": [
                            "Outcome",
                            "Predicted probability",
                            "95% CI lower bound",
                            "95% CI upper bound",
                        ],
                        "Value": [
                            outcome_definition(bundle),
                            f"{100*latest['Predicted probability']:.2f}%",
                            f"{100*latest['95% CI lower']:.2f}%",
                            f"{100*latest['95% CI upper']:.2f}%",
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with model_tab:
        st.markdown("### Model Summary")

        metrics = bundle.get("test_metrics", {})
        rows = [
            {"Item": "Outcome", "Value": outcome_definition(bundle)},
            {"Item": "Model", "Value": "Penalized logistic regression"},
            {"Item": "Class weighting", "Value": "Balanced"},
            {"Item": "Probability calibration", "Value": "Sigmoid / Platt calibration"},
            {"Item": "Calibration validation", "Value": "5-fold stratified cross-validation"},
            {
                "Item": "Bootstrap confidence interval",
                "Value": f"{len(bundle.get('bootstrap_models', []))} calibrated bootstrap models; percentile 95% CI",
            },
            {
                "Item": "Best development CV ROC-AUC",
                "Value": (
                    f"{bundle.get('best_cv_roc_auc'):.3f}"
                    if bundle.get("best_cv_roc_auc") is not None
                    else "Not available"
                ),
            },
        ]

        if "roc_auc" in metrics:
            rows.append({"Item": "Test ROC-AUC", "Value": f"{metrics['roc_auc']:.3f}"})
        if "brier_score" in metrics:
            rows.append({"Item": "Test Brier score", "Value": f"{metrics['brier_score']:.3f}"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Logistic regression formulation")
        st.latex(
            r"""
            \operatorname{logit}\left[P(Y=1\mid X)\right]
            =
            \beta_0 + \sum_{j=1}^{p}\beta_jX_j
            """
        )

        st.caption(
            "The coefficient table describes the fitted base penalized logistic regression. "
            "The probability presented by the application is subsequently recalibrated."
        )

        try:
            coef_df = extract_logistic_coefficients(bundle)
            coef_df["Coefficient"] = coef_df["Coefficient"].astype(float).round(4)
            coef_df["Odds ratio"] = coef_df["Odds ratio"].astype(float).round(4)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            st.caption(
                "For standardized numerical predictors, coefficients are on the standardized scale. "
                "No inferential p-values are reported because this is a penalized predictive model."
            )
        except Exception as exc:
            st.warning(f"Coefficient table could not be extracted: {exc}")

        with st.expander("Procedure category matching used by the application"):
            st.dataframe(
                procedure_mapping_table(bundle),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Best logistic-regression hyperparameters"):
            params = bundle.get("best_params", {})
            if params:
                st.dataframe(
                    pd.DataFrame(
                        [{"Hyperparameter": k, "Value": str(v)} for k, v in params.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("Hyperparameters are not available in this bundle.")
