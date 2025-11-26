import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Try to import LightGBM, but don't crash if it's not installed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


st.set_page_config(page_title="ML Model Training UI", layout="wide")

st.title("🧠 ML Model Training UI")
st.write(
    "Upload a CSV, select a model, train, evaluate with metrics, plot results, save the model, and run inference."
)

# --- Session state ---
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "target_cols" not in st.session_state:
    st.session_state.target_cols = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = "Regression"


# ========== 1. Upload CSV ==========
st.sidebar.header("1️⃣ Upload & Basic Settings")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV file from the sidebar to get started.")
    st.stop()


# ========== 2. Problem type, Model & Columns Selection ==========
st.sidebar.header("2️⃣ Problem Setup")

problem_type = st.sidebar.selectbox(
    "Problem type",
    ["Regression", "Classification"],
)

all_columns = df.columns.tolist()

# Model options depend on problem type
if problem_type == "Regression":
    model_name = st.sidebar.selectbox(
        "Choose a model",
        [
            "Linear Regression",
            "Random Forest Regressor",
            "KNN Regressor",
            "LightGBM Regressor" if HAS_LIGHTGBM else "LightGBM Regressor (LightGBM not installed)",
        ],
    )
else:  # Classification
    model_name = st.sidebar.selectbox(
        "Choose a model",
        [
            "Logistic Regression",
            "Random Forest Classifier",
            "KNN Classifier",
            "LightGBM Classifier" if HAS_LIGHTGBM else "LightGBM Classifier (LightGBM not installed)",
        ],
    )

# Features
feature_cols = st.sidebar.multiselect(
    "Select input feature columns (X)",
    options=all_columns,
)

# Targets
if problem_type == "Regression":
    target_cols = st.sidebar.multiselect(
        "Select target column(s) (y)",
        options=[c for c in all_columns if c not in feature_cols] if feature_cols else all_columns,
        help="You can select multiple targets for multi-output regression.",
    )
else:
    # For classification we'll keep it simple: one target column
    target_col_class = st.sidebar.selectbox(
        "Select target column (y)",
        options=[c for c in all_columns if c not in feature_cols] if feature_cols else all_columns,
    )
    target_cols = [target_col_class]

test_size = st.sidebar.slider("Test size (fraction for test set)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# Metrics selection
st.sidebar.header("3️⃣ Evaluation Metrics")
if problem_type == "Regression":
    metric_options = ["MSE", "MAE", "R²"]
else:
    metric_options = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]

selected_metrics = st.sidebar.multiselect(
    "Select evaluation metrics",
    metric_options,
    default=metric_options,
)

st.sidebar.header("4️⃣ Train")
train_button = st.sidebar.button("🚀 Train Model")


# ========== 3. Prepare data & Train ==========
if train_button:
    if not feature_cols:
        st.error("Please select at least one feature column.")
    elif not target_cols:
        st.error("Please select at least one target column.")
    elif "LightGBM" in model_name and not HAS_LIGHTGBM:
        st.error("LightGBM is not installed. Run `pip install lightgbm` and restart.")
    else:
        X = df[feature_cols]
        # Regression: 1 or more targets; Classification: exactly 1
        if problem_type == "Regression":
            if len(target_cols) == 1:
                y = df[target_cols[0]]
            else:
                y = df[target_cols]
        else:
            y = df[target_cols[0]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type == "Classification" else None
        )

        # Preprocessing: numeric + categorical
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Pick model
        if problem_type == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=random_state)
            elif model_name == "KNN Regressor":
                model = KNeighborsRegressor()
            elif "LightGBM" in model_name and HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(random_state=random_state)
            else:
                st.error("Unsupported or unavailable regression model selected.")
                st.stop()
        else:  # Classification
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(random_state=random_state)
            elif model_name == "KNN Classifier":
                model = KNeighborsClassifier()
            elif "LightGBM" in model_name and HAS_LIGHTGBM:
                model = lgb.LGBMClassifier(random_state=random_state)
            else:
                st.error("Unsupported or unavailable classification model selected.")
                st.stop()

        # Pipeline: preprocessor + model
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        with st.spinner("Training model..."):
            pipe.fit(X_train, y_train)

        # Save model and column info in session_state
        st.session_state.trained_model = pipe
        st.session_state.feature_cols = feature_cols
        st.session_state.target_cols = target_cols
        st.session_state.problem_type = problem_type

        # Predictions
        y_pred = pipe.predict(X_test)

        st.subheader("📊 Evaluation Results")

        # ========== 4. Metrics & Plots ==========
        if problem_type == "Regression":
            # Handle single vs multi-output regression
            if isinstance(y_test, pd.Series) or (hasattr(y_test, "ndim") and y_test.ndim == 1):
                # Single target
                if "MSE" in selected_metrics:
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"**MSE:** {mse:.4f}")
                if "MAE" in selected_metrics:
                    mae = mean_absolute_error(y_test, y_pred)
                    st.write(f"**MAE:** {mae:.4f}")
                if "R²" in selected_metrics:
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"**R²:** {r2:.4f}")
            else:
                # Multi-output regression: show per-target table
                metrics_data = []
                for i, col in enumerate(target_cols):
                    yt = y_test.iloc[:, i]
                    yp = y_pred[:, i]
                    row = {"Target": col}
                    if "MSE" in selected_metrics:
                        row["MSE"] = mean_squared_error(yt, yp)
                    if "MAE" in selected_metrics:
                        row["MAE"] = mean_absolute_error(yt, yp)
                    if "R²" in selected_metrics:
                        row["R²"] = r2_score(yt, yp)
                    metrics_data.append(row)
                metrics_df = pd.DataFrame(metrics_data).set_index("Target")
                st.write("**Regression metrics per target:**")
                st.dataframe(metrics_df.style.format("{:.4f}"))

            # Regression plots
            st.subheader("📈 Prediction Plots (Regression)")
            if isinstance(y_test, pd.Series) or (hasattr(y_test, "ndim") and y_test.ndim == 1):
                # Actual vs Predicted
                fig1, ax1 = plt.subplots()
                ax1.scatter(y_test, y_pred, alpha=0.7)
                ax1.set_xlabel("Actual")
                ax1.set_ylabel("Predicted")
                ax1.set_title("Actual vs Predicted")
                st.pyplot(fig1)

                # Residual plot
                residuals = y_test - y_pred
                fig2, ax2 = plt.subplots()
                ax2.scatter(y_pred, residuals, alpha=0.7)
                ax2.axhline(0, linestyle="--")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Residuals (y_true - y_pred)")
                ax2.set_title("Residual Plot")
                st.pyplot(fig2)
            else:
                st.subheader("📈 Prediction Plots (Multi-Output Regression)")

                for i, col in enumerate(target_cols):
                    yt = y_test.iloc[:, i]
                    yp = y_pred[:, i]

                    st.markdown(f"### 🎯 Target: **`{col}`**")

                    # === Actual vs Predicted with Fit Line ===
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(yt, yp, alpha=0.7, label="Data Points")

                    # Best fit line (y = m*x + c)
                    m, c = np.polyfit(yt, yp, 1)
                    ax1.plot(yt, m * yt + c, color="red", linewidth=2, label=f"Fit line: y={m:.2f}x+{c:.2f}")

                    ax1.set_xlabel("Actual")
                    ax1.set_ylabel("Predicted")
                    ax1.set_title(f"Actual vs Predicted — {col}")
                    ax1.legend()
                    st.pyplot(fig1)

                    # === Residual plot ===
                    residuals = yt - yp
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(yp, residuals, alpha=0.7)
                    ax2.axhline(0, linestyle="--", color="red")
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Residuals (y_true − y_pred)")
                    ax2.set_title(f"Residual Plot — {col}")
                    st.pyplot(fig2)



        else:
            # Classification metrics
            if "Accuracy" in selected_metrics:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy:** {acc:.4f}")
            if "Precision (macro)" in selected_metrics:
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                st.write(f"**Precision (macro):** {prec:.4f}")
            if "Recall (macro)" in selected_metrics:
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                st.write(f"**Recall (macro):** {rec:.4f}")
            if "F1-score (macro)" in selected_metrics:
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                st.write(f"**F1-score (macro):** {f1:.4f}")

            # Confusion matrix
            st.subheader("📊 Confusion Matrix")
            classes = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, labels=classes)

            fig_cm, ax_cm = plt.subplots()
            im = ax_cm.imshow(cm)
            ax_cm.figure.colorbar(im, ax=ax_cm)
            ax_cm.set_xticks(np.arange(len(classes)))
            ax_cm.set_yticks(np.arange(len(classes)))
            ax_cm.set_xticklabels(classes, rotation=45, ha="right")
            ax_cm.set_yticklabels(classes)
            ax_cm.set_xlabel("Predicted label")
            ax_cm.set_ylabel("True label")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

        # ========== 5. Save model ==========
        st.subheader("💾 Download Trained Model")

        buffer = BytesIO()
        joblib.dump(
            {
                "pipeline": pipe,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "problem_type": problem_type,
            },
            buffer,
        )
        buffer.seek(0)

        st.download_button(
            label="Download model as .joblib",
            data=buffer,
            file_name="trained_model.joblib",
            mime="application/octet-stream",
        )


# ========== 6. Inference with user input ==========
st.subheader("🔮 Inference: Predict from User Input")

if st.session_state.trained_model is None:
    st.info("Train a model first to enable inference.")
else:
    trained_model = st.session_state.trained_model
    feature_cols = st.session_state.feature_cols
    target_cols = st.session_state.target_cols
    problem_type = st.session_state.problem_type

    st.write("Enter values for each input feature to get a prediction:")

    input_values = {}
    with st.form("inference_form"):
        for col in feature_cols:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                default_val = float(col_data.mean()) if not col_data.isna().all() else 0.0
                input_values[col] = st.number_input(
                    f"{col} (numeric)",
                    value=default_val,
                )
            else:
                unique_vals = col_data.dropna().unique().tolist()
                if 0 < len(unique_vals) <= 50:
                    default_val = unique_vals[0]
                    input_values[col] = st.selectbox(
                        f"{col} (categorical)",
                        options=unique_vals,
                        index=0,
                    )
                else:
                    input_values[col] = st.text_input(
                        f"{col} (text/categorical)",
                        value=str(col_data.dropna().iloc[0]) if col_data.dropna().size > 0 else "",
                    )

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([input_values])
            pred = trained_model.predict(input_df)

            if problem_type == "Regression":
                if len(target_cols) == 1:
                    st.success(f"✅ Predicted {target_cols[0]}: **{float(pred[0]):.4f}**")
                else:
                    st.success("✅ Predicted values:")
                    for i, col in enumerate(target_cols):
                        st.write(f"- **{col}**: {float(pred[0][i]):.4f}")
            else:
                st.success(f"✅ Predicted {target_cols[0]}: **{pred[0]}**")
