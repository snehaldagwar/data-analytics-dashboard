
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="AML Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

DATA_FILE = DATA_FOLDER / "aml_transactions.csv"


# ------------------------------------------------
# 2. GENERATE SAMPLE TRANSACTION DATASET
# ------------------------------------------------

@st.cache_data
def generate_data():

    np.random.seed(42)

    n = 6000

    amount = np.round(
        np.random.lognormal(7.2, 1.0, n), 2
    )

    hour = np.random.randint(0, 24, n)

    transactions_24h = np.random.poisson(3, n)

    new_beneficiary = np.random.binomial(
        1, 0.14, n
    )

    country_risk = np.random.choice(
        [0, 1, 2],
        n,
        p=[0.78, 0.17, 0.05]
    )

    account_age = np.random.randint(1, 4000, n)

    weekend = np.random.binomial(1, 0.29, n)

    # Generate synthetic suspicious transaction labels

    risk_score = (
        (amount > 4500) * 1.5
        + (hour <= 4) * 0.8
        + (transactions_24h >= 8) * 1.0
        + new_beneficiary * 1.0
        + (country_risk == 2) * 1.2
        + (account_age < 30) * 0.7
        + np.random.normal(0, 0.65, n)
    )

    threshold = np.quantile(risk_score, 0.88)

    suspicious = (
        risk_score >= threshold
    ).astype(int)

    data = pd.DataFrame({

        "Transaction_ID": [
            f"TXN{100001+i}" for i in range(n)
        ],

        "Customer_ID": [
            f"CUST{np.random.randint(1000,9999)}"
            for _ in range(n)
        ],

        "Transaction_Date": (
            pd.Timestamp("2026-10-01")
            + pd.to_timedelta(
                np.random.randint(0, 34, n),
                unit="D"
            )
        ),

        "Amount": amount,
        "Hour": hour,
        "Transactions_Last_24H": transactions_24h,
        "New_Beneficiary": new_beneficiary,
        "Country_Risk": country_risk,
        "Account_Age_Days": account_age,
        "Weekend": weekend,
        "Is_Suspicious": suspicious
    })

    return data


# ------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------

if DATA_FILE.exists():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["Transaction_Date"]
    )

else:

    df = generate_data()

    df.to_csv(DATA_FILE, index=False)


# ------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# ------------------------------------------------

@st.cache_resource
def train_model():

    features = [
        "Amount",
        "Hour",
        "Transactions_Last_24H",
        "New_Beneficiary",
        "Country_Risk",
        "Account_Age_Days",
        "Weekend"
    ]

    X = df[features]
    y = df["Is_Suspicious"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),

        "ROC AUC": roc_auc_score(
            y_test, probabilities
        ),

        "Confusion Matrix": confusion_matrix(
            y_test, predictions
        )
    }

    return model, metrics, features


model, metrics, features = train_model()


# ------------------------------------------------
# 5. PREDICT TRANSACTION RISK
# ------------------------------------------------

df["Fraud_Probability"] = model.predict_proba(
    df[features]
)[:, 1]

df["Predicted_Status"] = (
    df["Fraud_Probability"] >= 0.50
).astype(int)

df["Risk_Category"] = pd.cut(
    df["Fraud_Probability"],
    bins=[-0.001, 0.30, 0.60, 0.80, 1.001],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
).astype(str)


# ------------------------------------------------
# 6. SIDEBAR
# ------------------------------------------------

st.sidebar.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        margin-bottom:25px;
    ">
        <div style="
            background:#2563eb;
            color:white;
            width:48px;
            height:48px;
            border-radius:50%;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:24px;
            font-weight:bold;
        ">S</div>

        <div>
            <h2 style="margin:0;">AML</h2>
            <small>Transaction Monitor</small>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Transactions",
        "Model Insights",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.success("Monitoring system active")

st.sidebar.caption(
    "Portfolio demonstration | Synthetic data"
)


# ------------------------------------------------
# 7. HEADER
# ------------------------------------------------

st.markdown(
    """
    <h1 style="color:#2563eb;">
        AI-Powered AML Transaction Monitoring
    </h1>

    <p>
        Detect suspicious financial activities
        using Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------
# 8. DASHBOARD
# ------------------------------------------------

if page == "Dashboard":

    st.subheader("Financial Transaction Overview")

    total_transactions = len(df)

    suspicious_transactions = int(
        df["Predicted_Status"].sum()
    )

    high_risk = int(
        df["Risk_Category"].isin(
            ["High", "Critical"]
        ).sum()
    )

    alert_rate = (
        suspicious_transactions / total_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Suspicious Transactions",
        f"{suspicious_transactions:,}"
    )

    col3.metric(
        "High Risk Transactions",
        f"{high_risk:,}"
    )

    col4.metric(
        "Alert Rate",
        f"{alert_rate:.2f}%"
    )

    st.divider()

    # Transaction trend

    st.subheader("Transaction Analysis")

    left, right = st.columns(2)

    with left:

        daily = (
            df.groupby(
                df["Transaction_Date"].dt.date
            )
            .agg(
                Transactions=("Transaction_ID", "count"),
                Suspicious=("Predicted_Status", "sum")
            )
            .reset_index()
        )

        fig = px.line(
            daily,
            x="Transaction_Date",
            y=["Transactions", "Suspicious"],
            markers=True,
            title="Daily Transaction Trends"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        risk_data = (
            df["Risk_Category"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
            .reset_index()
        )

        risk_data.columns = [
            "Risk Category",
            "Count"
        ]

        fig = px.pie(
            risk_data,
            names="Risk Category",
            values="Count",
            hole=0.5,
            title="Transaction Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Recent suspicious transactions

    st.subheader("Recent Suspicious Transactions")

    alerts = df[
        df["Predicted_Status"] == 1
    ].sort_values(
        "Fraud_Probability",
        ascending=False
    ).head(15)

    st.dataframe(
        alerts[
            [
                "Transaction_ID",
                "Customer_ID",
                "Amount",
                "Risk_Category",
                "Fraud_Probability",
                "Transaction_Date"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------
# 9. TRANSACTION EXPLORER
# ------------------------------------------------

elif page == "Transactions":

    st.subheader("Transaction Explorer")

    selected_risk = st.multiselect(
        "Select Risk Category",
        ["Low", "Medium", "High", "Critical"],
        default=["High", "Critical"]
    )

    filtered = df[
        df["Risk_Category"].isin(selected_risk)
    ]

    search = st.text_input(
        "Search Transaction ID"
    )

    if search:

        filtered = filtered[
            filtered["Transaction_ID"].str.contains(
                search,
                case=False
            )
        ]

    st.write(
        f"Records found: {len(filtered)}"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Transaction Report",
        data=csv,
        file_name="AML_Transaction_Report.csv",
        mime="text/csv"
    )


# ------------------------------------------------
# 10. MODEL INSIGHTS
# ------------------------------------------------

elif page == "Model Insights":

    st.subheader("Machine Learning Model Performance")

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        [
            ("Accuracy", metrics["Accuracy"]),
            ("Precision", metrics["Precision"]),
            ("Recall", metrics["Recall"]),
            ("F1 Score", metrics["F1 Score"]),
            ("ROC AUC", metrics["ROC AUC"])
        ]
    ):

        col.metric(
            name,
            f"{value:.3f}"
        )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = metrics["Confusion Matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Suspicious"],
        columns=["Predicted Normal", "Predicted Suspicious"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Factors Influencing Model Predictions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------
# 11. ABOUT PROJECT
# ------------------------------------------------

elif page == "About Project":

    st.subheader(
        "AI-Powered AML Transaction Monitoring System"
    )

    st.write(
        """
        This project demonstrates how machine learning
        can help identify potentially suspicious
        financial transactions.

        The application processes transaction data,
        trains a Random Forest classification model,
        predicts transaction risk and displays
        analytical insights through an interactive
        dashboard.
        """
    )

    st.markdown("### Technologies Used")

    st.write(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Plotly
        - Streamlit
        """
    )

    st.warning(
        "This is an educational prototype using "
        "synthetic data. It is not a production "
        "banking fraud detection system."
    )
