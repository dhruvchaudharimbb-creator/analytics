import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CUSTOMER SALES & ANALYTICS DASHBOARD
# Streamlit Cloud friendly - no Plotly dependency required
# ============================================================

st.set_page_config(
    page_title="Customer Sales & Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
    .stApp {
        background: #F4F2ED;
    }

    [data-testid="stSidebar"] {
        background: #202A33;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: white;
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid #DDD9D0;
        border-left: 7px solid #158F8B;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 750;
        color: #17202A;
        margin: 0;
    }

    .hero-sub {
        color: #6B7280;
        margin-top: 6px;
        font-size: 15px;
    }

    .metric-card {
        background: white;
        border: 1px solid #DDD9D0;
        border-radius: 13px;
        padding: 16px;
        min-height: 105px;
    }

    .metric-label {
        color: #73777C;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #17202A;
        font-size: 25px;
        font-weight: 750;
        margin-top: 7px;
    }

    .prediction-card {
        background: #17202A;
        color: white;
        padding: 22px;
        border-radius: 15px;
        border-left: 6px solid #158F8B;
    }

    .prediction-label {
        color: #B8C1C7;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-result {
        font-size: 28px;
        font-weight: 750;
        margin-top: 5px;
    }

    .prediction-confidence {
        color: #B9E5E2;
        margin-top: 5px;
    }

    .insight {
        background: #E5F3F1;
        border: 1px solid #C5E5E1;
        padding: 14px 17px;
        border-radius: 12px;
        color: #17202A;
        margin: 8px 0 18px;
    }

    .footer {
        text-align: center;
        color: #777D82;
        padding: 25px 0 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CSV
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "realistic_e_commerce_sales_data-selected-columns(1).csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    # Fallback for Streamlit Cloud if the CSV is not found beside app.py.
    return None

df = load_data()

if df is None:
    st.error(
        "CSV file not found. Upload the CSV using the sidebar, "
        "or place it in the same GitHub folder as app.py."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload the e-commerce CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)


# ============================================================
# CHECK DATASET
# ============================================================

REQUIRED_COLUMNS = [
    "Customer ID",
    "Gender",
    "Region",
    "Age",
    "Product Name",
    "Category",
    "Unit Price",
    "Quantity",
    "Total Price",
    "Shipping Fee"
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The uploaded CSV is missing: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Numeric cleanup
numeric_columns = [
    "Age",
    "Unit Price",
    "Quantity",
    "Total Price",
    "Shipping Fee"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Region"] = df["Region"].fillna("Unknown")

# Remove unusable rows only if essential fields are missing.
df = df.dropna(
    subset=["Customer ID", "Total Price", "Quantity"]
).copy()

if df.empty:
    st.error("The CSV does not contain usable sales records.")
    st.stop()


# ============================================================
# CUSTOMER-LEVEL ANALYTICS
# ============================================================

customer_df = (
    df.groupby("Customer ID")
    .agg(
        Gender=("Gender", lambda x: x.mode().iloc[0]),
        Region=("Region", lambda x: x.mode().iloc[0]),
        Age=("Age", "median"),
        Orders=("Customer ID", "size"),
        Avg_Unit_Price=("Unit Price", "mean"),
        Avg_Quantity=("Quantity", "mean"),
        Avg_Shipping=("Shipping Fee", "mean"),
        Total_Spend=("Total Price", "sum"),
        Avg_Order_Value=("Total Price", "mean")
    )
    .reset_index()
)

# Customer sales tiers are derived from the actual customer spending.
# Total_Spend is deliberately NOT used as a model feature.
try:
    customer_df["Sales_Tier"] = pd.qcut(
        customer_df["Total_Spend"],
        q=3,
        labels=["Value", "Regular", "Premium"],
        duplicates="drop"
    ).astype(str)
except ValueError:
    customer_df["Sales_Tier"] = "Regular"

# Safety check: ensure there are at least two classes.
if customer_df["Sales_Tier"].nunique() < 2:
    customer_df["Sales_Tier"] = pd.cut(
        customer_df["Total_Spend"],
        bins=3,
        labels=["Value", "Regular", "Premium"],
        include_lowest=True
    ).astype(str)

customer_df["Sales_Tier"] = customer_df["Sales_Tier"].replace(
    {"nan": "Regular"}
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model_df = customer_df.copy()

categorical_columns = ["Gender", "Region"]

encoded_df = pd.get_dummies(
    model_df,
    columns=categorical_columns,
    drop_first=True
)

label_encoder = LabelEncoder()
encoded_df["Tier_Code"] = label_encoder.fit_transform(
    encoded_df["Sales_Tier"]
)

excluded_columns = [
    "Customer ID",
    "Sales_Tier",
    "Tier_Code",
    "Total_Spend"
]

feature_columns = [
    col for col in encoded_df.columns
    if col not in excluded_columns
]

X = encoded_df[feature_columns]
y = encoded_df["Tier_Code"]

# Fill any remaining numerical gaps safely.
X = X.fillna(0)

if y.nunique() >= 2 and len(customer_df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose().round(3)

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_))
    )
