
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Cab Booking Analytics Dashboard",
    page_icon=":taxi:",
    layout="wide",
    initial_sidebar_state="expanded",
)


THEME = {
    "primary": "#0F766E",
    "secondary": "#14B8A6",
    "accent": "#F59E0B",
    "danger": "#DC2626",
    "slate": "#0F172A",
    "muted": "#64748B",
    "bg": "#F5FBFA",
}


def get_data_file() -> str:
    """Prefer cab_data.csv when present, otherwise fall back to final_data.csv in the local folder."""
    if Path("cab_data.csv").exists():
        return "cab_data.csv"
    return "final_data.csv"


DATA_FILE = get_data_file()


def apply_custom_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.10), transparent 24%),
                linear-gradient(180deg, #F9FCFC 0%, {THEME['bg']} 100%);
            color: {THEME['slate']};
        }}
        .block-container {{
            padding-top: 1.05rem;
            padding-bottom: 2rem;
        }}
        .hero {{
            background: linear-gradient(135deg, {THEME['primary']}, #134E4A);
            color: white;
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            box-shadow: 0 18px 34px rgba(15, 118, 110, 0.18);
            margin-bottom: 1rem;
        }}
        .section-title {{
            margin-top: 0.45rem;
            margin-bottom: 0.3rem;
            color: {THEME['slate']};
            font-weight: 700;
        }}
        .sub-card {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15, 23, 42, 0.05);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}
        .insight-box {{
            background: white;
            border-left: 5px solid {THEME['accent']};
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    """Load the local CSV dataset once and cache it for responsive interactions."""
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the cab dataset for EDA, dashboarding, and regression."""
    data = df.copy()

    rename_map = {
        "Customer_ID_x": "Customer_ID",
        "City_x": "City",
        "Ride_Type_Used": "Ride_Type",
        "Vehicle_Type_x": "Vehicle_Type",
        "Final_Fare_INR": "Fare_Amount",
        "Driver_Rating_Given": "Driver_Rating",
    }
    data = data.rename(columns=rename_map)

    data["Start_Time"] = pd.to_datetime(data["Start_Time"], errors="coerce")
    data["End_Time"] = pd.to_datetime(data["End_Time"], errors="coerce")

    numeric_columns = [
        "Distance_KM",
        "Duration_Minutes",
        "Fare_Amount",
        "Fare_Amount_INR",
        "Discount_Amount_INR",
        "Tip_Amount_INR",
        "Surge_Multiplier",
        "Driver_Rating",
        "Customer_Rating_Given",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    fill_map = {
        "Payment_Method": "Unknown",
        "Gender": "Unknown",
        "Ride_Type": "Unknown",
        "Vehicle_Type": "Unknown",
        "City": "Unknown",
        "Trip_Status": "Unknown",
    }
    for column, value in fill_map.items():
        if column in data.columns:
            data[column] = data[column].fillna(value)

    for column in ["Distance_KM", "Duration_Minutes", "Fare_Amount", "Surge_Multiplier", "Driver_Rating"]:
        if column in data.columns:
            data[column] = data[column].fillna(data[column].median())

    data["Trip_Date"] = data["Start_Time"].dt.date
    data["Trip_Month"] = data["Start_Time"].dt.to_period("M").dt.to_timestamp()
    data["Trip_Hour"] = data["Start_Time"].dt.hour.fillna(0).astype(int)
    data["Trip_DayOfWeek"] = data["Start_Time"].dt.dayofweek.fillna(0).astype(int)
    data["Is_Peak_Trip"] = (data["Trip_Hour"].between(8, 11) | data["Trip_Hour"].between(17, 21)).astype(int)
    data["Duplicate_Trip_Record"] = data.duplicated(subset="Trip_ID", keep=False)

    return data


@st.cache_data(show_spinner=False)
def filter_data(
    df: pd.DataFrame,
    city: str,
    ride_types: list[str],
    genders: list[str],
    trip_statuses: list[str],
    vehicle_types: list[str],
    payment_methods: list[str],
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    fare_range: Tuple[float, float],
) -> pd.DataFrame:
    filtered = df.copy()

    if city != "All":
        filtered = filtered[filtered["City"] == city]
    if ride_types:
        filtered = filtered[filtered["Ride_Type"].isin(ride_types)]
    if genders:
        filtered = filtered[filtered["Gender"].isin(genders)]
    if trip_statuses:
        filtered = filtered[filtered["Trip_Status"].isin(trip_statuses)]
    if vehicle_types:
        filtered = filtered[filtered["Vehicle_Type"].isin(vehicle_types)]
    if payment_methods:
        filtered = filtered[filtered["Payment_Method"].isin(payment_methods)]

    filtered = filtered[
        filtered["Fare_Amount"].between(fare_range[0], fare_range[1], inclusive="both")
    ]

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[filtered["Start_Time"].between(start_date, end_date, inclusive="both")]
    return filtered


@st.cache_data(show_spinner=False)
def get_trip_view(df: pd.DataFrame) -> pd.DataFrame:
    """The raw file contains repeated Trip_ID rows from joins, so dashboard KPIs should use one row per trip."""
    return df.sort_values("Start_Time").drop_duplicates(subset="Trip_ID", keep="first")


def previous_period(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    span = max(max_date - min_date, pd.Timedelta(days=1))
    prev_end = min_date - pd.Timedelta(seconds=1)
    prev_start = prev_end - span
    return df[df["Start_Time"].between(prev_start, prev_end, inclusive="both")]


def inr(value: float) -> str:
    if pd.isna(value):
        value = 0
    return f"Rs {value:,.0f}"


def delta_text(current: float, baseline: float) -> str:
    if pd.isna(baseline) or baseline == 0:
        return "N/A"
    delta = ((current - baseline) / baseline) * 100
    return f"{delta:+.1f}% vs previous"


# KPI selection comment:
# These four KPIs were chosen because they summarize demand, customer reach, commercial performance,
# and value-per-trip in one glance, which is ideal for the top section of a professional dashboard.
def create_kpis(filtered_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    total_customers = filtered_df["Customer_ID"].nunique()
    total_rides = filtered_df["Trip_ID"].nunique()
    total_revenue = filtered_df["Fare_Amount"].sum()
    average_fare = filtered_df["Fare_Amount"].mean()

    base_customers = baseline_df["Customer_ID"].nunique()
    base_rides = baseline_df["Trip_ID"].nunique()
    base_revenue = baseline_df["Fare_Amount"].sum()
    base_average_fare = baseline_df["Fare_Amount"].mean()

    cols = st.columns(4)
    metrics = [
        ("Total Customers", f"{total_customers:,}", delta_text(total_customers, base_customers)),
        ("Total Rides", f"{total_rides:,}", delta_text(total_rides, base_rides)),
        ("Total Revenue", inr(total_revenue), delta_text(total_revenue, base_revenue)),
        ("Average Fare", inr(average_fare), delta_text(average_fare, base_average_fare)),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta=delta)


def render_data_overview(data: pd.DataFrame, trip_data: pd.DataFrame) -> None:
    st.markdown("<h3 class='section-title'>Data Overview</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-card'>This EDA mode is intentionally minimal and informative so the user can inspect raw structure, summary statistics, missing values, and basic fare or distance behaviour before moving to the full dashboard.</div>",
        unsafe_allow_html=True,
    )

    row_count = st.slider("Number of rows to display", min_value=5, max_value=100, value=15, step=5)
    st.dataframe(data.head(row_count), use_container_width=True)

    col1, col2 = st.columns([1.35, 1])
    with col1:
        column_info = pd.DataFrame(
            {
                "Column": data.columns,
                "Data Type": [str(dtype) for dtype in data.dtypes],
                "Missing Values": data.isna().sum().values,
                "Unique Values": data.nunique(dropna=True).values,
            }
        )
        st.markdown("<h4 class='section-title'>Column Info</h4>", unsafe_allow_html=True)
        st.dataframe(column_info, use_container_width=True, hide_index=True)
    with col2:
        stats_column = st.selectbox(
            "Select feature for summary statistics",
            ["Fare_Amount", "Distance_KM", "Duration_Minutes", "Surge_Multiplier", "Driver_Rating"],
        )
        series = trip_data[stats_column].dropna()
        summary = pd.DataFrame(
            {
                "Statistic": ["Mean", "Median", "Min", "Max", "Std"],
                "Value": [series.mean(), series.median(), series.min(), series.max(), series.std()],
            }
        )
        st.markdown("<h4 class='section-title'>Data Summary</h4>", unsafe_allow_html=True)
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

    dq1, dq2 = st.columns(2)
    with dq1:
        st.markdown("<h4 class='section-title'>Data Cleaning Info</h4>", unsafe_allow_html=True)
        st.metric("Null Values", f"{int(data.isna().sum().sum()):,}")
        st.metric("Duplicate Trip Records", f"{int(data['Trip_ID'].duplicated().sum()):,}")
        missing_df = data.isna().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False)
        if missing_df.empty:
            st.success("No missing values detected in the current filtered slice.")
        else:
            missing_fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                title="Missing Values by Column",
                color="Missing Values",
                color_continuous_scale=["#99F6E4", THEME["accent"]],
            )
            missing_fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(missing_fig, use_container_width=True)
    with dq2:
        st.markdown("<h4 class='section-title'>Basic Visualizations</h4>", unsafe_allow_html=True)
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            fare_hist = px.histogram(
                trip_data,
                x="Fare_Amount",
                nbins=30,
                title="Fare Distribution",
                color_discrete_sequence=[THEME["secondary"]],
            )
            fare_hist.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fare_hist, use_container_width=True)
        with hist_col2:
            distance_hist = px.histogram(
                trip_data,
                x="Distance_KM",
                nbins=30,
                title="Distance Distribution",
                color_discrete_sequence=[THEME["primary"]],
            )
            distance_hist.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(distance_hist, use_container_width=True)

        box_fig = px.box(
            trip_data,
            x="Ride_Type",
            y="Fare_Amount",
            title="Fare Boxplot by Ride Type",
            color="Ride_Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        box_fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(box_fig, use_container_width=True)


# Visualization choice comment:
# Line charts show change over time clearly, bar charts compare categories effectively, pie charts show share,
# histograms reveal distribution shape, and a heatmap makes correlation patterns easy to scan.
def plot_rides_over_time(df: pd.DataFrame) -> go.Figure:
    trend = df.groupby("Trip_Month", as_index=False).agg(Rides=("Trip_ID", "nunique")).sort_values("Trip_Month")
    fig = px.line(trend, x="Trip_Month", y="Rides", markers=True, title="Rides Over Time")
    fig.update_traces(line_color=THEME["primary"], line_width=3)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_revenue_over_time(df: pd.DataFrame) -> go.Figure:
    trend = df.groupby("Trip_Month", as_index=False).agg(Revenue=("Fare_Amount", "sum")).sort_values("Trip_Month")
    fig = px.line(trend, x="Trip_Month", y="Revenue", markers=True, title="Revenue Over Time")
    fig.update_traces(line_color=THEME["accent"], line_width=3)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_revenue_by_ride_type(df: pd.DataFrame) -> go.Figure:
    chart = df.groupby("Ride_Type", as_index=False).agg(Revenue=("Fare_Amount", "sum")).sort_values("Revenue", ascending=False)
    fig = px.bar(
        chart,
        x="Ride_Type",
        y="Revenue",
        title="Revenue by Ride Type",
        color="Ride_Type",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(template="plotly_white", showlegend=False, height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_rides_by_city(df: pd.DataFrame) -> go.Figure:
    chart = df.groupby("City", as_index=False).agg(Rides=("Trip_ID", "nunique")).sort_values("Rides", ascending=False)
    fig = px.bar(
        chart,
        x="City",
        y="Rides",
        title="Rides by City",
        color="City",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white", showlegend=False, height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_grouped_ride_type_city(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby(["Ride_Type", "City"], as_index=False).agg(Rides=("Trip_ID", "nunique"))
    fig = px.bar(
        grouped,
        x="Ride_Type",
        y="Rides",
        color="City",
        barmode="group",
        title="Ride Type vs City",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_payment_pie(df: pd.DataFrame) -> go.Figure:
    chart = df.groupby("Payment_Method", as_index=False).agg(Rides=("Trip_ID", "nunique"))
    fig = px.pie(
        chart,
        names="Payment_Method",
        values="Rides",
        hole=0.45,
        title="Payment Method Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_fare_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Fare_Amount",
        nbins=30,
        title="Fare Distribution",
        color_discrete_sequence=[THEME["secondary"]],
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr_cols = ["Distance_KM", "Duration_Minutes", "Fare_Amount", "Surge_Multiplier", "Driver_Rating", "Trip_Hour", "Trip_DayOfWeek"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealgrn", title="Correlation Heatmap")
    fig.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=60, b=10))
    return fig


@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame) -> dict:
    """Train a simple linear regression model to predict fare using distance, ride type, city, and time features."""
    model_df = df[["Fare_Amount", "Distance_KM", "Duration_Minutes", "Surge_Multiplier", "Trip_Hour", "Trip_DayOfWeek", "Ride_Type", "City"]].dropna().copy()
    if len(model_df) < 20:
        return {}

    features = pd.get_dummies(
        model_df[["Distance_KM", "Duration_Minutes", "Surge_Multiplier", "Trip_Hour", "Trip_DayOfWeek", "Ride_Type", "City"]],
        columns=["Ride_Type", "City"],
        drop_first=False,
        dtype=float,
    )
    y = model_df["Fare_Amount"].to_numpy(dtype=float)
    x = features.to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    predictions = x_design @ coefficients
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot else 0
    mae = float(np.mean(np.abs(y - predictions)))
    rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
    safe_actual = np.where(y == 0, np.nan, y)
    mape = float(np.nanmean(np.abs((y - predictions) / safe_actual)) * 100)

    equation_terms = [f"{coefficients[0]:.2f}"]
    for feature_name, coefficient in zip(features.columns[:6], coefficients[1:7]):
        cleaned_name = feature_name.replace("_", " ")
        equation_terms.append(f"({coefficient:.2f} x {cleaned_name})")

    return {
        "intercept": coefficients[0],
        "coefficients": coefficients[1:],
        "feature_columns": features.columns.tolist(),
        "r_squared": r_squared,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "actual": y,
        "predicted": predictions,
        "equation_preview": "Fare = " + " + ".join(equation_terms),
        "means": model_df[["Distance_KM", "Duration_Minutes", "Surge_Multiplier", "Trip_Hour", "Trip_DayOfWeek"]].mean(),
        "ride_types": sorted(df["Ride_Type"].dropna().unique().tolist()),
        "cities": sorted(df["City"].dropna().unique().tolist()),
    }


def make_prediction(model: dict, distance: float, duration: float, surge: float, hour: int, day_of_week: int, ride_type: str, city: str) -> float:
    row = {feature: 0.0 for feature in model["feature_columns"]}
    base_values = {
        "Distance_KM": float(distance),
        "Duration_Minutes": float(duration),
        "Surge_Multiplier": float(surge),
        "Trip_Hour": float(hour),
        "Trip_DayOfWeek": float(day_of_week),
        f"Ride_Type_{ride_type}": 1.0,
        f"City_{city}": 1.0,
    }
    for key, value in base_values.items():
        if key in row:
            row[key] = value
    x = np.array([row[column] for column in model["feature_columns"]], dtype=float)
    predicted = model["intercept"] + np.dot(x, model["coefficients"])
    return float(max(predicted, 0.0))


def plot_regression_scatter(model: dict) -> go.Figure:
    sample_df = pd.DataFrame({"Actual Fare": model["actual"], "Predicted Fare": model["predicted"]})
    fig = px.scatter(
        sample_df,
        x="Actual Fare",
        y="Predicted Fare",
        opacity=0.6,
        title="Actual vs Predicted Fare",
        color="Predicted Fare",
        color_continuous_scale=["#99F6E4", THEME["primary"], THEME["accent"]],
        trendline="ols",
    )
    min_axis = float(min(sample_df["Actual Fare"].min(), sample_df["Predicted Fare"].min()))
    max_axis = float(max(sample_df["Actual Fare"].max(), sample_df["Predicted Fare"].max()))
    fig.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            name="Perfect Fit",
            line=dict(color=THEME["danger"], dash="dash", width=2),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar_title="Pred Fare",
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def plot_actual_predicted_line(model: dict) -> go.Figure:
    compare_df = pd.DataFrame({"Actual Fare": model["actual"], "Predicted Fare": model["predicted"]}).head(120).reset_index(drop=True)
    compare_df["Observation"] = compare_df.index + 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=compare_df["Observation"],
            y=compare_df["Actual Fare"],
            mode="lines",
            name="Actual Fare",
            line=dict(color=THEME["accent"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=compare_df["Observation"],
            y=compare_df["Predicted Fare"],
            mode="lines",
            name="Predicted Fare",
            line=dict(color=THEME["primary"], width=2.5),
            fill="tonexty",
            fillcolor="rgba(20, 184, 166, 0.08)",
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Comparison",
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=1.08),
        xaxis_title="Observation",
        yaxis_title="Fare Amount (INR)",
    )
    return fig


def generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    if df.empty:
        return ["No insights can be generated because the selected filters returned zero records."]

    top_city = df.groupby("City")["Trip_ID"].nunique().sort_values(ascending=False)
    top_ride_type = df.groupby("Ride_Type")["Fare_Amount"].sum().sort_values(ascending=False)
    top_month = df.groupby("Trip_Month")["Fare_Amount"].sum().sort_values(ascending=False)
    avg_distance = df["Distance_KM"].mean()
    avg_rating = df["Driver_Rating"].mean()

    if not top_city.empty:
        insights.append(f"{top_city.index[0]} records the highest ride volume in the current filtered view.")
    if not top_ride_type.empty:
        insights.append(f"{top_ride_type.index[0]} is the top revenue-generating ride type with {inr(top_ride_type.iloc[0])} in fares.")
    if not top_month.empty:
        insights.append(f"Revenue peaks in {top_month.index[0].strftime('%b %Y')}, indicating the strongest demand period.")
    insights.append(f"Average trip distance is {avg_distance:.2f} km, which helps explain the current fare profile.")
    insights.append(f"Average driver rating is {avg_rating:.2f}/5, providing a service-quality signal alongside ride performance.")
    return insights


def render_dashboard(filtered_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    st.markdown("<h3 class='section-title'>Dashboard</h3>", unsafe_allow_html=True)
    # UX design comment:
    # The layout is intentionally top -> middle -> bottom so users see KPIs first, then comparisons,
    # then the predictive model, which creates a logical and low-clutter analytics flow.
    st.markdown(
        "<div class='sub-card'>This dashboard is arranged like a professional analytics product: top KPIs for quick monitoring, a middle grid of charts for analysis, and a bottom predictive block for advanced insight.</div>",
        unsafe_allow_html=True,
    )

    create_kpis(filtered_df, baseline_df)

    st.markdown("<h4 class='section-title'>Middle Section: Visual Analytics</h4>", unsafe_allow_html=True)
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(plot_rides_over_time(filtered_df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(plot_revenue_over_time(filtered_df), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(plot_revenue_by_ride_type(filtered_df), use_container_width=True)
    with row2_col2:
        st.plotly_chart(plot_rides_by_city(filtered_df), use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(plot_grouped_ride_type_city(filtered_df), use_container_width=True)
    with row3_col2:
        st.plotly_chart(plot_payment_pie(filtered_df), use_container_width=True)

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        st.plotly_chart(plot_fare_histogram(filtered_df), use_container_width=True)
    with row4_col2:
        st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)

    st.markdown("<h4 class='section-title'>Bottom Section: Linear Regression Model</h4>", unsafe_allow_html=True)
    model = train_model(filtered_df)
    if not model:
        st.warning("Not enough records are available to train the regression model after applying the current filters.")
    else:
        input_cols = st.columns(4)
        with input_cols[0]:
            distance_input = st.number_input("Distance (KM)", min_value=0.0, value=float(model["means"]["Distance_KM"]), step=0.5)
        with input_cols[1]:
            duration_input = st.number_input("Duration (Minutes)", min_value=0.0, value=float(model["means"]["Duration_Minutes"]), step=1.0)
        with input_cols[2]:
            surge_input = st.number_input("Surge Multiplier", min_value=1.0, value=max(1.0, float(model["means"]["Surge_Multiplier"])), step=0.05)
        with input_cols[3]:
            hour_input = st.slider("Trip Hour", min_value=0, max_value=23, value=int(round(model["means"]["Trip_Hour"])))

        selection_cols = st.columns(3)
        with selection_cols[0]:
            ride_type_input = st.selectbox("Ride Type for Prediction", options=model["ride_types"])
        with selection_cols[1]:
            city_input = st.selectbox("City for Prediction", options=model["cities"])
        with selection_cols[2]:
            day_of_week_input = st.selectbox(
                "Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                index=int(round(model["means"]["Trip_DayOfWeek"])) if 0 <= int(round(model["means"]["Trip_DayOfWeek"])) <= 6 else 0,
            )

        predicted_fare = make_prediction(
            model,
            distance=distance_input,
            duration=duration_input,
            surge=surge_input,
            hour=hour_input,
            day_of_week=day_of_week_input,
            ride_type=ride_type_input,
            city=city_input,
        )

        coeff_preview = pd.DataFrame(
            {
                "Feature": ["Intercept"] + model["feature_columns"][:10],
                "Coefficient": [model["intercept"]] + model["coefficients"][:10].tolist(),
            }
        )

        model_col1, model_col2 = st.columns([1, 1.15])
        with model_col1:
            metric_row_1 = st.columns(2)
            with metric_row_1[0]:
                st.metric("Predicted Fare", inr(predicted_fare))
            with metric_row_1[1]:
                st.metric("R-squared", f"{model['r_squared']:.3f}")

            metric_row_2 = st.columns(3)
            with metric_row_2[0]:
                st.metric("MAE", f"{model['mae']:.2f}")
            with metric_row_2[1]:
                st.metric("RMSE", f"{model['rmse']:.2f}")
            with metric_row_2[2]:
                st.metric("MAPE", f"{model['mape']:.1f}%")

            st.markdown("**Prediction Summary**")
            prediction_summary = pd.DataFrame(
                {
                    "Input Feature": ["Distance (KM)", "Duration (Minutes)", "Surge Multiplier", "Trip Hour", "Day of Week", "Ride Type", "City"],
                    "Selected Value": [distance_input, duration_input, surge_input, hour_input, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week_input], ride_type_input, city_input],
                }
            )
            st.dataframe(prediction_summary, use_container_width=True, hide_index=True)

            st.markdown("**Model Equation Preview**")
            st.code(model["equation_preview"], language="text")
            st.info(
                "The regression model uses distance, duration, surge multiplier, trip hour, day of week, ride type, and city to estimate fare amount. R-squared shows how much fare variation is explained by the model, while MAE and RMSE show average prediction error in fare units."
            )
            st.dataframe(coeff_preview.round(3), use_container_width=True, hide_index=True)
        with model_col2:
            scatter_tab, compare_tab = st.tabs(["Scatter + Regression Line", "Actual vs Predicted"])
            with scatter_tab:
                st.plotly_chart(plot_regression_scatter(model), use_container_width=True)
            with compare_tab:
                st.plotly_chart(plot_actual_predicted_line(model), use_container_width=True)

    st.markdown("<h4 class='section-title'>Insights</h4>", unsafe_allow_html=True)
    for insight in generate_insights(filtered_df):
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)


def main() -> None:
    apply_custom_theme()
    st.markdown(
        f"""
        <div class="hero">
            <h1 style="margin:0;">Cab Booking Interactive Analytics Dashboard</h1>
            <p style="margin:0.45rem 0 0 0; font-size:1rem;">
                A two-mode Streamlit application for cab booking EDA and dashboard analytics using the local dataset file <strong>{DATA_FILE}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading cab booking data and preparing analytics..."):
        time.sleep(0.5)
        raw_df = load_data(DATA_FILE)
        data = preprocess_data(raw_df)

    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Choose Mode", ["Data Overview", "Dashboard"])

    st.sidebar.header("Interactive Filters")
    city = st.sidebar.selectbox("City Filter", ["All"] + sorted(data["City"].dropna().unique().tolist()))
    ride_type_options = sorted(data["Ride_Type"].dropna().unique().tolist())
    ride_types = st.sidebar.multiselect("Ride Type Filter", options=ride_type_options, default=ride_type_options)
    gender_options = sorted(data["Gender"].dropna().unique().tolist())
    genders = st.sidebar.multiselect("Gender Filter", options=gender_options, default=gender_options)
    trip_status_options = sorted(data["Trip_Status"].dropna().unique().tolist())
    trip_statuses = st.sidebar.multiselect("Trip Status Filter", options=trip_status_options, default=trip_status_options)
    vehicle_type_options = sorted(data["Vehicle_Type"].dropna().unique().tolist())
    vehicle_types = st.sidebar.multiselect("Vehicle Type Filter", options=vehicle_type_options, default=vehicle_type_options)
    payment_method_options = sorted(data["Payment_Method"].dropna().unique().tolist())
    payment_methods = st.sidebar.multiselect("Payment Method Filter", options=payment_method_options, default=payment_method_options)

    min_date = data["Start_Time"].min().date()
    max_date = data["Start_Time"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        date_range = (min_date, max_date)

    min_fare = float(data["Fare_Amount"].min())
    max_fare = float(data["Fare_Amount"].max())
    fare_range = st.sidebar.slider("Fare Range", min_value=min_fare, max_value=max_fare, value=(min_fare, max_fare))

    filtered_raw = filter_data(
        data,
        city=city,
        ride_types=ride_types,
        genders=genders,
        trip_statuses=trip_statuses,
        vehicle_types=vehicle_types,
        payment_methods=payment_methods,
        date_range=date_range,
        fare_range=fare_range,
    )
    filtered = get_trip_view(filtered_raw)

    baseline = previous_period(
        get_trip_view(data),
        pd.to_datetime(date_range[0]),
        pd.to_datetime(date_range[1]),
    )

    if filtered.empty:
        st.warning("No records match the selected filters. Please broaden the filter values.")
        return

    if mode == "Data Overview":
        render_data_overview(filtered_raw, filtered)
    else:
        render_dashboard(filtered, baseline)

    st.caption(
        f"Current filtered view: {len(filtered_raw):,} raw rows | {filtered['Trip_ID'].nunique():,} unique rides | Date coverage: {filtered['Start_Time'].min():%d %b %Y} to {filtered['Start_Time'].max():%d %b %Y}"
    )


if __name__ == "__main__":
    main()
