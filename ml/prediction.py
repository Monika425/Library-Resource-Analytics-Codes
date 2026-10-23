import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

base_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

output_dir = os.path.join(base_dir, "output")

os.makedirs(output_dir, exist_ok=True)

input_file = os.path.join(
    output_dir,
    "cleaned_library_circulation.csv"
)

df = pd.read_csv(input_file)

df["month_date"] = pd.to_datetime(
    df["month_date"],
    errors="coerce"
)

df["circulation_count"] = pd.to_numeric(
    df["circulation_format_total"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "month_date",
        "resource_type",
        "circulation_count"
    ]
)

df = df[
    (df["circulation_count"] >= 0) &
    (df["resource_type"].astype(str).str.strip() != "")
].copy()

df["month_date"] = (
    df["month_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly = df.groupby(
    ["month_date", "resource_type"],
    as_index=False
)["circulation_count"].sum()

months = pd.date_range(
    monthly["month_date"].min(),
    monthly["month_date"].max(),
    freq="MS"
)

resources = sorted(
    monthly["resource_type"].unique()
)

grid = pd.MultiIndex.from_product(
    [months, resources],
    names=["month_date", "resource_type"]
).to_frame(index=False)

monthly = grid.merge(
    monthly,
    on=["month_date", "resource_type"],
    how="left"
)

monthly["circulation_count"] = (
    monthly["circulation_count"].fillna(0)
)

monthly = monthly.sort_values(
    ["resource_type", "month_date"]
).reset_index(drop=True)

monthly["lag_1"] = (
    monthly.groupby("resource_type")["circulation_count"].shift(1)
)

monthly["lag_2"] = (
    monthly.groupby("resource_type")["circulation_count"].shift(2)
)

monthly["lag_3"] = (
    monthly.groupby("resource_type")["circulation_count"].shift(3)
)

monthly["rolling_3"] = (
    monthly.groupby("resource_type")["circulation_count"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

monthly["year"] = monthly["month_date"].dt.year
monthly["month"] = monthly["month_date"].dt.month

model_data = monthly.dropna(
    subset=["lag_1", "lag_2", "lag_3", "rolling_3"]
).copy()

dates = sorted(model_data["month_date"].unique())

if len(dates) < 2:
    raise ValueError(
        "Not enough monthly data to train and test the prediction model."
    )

split_position = int(len(dates) * 0.8)
split_position = max(1, min(split_position, len(dates) - 1))
split_date = dates[split_position]

train = model_data[
    model_data["month_date"] < split_date
].copy()

test = model_data[
    model_data["month_date"] >= split_date
].copy()

features = [
    "resource_type",
    "year",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_3"
]

x_train = pd.get_dummies(
    train[features],
    columns=["resource_type"]
)

x_test = pd.get_dummies(
    test[features],
    columns=["resource_type"]
)

x_test = x_test.reindex(
    columns=x_train.columns,
    fill_value=0
)

y_train = train["circulation_count"]
y_test = test["circulation_count"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

prediction = model.predict(x_test)

mae = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))

print("Prediction Results")
print("==================")
print("Training records:", len(train))
print("Testing records:", len(test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

test_result = test[
    ["month_date", "resource_type", "circulation_count"]
].copy()

test_result["predicted_demand"] = prediction

test_file = os.path.join(
    output_dir,
    "prediction_test_results.csv"
)

test_result.to_csv(
    test_file,
    index=False
)

print("\nTest predictions saved:")
print(test_file)

full_x = pd.get_dummies(
    model_data[features],
    columns=["resource_type"]
)

full_x = full_x.reindex(
    columns=x_train.columns,
    fill_value=0
)

model.fit(
    full_x,
    model_data["circulation_count"]
)

history = {}

for resource in resources:
    temp = monthly[
        monthly["resource_type"] == resource
    ].sort_values("month_date")

    history[resource] = list(
        temp["circulation_count"].astype(float)
    )

last_date = monthly["month_date"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

future_rows = []

for date in future_dates:
    for resource in resources:
        values = history[resource]

        row_data = {
            "resource_type": resource,
            "year": date.year,
            "month": date.month,
            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_3": values[-3],
            "rolling_3": np.mean(values[-3:])
        }

        row = pd.DataFrame([row_data])

        row = pd.get_dummies(
            row,
            columns=["resource_type"]
        )

        row = row.reindex(
            columns=x_train.columns,
            fill_value=0
        )

        value = max(
            0,
            float(model.predict(row)[0])
        )

        value = round(value, 2)

        history[resource].append(value)

        future_rows.append([
            date,
            resource,
            value
        ])

forecast = pd.DataFrame(
    future_rows,
    columns=[
        "month_date",
        "resource_type",
        "predicted_demand"
    ]
)

forecast_file = os.path.join(
    output_dir,
    "future_demand_forecast.csv"
)

forecast.to_csv(
    forecast_file,
    index=False
)

print("\nFuture Demand Forecast")
print("======================")
print(forecast.to_string(index=False))

print("\nForecast saved:")
print(forecast_file)

