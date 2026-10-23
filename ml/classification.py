import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os

base_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

output_dir = os.path.join(base_dir, "output")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(
    os.path.join(
        base_dir,
        "output",
        "cleaned_library_circulation.csv"
    )
)

df["month_date"] = pd.to_datetime(
    df["month_date"]
)

df["circulation_count"] = pd.to_numeric(
    df["circulation_format_total"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "circulation_count",
        "year",
        "month",
        "location_acronym",
        "resource_type"
    ]
)

x = df[
    [
        "year",
        "month",
        "location_acronym",
        "resource_type",
        "audience"
    ]
].copy()

y = df["circulation_count"].copy()

x_train, x_test, y_train_count, y_test_count = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

q1 = np.quantile(
    y_train_count,
    1 / 3
)

q2 = np.quantile(
    y_train_count,
    2 / 3
)

def make_class(x):
    if x <= q1:
        return "Low"
    elif x <= q2:
        return "Medium"
    else:
        return "High"

y_train = y_train_count.apply(make_class)
y_test = y_test_count.apply(make_class)

x_train = pd.get_dummies(
    x_train,
    columns=[
        "location_acronym",
        "resource_type",
        "audience"
    ]
)

x_test = pd.get_dummies(
    x_test,
    columns=[
        "location_acronym",
        "resource_type",
        "audience"
    ]
)

x_test = x_test.reindex(
    columns=x_train.columns,
    fill_value=0
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    x_train,
    y_train
)

prediction = model.predict(
    x_test
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Classification Results")
print("======================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        prediction
    )
)

result = x_test.copy()

result["actual"] = y_test.values
result["predicted"] = prediction

result.to_csv(
    os.path.join(output_dir, "classification_results.csv"),
    index=False
)

print(
    "\nSaved:"
)

print(
    os.path.join(output_dir, "classification_results.csv")
)