import pandas as pd

df = pd.read_csv("dataset/library_circulation.csv")

print("Original dataset:")
print(df.head())

print("\nOriginal shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

df["circ_id"] = (
    df["circ_id"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["circ_id"] = pd.to_numeric(
    df["circ_id"],
    errors="coerce"
)

df["month_date"] = pd.to_datetime(
    df["month_date"],
    errors="coerce"
)

df["location_acronym"] = (
    df["location_acronym"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["circulation_format"] = (
    df["circulation_format"]
    .astype(str)
    .str.strip()
)

df["circulation_format_total"] = (
    df["circulation_format_total"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["circulation_format_total"] = pd.to_numeric(
    df["circulation_format_total"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "month_date",
        "location_acronym",
        "circulation_format",
        "circulation_format_total"
    ]
)

df = df.drop_duplicates()

df = df[
    df["circulation_format_total"] >= 0
]

df = df[
    ~df["circulation_format"]
    .str.contains(
        "Total",
        case=False,
        na=False
    )
]

df["year"] = df["month_date"].dt.year

df["month"] = df["month_date"].dt.month

df["month_name"] = (
    df["month_date"].dt.month_name()
)

df["quarter"] = (
    "Q" +
    df["month_date"]
    .dt.quarter
    .astype(str)
)

def get_audience(x):

    x = x.strip().lower()

    if x.startswith("adult"):
        return "Adult"

    if x.startswith("children"):
        return "Children"

    if x.startswith("child"):
        return "Children"

    if x.startswith("teen"):
        return "Teen"

    if x.startswith("youth"):
        return "Youth"

    return "General"


def get_resource_type(x):

    prefixes = [
        "Adult ",
        "Children ",
        "Child ",
        "Teen ",
        "Youth "
    ]

    for prefix in prefixes:

        if x.lower().startswith(
            prefix.lower()
        ):
            return x[len(prefix):].strip()

    return x


df["audience"] = (
    df["circulation_format"]
    .apply(get_audience)
)

df["resource_type"] = (
    df["circulation_format"]
    .apply(get_resource_type)
)

df = df[
    [
        "circ_id",
        "month_date",
        "year",
        "quarter",
        "month",
        "month_name",
        "location_acronym",
        "circulation_format",
        "audience",
        "resource_type",
        "circulation_format_total"
    ]
]

df = df.sort_values(
    [
        "month_date",
        "location_acronym",
        "circulation_format"
    ]
)

df.to_csv(
    "output/cleaned_library_circulation.csv",
    index=False
)

print("\nFinal dataset:")
print(df.head())

print("\nFinal shape:")
print(df.shape)

print("\nSaved as:")
print("output/cleaned_library_circulation.csv")