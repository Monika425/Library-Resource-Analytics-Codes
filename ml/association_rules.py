import pandas as pd
import os

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

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

df["location_acronym"] = (
    df["location_acronym"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["resource_type"] = (
    df["resource_type"]
    .astype(str)
    .str.strip()
)

df = df.dropna(
    subset=[
        "month_date",
        "location_acronym",
        "resource_type",
        "circulation_count"
    ]
)

df = df[
    (df["circulation_count"] >= 0) &
    (df["resource_type"] != "") &
    (df["location_acronym"] != "")
]

df = df[
    df["location_acronym"] != "TOTAL"
]

data = df.groupby(
    [
        "month_date",
        "location_acronym",
        "resource_type"
    ],
    as_index=False
)["circulation_count"].sum()

data["present"] = (
    data["circulation_count"] > 0
)

transactions = data.pivot_table(
    index=[
        "month_date",
        "location_acronym"
    ],
    columns="resource_type",
    values="present",
    aggfunc="max",
    fill_value=False
)

transactions = transactions.astype(bool)

print("Transaction Matrix:")
print(transactions)

print("\nNumber of transactions:", len(transactions))
print("Number of resource types:", len(transactions.columns))

frequent = fpgrowth(
    transactions,
    min_support=0.10,
    use_colnames=True,
    max_len=2
)

if frequent.empty:
    frequent = fpgrowth(
        transactions,
        min_support=0.05,
        use_colnames=True,
        max_len=2
    )

if frequent.empty:
    print("\nNo frequent itemsets found. Try lowering min_support.")
else:
    print("\nFrequent Itemsets:")
    print(frequent.to_string(index=False))

    if frequent["itemsets"].apply(len).max() < 2:
        print("\nNot enough multi-resource itemsets to generate rules.")
    else:
        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=0.70
        )

        if rules.empty:
            print("\nNo association rules meet the confidence threshold.")
        else:
            rules = rules[
                (rules["lift"] > 1.10) &
                (rules["antecedents"].apply(len) == 1) &
                (rules["consequents"].apply(len) == 1)
            ].copy()

            rules = rules.sort_values(
                ["lift", "confidence", "support"],
                ascending=False
            )

            rules = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(100)

            rules["antecedents"] = rules["antecedents"].apply(
                lambda x: ", ".join(sorted(x))
            )

            rules["consequents"] = rules["consequents"].apply(
                lambda x: ", ".join(sorted(x))
            )

            if rules.empty:
                print("\nNo rules meet the selected support, confidence, and lift criteria.")
            else:
                print("\nStrong Association Rules:")
                print(rules.to_string(index=False))

                output_file = os.path.join(
                    output_dir,
                    "association_rules.csv"
                )

                rules.to_csv(
                    output_file,
                    index=False
                )

                print("\nSaved successfully:")
                print(output_file)
                print("Number of rules saved:", len(rules))
