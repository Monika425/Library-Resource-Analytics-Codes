import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "ml" else BASE_DIR
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
INPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_library_circulation.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "Cleaned dataset not found. Run preprocess.py first so this file exists: "
            + INPUT_FILE
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "month_date",
        "location_acronym",
        "resource_type",
        "circulation_format_total",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df["location_acronym"] = (
        df["location_acronym"].astype(str).str.strip().str.upper()
    )
    df["resource_type"] = df["resource_type"].astype(str).str.strip()
    df["circulation_format_total"] = pd.to_numeric(
        df["circulation_format_total"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "month_date",
            "location_acronym",
            "resource_type",
            "circulation_format_total",
        ]
    )
    df = df[
        (df["circulation_format_total"] >= 0)
        & (df["resource_type"] != "")
        & (df["location_acronym"] != "")
        & (df["location_acronym"] != "TOTAL")
    ]

    if df.empty:
        raise ValueError("No valid records remain after cleaning.")

    # A transaction represents the resource types used at one location in one month.
    grouped = df.groupby(
        ["month_date", "location_acronym", "resource_type"],
        as_index=False,
    )["circulation_format_total"].sum()

    grouped["present"] = grouped["circulation_format_total"] > 0

    transactions = grouped.pivot_table(
        index=["month_date", "location_acronym"],
        columns="resource_type",
        values="present",
        aggfunc="max",
        fill_value=False,
    ).astype(bool)

    transactions = transactions.loc[:, transactions.any(axis=0)]

    print("APRIORI ASSOCIATION ANALYSIS")
    print("----------------------------")
    print("Transactions:", len(transactions))
    print("Resource types:", len(transactions.columns))

    if len(transactions) < 2 or len(transactions.columns) < 2:
        raise ValueError(
            "Apriori needs at least two transactions and two resource types."
        )

    # Lower support thresholds are tried if the dataset is sparse.
    frequent = pd.DataFrame()
    used_support = None

    for support in (0.10, 0.05, 0.02, 0.01):
        frequent = apriori(
            transactions,
            min_support=support,
            use_colnames=True,
            max_len=3,
        )
        if not frequent.empty:
            used_support = support
            break

    if frequent.empty:
        print("No frequent itemsets were found, even at 1% support.")
        return

    frequent["itemsets_text"] = frequent["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )
    frequent["itemset_size"] = frequent["itemsets"].apply(len)

    itemsets_output = frequent[
        ["itemsets_text", "itemset_size", "support"]
    ].sort_values(
        ["itemset_size", "support"],
        ascending=[True, False],
    )

    itemsets_file = os.path.join(OUTPUT_DIR, "apriori_frequent_itemsets.csv")
    itemsets_output.to_csv(itemsets_file, index=False)

    print(f"Minimum support used: {used_support:.0%}")
    print("\nFrequent itemsets:")
    print(itemsets_output.head(20).to_string(index=False))

    if frequent["itemsets"].apply(len).max() < 2:
        print("\nOnly single-item sets were found; no association rules can be generated.")
        print("Frequent itemsets saved to:", itemsets_file)
        return

    rules = association_rules(
        frequent[["support", "itemsets"]],
        metric="confidence",
        min_threshold=0.50,
    )

    if rules.empty:
        print("\nNo rules met the 50% confidence threshold.")
        print("Frequent itemsets saved to:", itemsets_file)
        return

    rules = rules[
        (rules["antecedents"].apply(len) >= 1)
        & (rules["consequents"].apply(len) >= 1)
        & (rules["lift"] > 1.0)
    ].copy()

    if rules.empty:
        print("\nNo rules with lift greater than 1 were found.")
        print("Frequent itemsets saved to:", itemsets_file)
        return

    rules["antecedents"] = rules["antecedents"].apply(
        lambda items: ", ".join(sorted(items))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda items: ", ".join(sorted(items))
    )

    rules = rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].sort_values(
        ["lift", "confidence", "support"],
        ascending=False,
    )

    rules_file = os.path.join(OUTPUT_DIR, "apriori_association_rules.csv")
    rules.to_csv(rules_file, index=False)

    print("\nTop association rules:")
    print(rules.head(20).to_string(index=False))
    print("\nSaved frequent itemsets to:", itemsets_file)
    print("Saved association rules to:", rules_file)
    print("\nInterpretation:")
    print("- Support: proportion of monthly-location transactions containing the itemset.")
    print("- Confidence: how often the consequent appears when the antecedent appears.")
    print("- Lift: how much more often the items occur together than expected by chance.")
    print("These are co-occurrence patterns, not proof that one resource causes another.")


if __name__ == "__main__":
    main()
