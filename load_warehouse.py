import pandas as pd
import mysql.connector

df = pd.read_csv(
    "output/cleaned_library_circulation.csv"
)

df["month_date"] = pd.to_datetime(
    df["month_date"]
)

con = mysql.connector.connect(
    host="localhost",
    user="root",
    password="sql@pgp007",
    database="library_dw"
)

cur = con.cursor()

print("Connected to MySQL.")


# DATE DIMENSION

dates = (
    df[
        [
            "month_date",
            "year",
            "quarter",
            "month",
            "month_name"
        ]
    ]
    .drop_duplicates()
)

for _, r in dates.iterrows():

    cur.execute(
        """
        INSERT IGNORE INTO dim_date
        (
            full_date,
            year,
            quarter,
            month,
            month_name
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            r["month_date"].date(),
            int(r["year"]),
            r["quarter"],
            int(r["month"]),
            r["month_name"]
        )
    )

con.commit()

print("Date dimension loaded.")


# LOCATION DIMENSION

locations = (
    df["location_acronym"]
    .drop_duplicates()
)

for location in locations:

    cur.execute(
        """
        INSERT IGNORE INTO dim_location
        (
            location_acronym
        )
        VALUES (%s)
        """,
        (location,)
    )

con.commit()

print("Location dimension loaded.")


# RESOURCE DIMENSION

resources = (
    df[
        [
            "circulation_format",
            "audience",
            "resource_type"
        ]
    ]
    .drop_duplicates()
)

for _, r in resources.iterrows():

    cur.execute(
        """
        INSERT IGNORE INTO dim_resource
        (
            circulation_format,
            audience,
            resource_type
        )
        VALUES (%s, %s, %s)
        """,
        (
            r["circulation_format"],
            r["audience"],
            r["resource_type"]
        )
    )

con.commit()

print("Resource dimension loaded.")


# CREATE LOOKUP DICTIONARIES

cur.execute(
    """
    SELECT date_id, full_date
    FROM dim_date
    """
)

date_map = {}

for date_id, date in cur.fetchall():

    date_map[str(date)] = date_id


cur.execute(
    """
    SELECT location_id, location_acronym
    FROM dim_location
    """
)

location_map = {}

for location_id, location in cur.fetchall():

    location_map[location] = location_id


cur.execute(
    """
    SELECT resource_id, circulation_format
    FROM dim_resource
    """
)

resource_map = {}

for resource_id, resource in cur.fetchall():

    resource_map[resource] = resource_id


# FACT TABLE

rows = []

for _, r in df.iterrows():

    date_key = date_map[
        str(r["month_date"].date())
    ]

    location_key = location_map[
        r["location_acronym"]
    ]

    resource_key = resource_map[
        r["circulation_format"]
    ]

    if pd.isna(r["circ_id"]):
        source_id = None
    else:
        source_id = int(r["circ_id"])

    count = int(
        r["circulation_format_total"]
    )

    rows.append(
        (
            source_id,
            date_key,
            location_key,
            resource_key,
            count
        )
    )


cur.executemany(
    """
    INSERT INTO fact_circulation
    (
        source_circ_id,
        date_id,
        location_id,
        resource_id,
        circulation_count
    )
    VALUES (%s, %s, %s, %s, %s)
    """,
    rows
)

con.commit()

print(
    "Fact table loaded:",
    len(rows),
    "records"
)

cur.close()

con.close()

print("Data warehouse completed.")