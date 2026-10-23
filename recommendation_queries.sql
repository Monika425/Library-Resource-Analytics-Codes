USE library_dw;

SELECT
    r.resource_type,
    SUM(f.circulation_count) AS total_demand
FROM fact_circulation f
JOIN dim_resource r
    ON f.resource_id = r.resource_id
GROUP BY r.resource_type
ORDER BY total_demand DESC;




USE library_dw;

SELECT
    l.location_acronym,
    r.resource_type,
    SUM(f.circulation_count) AS total_demand
FROM fact_circulation f
JOIN dim_location l
    ON f.location_id = l.location_id
JOIN dim_resource r
    ON f.resource_id = r.resource_id
GROUP BY
    l.location_acronym,
    r.resource_type
ORDER BY
    l.location_acronym,
    total_demand DESC;




USE library_dw;

SELECT
    r.resource_type,
    SUM(f.circulation_count) AS total_demand,
    COUNT(*) AS usage_records,
    ROUND(AVG(f.circulation_count), 2) AS average_demand
FROM fact_circulation f
JOIN dim_resource r
    ON f.resource_id = r.resource_id
GROUP BY r.resource_type
ORDER BY average_demand DESC;




USE library_dw;

SELECT
    d.year,
    r.resource_type,
    SUM(f.circulation_count) AS total_demand
FROM fact_circulation f
JOIN dim_date d
    ON f.date_id = d.date_id
JOIN dim_resource r
    ON f.resource_id = r.resource_id
GROUP BY
    d.year,
    r.resource_type
ORDER BY
    d.year,
    total_demand DESC;
