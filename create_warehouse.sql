DROP DATABASE IF EXISTS library_dw;

CREATE DATABASE library_dw;

USE library_dw;


CREATE TABLE dim_date (
    date_id INT AUTO_INCREMENT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INT NOT NULL,
    quarter VARCHAR(5) NOT NULL,
    month INT NOT NULL,
    month_name VARCHAR(20) NOT NULL
);


CREATE TABLE dim_location (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    location_acronym VARCHAR(50) NOT NULL UNIQUE
);


CREATE TABLE dim_resource (
    resource_id INT AUTO_INCREMENT PRIMARY KEY,
    circulation_format VARCHAR(100) NOT NULL UNIQUE,
    audience VARCHAR(50),
    resource_type VARCHAR(100)
);


CREATE TABLE fact_circulation (
    circulation_id INT AUTO_INCREMENT PRIMARY KEY,

    source_circ_id INT,

    date_id INT NOT NULL,

    location_id INT NOT NULL,

    resource_id INT NOT NULL,

    circulation_count INT NOT NULL,

    FOREIGN KEY (date_id)
        REFERENCES dim_date(date_id),

    FOREIGN KEY (location_id)
        REFERENCES dim_location(location_id),

    FOREIGN KEY (resource_id)
        REFERENCES dim_resource(resource_id)
);