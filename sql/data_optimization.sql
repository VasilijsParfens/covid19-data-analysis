-- ==============================
-- 1. Create database if it doesn't exist
-- ==============================
CREATE DATABASE IF NOT EXISTS COVID19_DB_COPY;

-- ==============================
-- 2. Use the database
-- ==============================
USE DATABASE COVID19_DB_COPY;

-- ==============================
-- 3. Create schema if it doesn't exist
-- ==============================
CREATE SCHEMA IF NOT EXISTS PUBLIC;

-- ==============================
-- 4. Use the schema
-- ==============================
USE SCHEMA PUBLIC;

-- ==============================
-- 5. Create a copy of the Marketplace COVID-19 dataset with only needed columns
-- ==============================
CREATE OR REPLACE TABLE ECDC_GLOBAL_COPY AS
SELECT 
    DATE,
    COUNTRY_REGION,
    CASES_SINCE_PREV_DAY,
    DEATHS,
    POPULATION
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL;

-- ==============================
-- 6. Apply clustering on DATE and COUNTRY_REGION for query optimization
-- ==============================
ALTER TABLE ECDC_GLOBAL_COPY
CLUSTER BY (DATE, COUNTRY_REGION);

-- ==============================
-- 7. Create filtered table for active cases to speed up trend queries
-- ==============================
CREATE OR REPLACE TABLE ECDC_GLOBAL_ACTIVE AS
SELECT *
FROM ECDC_GLOBAL_COPY
WHERE CASES_SINCE_PREV_DAY IS NOT NULL AND CASES_SINCE_PREV_DAY > 0;

-- ==============================
-- 8. Create materialized views for frequent aggregations
-- ==============================

-- Total new cases by DATE (global trend)
CREATE OR REPLACE MATERIALIZED VIEW MV_TOTAL_NEW_CASES AS
SELECT 
    DATE, 
    SUM(CASES_SINCE_PREV_DAY) AS total_new_cases
FROM ECDC_GLOBAL_ACTIVE
GROUP BY DATE;

-- Total deaths by COUNTRY_REGION
CREATE OR REPLACE MATERIALIZED VIEW MV_TOTAL_DEATHS AS
SELECT 
    COUNTRY_REGION, 
    SUM(DEATHS) AS total_deaths
FROM ECDC_GLOBAL_COPY
GROUP BY COUNTRY_REGION;

-- Average, max, min new cases per COUNTRY_REGION
CREATE OR REPLACE MATERIALIZED VIEW MV_NEW_CASES_STATS AS
SELECT
    COUNTRY_REGION,
    AVG(CASES_SINCE_PREV_DAY) AS avg_new_cases,
    MAX(CASES_SINCE_PREV_DAY) AS max_new_cases,
    MIN(CASES_SINCE_PREV_DAY) AS min_new_cases
FROM ECDC_GLOBAL_ACTIVE
GROUP BY COUNTRY_REGION;

-- ==============================
-- 9. Run queries on optimized tables and materialized views
-- ==============================

-- 9.1 Show all tables
SHOW TABLES;

-- 9.2 Describe table structure
DESCRIBE TABLE ECDC_GLOBAL_COPY;

-- 9.3 Count total records by COUNTRY_REGION
SELECT 
    COUNTRY_REGION, 
    COUNT(1) AS total_records
FROM ECDC_GLOBAL_COPY
GROUP BY COUNTRY_REGION
ORDER BY total_records DESC;

-- 9.4 Find the date range of the dataset
SELECT 
    MIN(DATE) AS start_date,
    MAX(DATE) AS end_date
FROM ECDC_GLOBAL_COPY;

-- 9.5 Average, max, min new cases per COUNTRY_REGION (materialized view)
SELECT * FROM MV_NEW_CASES_STATS
ORDER BY avg_new_cases DESC;

-- 9.6 Check for missing values (nulls) in key columns
SELECT
    COUNT(1) AS total_rows,
    COUNT(CASES_SINCE_PREV_DAY) AS cases_not_null,
    COUNT(DEATHS) AS deaths_not_null,
    COUNT(POPULATION) AS population_not_null
FROM ECDC_GLOBAL_COPY;

-- 9.7 Total new cases by DATE (global trend, materialized view)
SELECT * FROM MV_TOTAL_NEW_CASES
ORDER BY DATE;

-- 9.8 Top 10 COUNTRY_REGION with highest total deaths (materialized view)
SELECT * 
FROM MV_TOTAL_DEATHS
ORDER BY total_deaths DESC
LIMIT 10;

-- 9.9 Example of optimized date-range query using clustering
SELECT DATE, SUM(CASES_SINCE_PREV_DAY) AS total_new_cases
FROM ECDC_GLOBAL_ACTIVE
WHERE DATE BETWEEN '2019-12-01' AND '2020-06-30'
GROUP BY DATE
ORDER BY DATE;
