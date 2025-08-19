-- 1. Show all tables in the current schema
SHOW TABLES IN SCHEMA COVID19_DB_COPY.PUBLIC;

-- 2. Describe the structure of the COVID-19 dataset table
DESCRIBE TABLE COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL;

-- 3. Count total records grouped by COUNTRY_REGION
SELECT 
    COUNTRY_REGION, 
    COUNT(*) AS total_records
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL
GROUP BY COUNTRY_REGION
ORDER BY total_records DESC;

-- 4. Find the date range of the dataset
SELECT 
    MIN(DATE) AS start_date,
    MAX(DATE) AS end_date
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL;

-- 5. Calculate average, max and min new COVID-19 cases per COUNTRY_REGION
SELECT
    COUNTRY_REGION,
    AVG(CASES_SINCE_PREV_DAY) AS avg_new_cases,
    MAX(CASES_SINCE_PREV_DAY) AS max_new_cases,
    MIN(CASES_SINCE_PREV_DAY) AS min_new_cases
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL
GROUP BY COUNTRY_REGION
ORDER BY avg_new_cases DESC;

-- 6. Check for missing values (nulls) in key columns
SELECT
    COUNT(*) AS total_rows,
    COUNT(CASES) AS cases_not_null,
    COUNT(DEATHS) AS deaths_not_null,
    COUNT(POPULATION) AS population_not_null
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL;

-- 7. Aggregate total new COVID-19 cases by DATE (global trend)
SELECT
    DATE,
    SUM(CASES_SINCE_PREV_DAY) AS total_new_cases
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL
GROUP BY DATE
ORDER BY DATE;

-- 8. Top 10 COUNTRY_REGION with highest total DEATHS
SELECT
    COUNTRY_REGION,
    SUM(DEATHS) AS total_deaths
FROM COVID19_DB_COPY.PUBLIC.ECDC_GLOBAL
GROUP BY COUNTRY_REGION
ORDER BY total_deaths DESC
LIMIT 10;
