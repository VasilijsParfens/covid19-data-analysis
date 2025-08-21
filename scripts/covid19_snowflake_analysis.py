import snowflake.connector
import pandas as pd
from ydata_profiling import ProfileReport
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def fetch_covid_data():
    """Establish connection to Snowflake, execute query, and return covid data as DataFrame."""
    query = """
    SELECT COUNTRY_REGION, SUM(CASES) AS TOTAL_CASES, SUM(DEATHS) AS TOTAL_DEATHS
    FROM ECDC_GLOBAL
    GROUP BY COUNTRY_REGION
    """

    conn_params = {
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'role': os.getenv('SNOWFLAKE_ROLE')
    }

    with snowflake.connector.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    # Rename columns to consistent format
    df.rename(columns={'TOTAL_CASES': 'Total_Cases', 'TOTAL_DEATHS': 'Total_Deaths'}, inplace=True)

    # Normalize key for merging
    df['COUNTRY_KEY'] = df['COUNTRY_REGION'].str.lower().str.strip()

    # Preview dataset
    print("\n--- Preview: COVID Data from Snowflake ---")
    print(df.head())
    print(df.info())

    return df


def load_population_data(filepath='data/world_population.csv'):
    """Load population data from CSV, clean and normalize keys, convert data types."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to read population data from {filepath}: {e}")

    # Normalize country key
    df['COUNTRY_KEY'] = df['Country/Territory'].str.lower().str.strip()

    # Convert necessary columns to numeric types
    df['Population'] = pd.to_numeric(df['2022 Population'], errors='coerce')
    df['Population_2020'] = pd.to_numeric(df['2020 Population'], errors='coerce')
    df['Area_km2'] = pd.to_numeric(df['Area (km²)'], errors='coerce')

    # Preview dataset
    print("\n--- Preview: Population Data ---")
    print(df.head())
    print(df.info())

    return df


def load_country_mapping(filepath='data/country_mapping.csv'):
    """Load country mapping file and normalize keys."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to read mapping data from {filepath}: {e}")

    df['SNOWFLAKE_KEY'] = df['snowflake_name'].str.lower().str.strip()
    df['CSV_KEY'] = df['csv_name'].str.lower().str.strip()

    # Preview mapping
    print("\n--- Preview: Country Mapping Data ---")
    print(df.head())
    print(df.info())

    return df[['SNOWFLAKE_KEY', 'CSV_KEY']]


def apply_country_mapping(covid_df, mapping_df):
    """Map Snowflake country keys to CSV country keys for consistent merging."""
    merged = covid_df.merge(mapping_df,
                           left_on='COUNTRY_KEY',
                           right_on='SNOWFLAKE_KEY',
                           how='left')

    # Replace key with CSV key if mapping exists
    merged['COUNTRY_KEY'] = merged['CSV_KEY'].combine_first(merged['COUNTRY_KEY'])

    # Drop helper columns
    merged.drop(['SNOWFLAKE_KEY', 'CSV_KEY'], axis=1, inplace=True)

    # Preview mapped dataset
    print("\n--- Preview: COVID Data after Country Mapping ---")
    print(merged.head())
    print(merged.info())

    return merged


def merge_datasets(covid_df, population_df):
    """Merge COVID data with population data on normalized country key."""
    merged = covid_df.merge(population_df, on='COUNTRY_KEY', how='inner')

    # Preview merged dataset
    print("\n--- Preview: Merged COVID + Population Data ---")
    print(merged.head())
    print(merged.info())

    return merged


def calculate_metrics(df):
    """Calculate per million, per area, CFR and population density metrics."""
    df['Cases_per_Million'] = df['Total_Cases'] / df['Population_2020'] * 1_000_000
    df['Deaths_per_Million'] = df['Total_Deaths'] / df['Population_2020'] * 1_000_000
    df['CFR_%'] = df['Total_Deaths'] / df['Total_Cases'] * 100
    df['Cases_per_km2'] = df['Total_Cases'] / df['Area_km2']
    df['Deaths_per_km2'] = df['Total_Deaths'] / df['Area_km2']
    df['Population_Density_Calc'] = df['Population_2020'] / df['Area_km2']

    # Preview metrics
    print("\n--- Preview: Dataset with Calculated Metrics ---")
    print(df.head())
    print(df.info())

    return df


def filter_columns(df):
    """Select relevant columns for the final dataset."""
    cols = ['COUNTRY_REGION', 'Total_Cases', 'Total_Deaths', 'Population_2020', 'Area_km2',
            'Cases_per_Million', 'Deaths_per_Million', 'CFR_%', 'Cases_per_km2',
            'Deaths_per_km2', 'Population_Density_Calc']
    final_df = df[cols]

    # Preview final dataset
    print("\n--- Preview: Final Dataset for Analysis ---")
    print(final_df.head())
    print(final_df.info())

    return final_df


def main():
    print("\n=== Starting COVID Data Analysis ===\n")

    covid_df = fetch_covid_data()
    population_df = load_population_data()
    mapping_df = load_country_mapping()
    covid_df = apply_country_mapping(covid_df, mapping_df)
    merged_df = merge_datasets(covid_df, population_df)
    enriched_df = calculate_metrics(merged_df)
    final_df = filter_columns(enriched_df)

    # Save to data/ folder
    output_csv = 'data/enriched_covid_data_filtered.csv'
    final_df.to_csv(output_csv, index=False)
    print(f"\n-> Dataset saved successfully to '{output_csv}'.")

    profile = ProfileReport(final_df, title="COVID-19 Enriched Data EDA Report", explorative=True)
    profile.to_file("reports/covid19_eda_report.html")
    print("-> EDA report saved as 'data/covid19_eda_report.html'.\n")

    print("=== COVID Data Analysis Completed Successfully ===\n")



if __name__ == "__main__":
    main()
