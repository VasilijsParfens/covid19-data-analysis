from pymongo import MongoClient
import snowflake.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_conn_str = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_conn_str)
mongo_db = mongo_client.covid_analysis
comments_collection = mongo_db.user_comments

# Snowflake connection parameters
sf_conn_params = {
    'user': os.getenv("SNOWFLAKE_USER"),
    'password': os.getenv("SNOWFLAKE_PASSWORD"),
    'account': os.getenv("SNOWFLAKE_ACCOUNT"),
    'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
    'database': os.getenv("SNOWFLAKE_DATABASE"),
    'schema': os.getenv("SNOWFLAKE_SCHEMA"),
    'role': os.getenv("SNOWFLAKE_ROLE")
}

def get_snowflake_connection():
    return snowflake.connector.connect(**sf_conn_params)
