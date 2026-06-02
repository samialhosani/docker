import os
import sqlite3
import pandas as pd
from config import load_config

config = load_config()
DATA_DIR = config.data_dir
DB_NAME = config.db_path


csv_files = [
    "Dim_Courses.csv",
    "Dim_Instructors.csv",
    "Dim_Lectures.csv",
    "Dim_Students.csv",
    "Fact_Assignment_Submissions.csv",
    "Fact_Enrollment.csv",
    "Fact_Exams.csv",
    "Fact_Final_Grades_Credit.csv",
    "Fact_Final_Grades_General2.csv",
    "Fact_Lecture_Progress.csv",
    "Report_GPA.csv",
    "Bridge_Course_Assessments.csv",
]


def initialize_database():
    """Reads CSVs from the data folder and loads them into a local SQLite DB."""

    conn = sqlite3.connect(DB_NAME)
    print(f"Connected to database: {DB_NAME}\n")

    for file_name in csv_files:
        file_path = os.path.join(DATA_DIR, file_name)

        table_name = file_name.replace(".csv", "")

        if os.path.exists(file_path):
            try:

                df = pd.read_csv(file_path)

                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Loaded '{file_name}' -> Table '{table_name}' ({len(df)} rows)")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        else:
            print(f"Warning: File not found at {file_path}")

    print("\n--- Database Verification ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables available in the database:")
    for table in tables:
        print(f" - {table[0]}")

    conn.close()
    print("\nDatabase initialization complete!")


if __name__ == "__main__":

    if not os.path.exists(DATA_DIR):
        print(
            f"Error: Directory '{DATA_DIR}' does not exist. Please create it and add your CSVs."
        )
    else:
        initialize_database()
