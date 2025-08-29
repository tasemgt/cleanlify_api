import sqlite3

DB_PATH = "hci.db"

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop existing tables if you're rebuilding (CAUTION: this clears data)
    cursor.execute("DROP TABLE IF EXISTS summaries")
    cursor.execute("DROP TABLE IF EXISTS cleans")
    cursor.execute("DROP TABLE IF EXISTS users")

    # --- Create users table ---
    cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        job_title TEXT,
        location TEXT,
        organisation TEXT,
        bio TEXT,
        create_date TEXT,
        member_plan TEXT,
        profession TEXT
    )
    """)

    # --- Create cleans table ---
    cursor.execute("""
    CREATE TABLE cleans (
        clean_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        clean_date TEXT,
        cleaning_mode TEXT,
        acceptance_ratio REAL,
        user_id INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)



    # --- Create summaries table ---
    cursor.execute("""
    CREATE TABLE summaries (
        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        clean_id INTEGER,
        column TEXT,
        total_values INTEGER,
        num_of_before_unique INTEGER,
        num_of_after_unique INTEGER,
        manual_corrections INTEGER,
        num_of_clusters INTEGER,
        num_of_majority INTEGER,
        total_num_of_single INTEGER,
        num_of_spell_check INTEGER,
        num_of_global_manual INTEGER,
        num_of_gkg INTEGER,
        num_of_llm INTEGER,
        acceptance_ratio REAL,
        FOREIGN KEY(clean_id) REFERENCES cleans(clean_id)
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized with users, cleans, and summaries tables.")

if __name__ == "__main__":
    initialize_db()