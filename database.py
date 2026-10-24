import sqlite3
import os

DATABASE_FOLDER = "data"
DATABASE_FILE = os.path.join(
    DATABASE_FOLDER,
    "attendance.db"
)


def get_connection():
    os.makedirs(DATABASE_FOLDER, exist_ok=True)

    return sqlite3.connect(DATABASE_FILE)


def create_tables():

    connection = get_connection()
    cursor = connection.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (student_id)
                REFERENCES students(student_id)
        )
    """)

    connection.commit()
    connection.close()

    print("✅ Database tables created successfully")


if __name__ == "__main__":
    create_tables()
    