import sqlite3

DATABASE_FILE = "data/attendance.db"


def add_student(student_id, name):

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO students (student_id, name)
            VALUES (?, ?)
            """,
            (student_id, name)
        )

        connection.commit()

        print(f"✅ Student added: {student_id} - {name}")

    except sqlite3.IntegrityError:
        print(f"⚠️ Student {student_id} already exists.")

    connection.close()


print("================================")
print("ADD STUDENT")
print("================================")

student_id = input("Enter Student ID: ").strip()
name = input("Enter Student Name: ").strip()

if student_id and name:
    add_student(student_id, name)
else:
    print("❌ Student ID and name cannot be empty.")