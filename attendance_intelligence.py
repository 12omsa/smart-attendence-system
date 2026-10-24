import sqlite3
from datetime import datetime, date, timedelta
import math


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_FILE = "data/attendance.db"

DEFAULT_REQUIRED_PERCENTAGE = 75.0


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DATABASE_FILE
    )


# ============================================================
# CREATE INTELLIGENCE TABLES
# ============================================================

def initialize_intelligence_database():

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # SEMESTERS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS semesters (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            semester_name TEXT NOT NULL,

            start_date TEXT NOT NULL,

            end_date TEXT NOT NULL,

            required_percentage REAL DEFAULT 75.0

        )
        """
    )


    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            semester_id INTEGER NOT NULL,

            subject_name TEXT NOT NULL,

            total_lectures INTEGER DEFAULT 45,

            FOREIGN KEY (semester_id)
                REFERENCES semesters(id)

        )
        """
    )


    # --------------------------------------------------------
    # TIMETABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS timetable (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            subject_id INTEGER NOT NULL,

            weekday INTEGER NOT NULL,

            lectures_per_day INTEGER DEFAULT 1,

            FOREIGN KEY (subject_id)
                REFERENCES subjects(id)

        )
        """
    )


    # --------------------------------------------------------
    # SUBJECT ATTENDANCE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS subject_attendance (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT NOT NULL,

            subject_id INTEGER NOT NULL,

            lecture_date TEXT NOT NULL,

            lecture_number INTEGER DEFAULT 1,

            status TEXT NOT NULL,

            UNIQUE(
                student_id,
                subject_id,
                lecture_date,
                lecture_number
            )

        )
        """
    )


    connection.commit()
    connection.close()


# ============================================================
# ADD SEMESTER
# ============================================================

def create_semester(
    semester_name,
    start_date,
    end_date,
    required_percentage=75.0
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO semesters
        (
            semester_name,
            start_date,
            end_date,
            required_percentage
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            semester_name,
            start_date,
            end_date,
            required_percentage
        )
    )

    semester_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return semester_id


# ============================================================
# ADD SUBJECT
# ============================================================

def create_subject(
    semester_id,
    subject_name,
    total_lectures=45
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO subjects
        (
            semester_id,
            subject_name,
            total_lectures
        )

        VALUES (?, ?, ?)
        """,
        (
            semester_id,
            subject_name,
            total_lectures
        )
    )

    subject_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return subject_id


# ============================================================
# ADD TIMETABLE DAY
# ============================================================

def add_timetable_day(
    subject_id,
    weekday,
    lectures_per_day=1
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO timetable
        (
            subject_id,
            weekday,
            lectures_per_day
        )

        VALUES (?, ?, ?)
        """,
        (
            subject_id,
            weekday,
            lectures_per_day
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# COUNT EXPECTED LECTURES
# ============================================================

def calculate_expected_lectures(
    subject_id,
    semester_id,
    until_date=None
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            start_date,
            end_date
        FROM semesters
        WHERE id = ?
        """,
        (semester_id,)
    )

    semester = cursor.fetchone()

    if not semester:

        connection.close()

        return 0


    start_date = datetime.strptime(
        semester[0],
        "%Y-%m-%d"
    ).date()


    end_date = datetime.strptime(
        semester[1],
        "%Y-%m-%d"
    ).date()


    if until_date is None:

        until_date = date.today()

    elif isinstance(
        until_date,
        str
    ):

        until_date = datetime.strptime(
            until_date,
            "%Y-%m-%d"
        ).date()


    # Never count beyond semester

    until_date = min(
        until_date,
        end_date
    )


    if until_date < start_date:

        connection.close()

        return 0


    cursor.execute(
        """
        SELECT
            weekday,
            lectures_per_day
        FROM timetable
        WHERE subject_id = ?
        """,
        (subject_id,)
    )

    timetable = cursor.fetchall()

    connection.close()


    if not timetable:

        return 0


    schedule = {
        weekday: lectures
        for weekday, lectures
        in timetable
    }


    total = 0

    current = start_date

    while current <= until_date:

        weekday = current.weekday()

        if weekday in schedule:

            total += schedule[
                weekday
            ]

        current += timedelta(
            days=1
        )


    # Never exceed planned subject lectures

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT total_lectures
        FROM subjects
        WHERE id = ?
        """,
        (subject_id,)
    )

    subject = cursor.fetchone()

    connection.close()


    if subject:

        total = min(
            total,
            subject[0]
        )


    return total


# ============================================================
# GET STUDENT ATTENDANCE
# ============================================================

def get_student_subject_attendance(
    student_id,
    subject_id
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*)

        FROM subject_attendance

        WHERE student_id = ?

        AND subject_id = ?

        AND status = 'Present'
        """,
        (
            student_id,
            subject_id
        )
    )

    result = cursor.fetchone()

    connection.close()

    return result[0] if result else 0


# ============================================================
# RECORD SUBJECT ATTENDANCE
# ============================================================

def record_subject_attendance(
    student_id,
    subject_id,
    lecture_date=None,
    lecture_number=1,
    status="Present"
):

    if lecture_date is None:

        lecture_date = date.today().strftime(
            "%Y-%m-%d"
        )


    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO subject_attendance
        (
            student_id,
            subject_id,
            lecture_date,
            lecture_number,
            status
        )

        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student_id,
            subject_id,
            lecture_date,
            lecture_number,
            status
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# CALCULATE ATTENDANCE PERCENTAGE
# ============================================================

def calculate_percentage(
    attended,
    conducted
):

    if conducted <= 0:

        return 100.0

    return round(
        (attended / conducted) * 100,
        2
    )


# ============================================================
# CAN STILL REACH REQUIRED %
# ============================================================

def calculate_required_future_lectures(
    attended,
    conducted,
    required_percentage=75.0
):

    if conducted <= 0:

        return 0


    current_percentage = (
        attended / conducted
    ) * 100


    if current_percentage >= required_percentage:

        return 0


    # Solve:
    #
    # (attended + x)
    # ---------------- >= required
    # (conducted + x)
    #
    # x = required*c - attended
    #     -----------------------
    #     1 - required


    required = (
        required_percentage / 100
    )


    denominator = (
        1 - required
    )


    if denominator <= 0:

        return None


    x = (
        required * conducted
        -
        attended
    ) / denominator


    return max(
        0,
        math.ceil(x)
    )


# ============================================================
# DEFUALTER CHECK
# ============================================================

def can_reach_required_percentage(
    attended,
    conducted,
    total_semester_lectures,
    required_percentage=75.0
):

    remaining = max(
        0,
        total_semester_lectures
        -
        conducted
    )


    # Best possible case:
    # student attends every remaining lecture.

    maximum_attendance = (
        attended
        +
        remaining
    )


    maximum_percentage = (
        maximum_attendance
        /
        total_semester_lectures
    ) * 100


    return (
        maximum_percentage
        >=
        required_percentage
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk(
    attended,
    conducted,
    total_semester_lectures,
    required_percentage=75.0
):

    percentage = calculate_percentage(
        attended,
        conducted
    )


    # --------------------------------------------------------
    # No lectures yet
    # --------------------------------------------------------

    if conducted == 0:

        return {
            "level": "LOW",
            "label": "No attendance data",
            "percentage": percentage
        }


    # --------------------------------------------------------
    # DEFINITE DEFAULTER
    # --------------------------------------------------------

    if not can_reach_required_percentage(

        attended,
        conducted,
        total_semester_lectures,
        required_percentage

    ):

        return {

            "level": "DEFAULTER",

            "label":
                "75% cannot be reached",

            "percentage":
                percentage

        }


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if percentage >= 85:

        level = "LOW"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    elif percentage >= 75:

        level = "MEDIUM"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    else:

        level = "HIGH"


    return {

        "level":
            level,

        "label":
            level,

        "percentage":
            percentage

    }


# ============================================================
# COMPLETE SUBJECT ANALYSIS
# ============================================================

def analyze_student_subject(
    student_id,
    subject_id,
    semester_id
):

    connection = get_connection()
    cursor = connection.cursor()


    # --------------------------------------------------------
    # SUBJECT
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            subject_name,
            total_lectures
        FROM subjects
        WHERE id = ?
        """,
        (subject_id,)
    )

    subject = cursor.fetchone()


    if not subject:

        connection.close()

        return None


    subject_name = subject[0]

    total_lectures = subject[1]


    # --------------------------------------------------------
    # SEMESTER
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            required_percentage
        FROM semesters
        WHERE id = ?
        """,
        (semester_id,)
    )

    semester = cursor.fetchone()


    required_percentage = (
        semester[0]
        if semester
        else DEFAULT_REQUIRED_PERCENTAGE
    )


    connection.close()


    # --------------------------------------------------------
    # EXPECTED LECTURES
    # --------------------------------------------------------

    conducted = calculate_expected_lectures(

        subject_id,
        semester_id

    )


    # --------------------------------------------------------
    # ATTENDED
    # --------------------------------------------------------

    attended = get_student_subject_attendance(

        student_id,
        subject_id

    )


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    attended = min(
        attended,
        conducted
    )


    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    percentage = calculate_percentage(

        attended,
        conducted

    )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = calculate_risk(

        attended,
        conducted,
        total_lectures,
        required_percentage

    )


    # --------------------------------------------------------
    # REQUIRED FUTURE LECTURES
    # --------------------------------------------------------

    future_required = (
        calculate_required_future_lectures(

            attended,
            conducted,
            required_percentage

        )
    )


    # --------------------------------------------------------
    # REMAINING LECTURES
    # --------------------------------------------------------

    remaining = max(

        0,

        total_lectures
        -
        conducted

    )


    # --------------------------------------------------------
    # BEST POSSIBLE FINAL ATTENDANCE
    # --------------------------------------------------------

    best_possible = (

        attended
        +
        remaining

    )


    best_possible_percentage = (

        best_possible
        /
        total_lectures

    ) * 100


    # --------------------------------------------------------
    # WORST POSSIBLE FINAL ATTENDANCE
    # --------------------------------------------------------

    worst_possible_percentage = (

        attended
        /
        total_lectures

    ) * 100


    return {

        "student_id":
            student_id,

        "subject_id":
            subject_id,

        "subject_name":
            subject_name,

        "total_lectures":
            total_lectures,

        "conducted":
            conducted,

        "attended":
            attended,

        "absent":
            conducted - attended,

        "remaining":
            remaining,

        "percentage":
            percentage,

        "required_percentage":
            required_percentage,

        "risk":
            risk["level"],

        "risk_label":
            risk["label"],

        "future_required":
            future_required,

        "best_possible_percentage":
            round(
                best_possible_percentage,
                2
            ),

        "worst_possible_percentage":
            round(
                worst_possible_percentage,
                2
            )

    }


# ============================================================
# ALL SUBJECT ANALYSIS FOR STUDENT
# ============================================================

def analyze_student(
    student_id,
    semester_id
):

    connection = get_connection()
    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            id
        FROM subjects
        WHERE semester_id = ?
        """,
        (semester_id,)
    )


    subjects = cursor.fetchall()

    connection.close()


    results = []


    for subject in subjects:

        result = analyze_student_subject(

            student_id,

            subject[0],

            semester_id

        )


        if result:

            results.append(
                result
            )


    return results


# ============================================================
# SEMESTER SUMMARY
# ============================================================

def get_student_semester_summary(
    student_id,
    semester_id
):

    results = analyze_student(

        student_id,
        semester_id

    )


    if not results:

        return {

            "student_id":
                student_id,

            "subjects":
                [],

            "overall_risk":
                "NO DATA"

        }


    total_conducted = sum(
        item["conducted"]
        for item in results
    )


    total_attended = sum(
        item["attended"]
        for item in results
    )


    overall_percentage = (
        calculate_percentage(

            total_attended,

            total_conducted

        )
    )


    defaulter_count = sum(

        1

        for item in results

        if item["risk"]
        ==
        "DEFAULTER"

    )


    high_count = sum(

        1

        for item in results

        if item["risk"]
        ==
        "HIGH"

    )


    if defaulter_count > 0:

        overall_risk = "DEFAULTER"


    elif high_count > 0:

        overall_risk = "HIGH"


    elif overall_percentage < 75:

        overall_risk = "MEDIUM"


    else:

        overall_risk = "LOW"


    return {

        "student_id":
            student_id,

        "subjects":
            results,

        "total_conducted":
            total_conducted,

        "total_attended":
            total_attended,

        "overall_percentage":
            overall_percentage,

        "overall_risk":
            overall_risk

    }