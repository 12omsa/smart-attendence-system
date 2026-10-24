
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    Response,
    session,
    flash
)

import sqlite3
import csv
import io
import os
import math

from datetime import datetime, timedelta
from functools import wraps

from register import register_face
from recognition import generate_frames


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "AI_ATTEND_SECRET",
    "AI_Attend_Security_Key_2026"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

DATABASE_FILE = "data/attendance.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    os.makedirs("data", exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE HELPERS
# ============================================================

def column_exists(cursor, table_name, column_name):

    cursor.execute(
        f"PRAGMA table_info({table_name})"
    )

    columns = cursor.fetchall()

    return any(
        column["name"] == column_name
        for column in columns
    )


def add_column_if_missing(
    cursor,
    table_name,
    column_name,
    column_definition
):

    if not column_exists(
        cursor,
        table_name,
        column_name
    ):

        cursor.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name}
            {column_definition}
            """
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Students
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT UNIQUE NOT NULL,

            name TEXT NOT NULL,

            created_at
            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT NOT NULL,

            date TEXT NOT NULL,

            time TEXT NOT NULL,

            status TEXT NOT NULL
        )
    """)

    # Add subject_code to old attendance table
    add_column_if_missing(
        cursor,
        "attendance",
        "subject_code",
        "TEXT"
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            subject_code TEXT UNIQUE NOT NULL,

            subject_name TEXT NOT NULL,

            total_lectures INTEGER DEFAULT 45,

            min_attendance REAL DEFAULT 75.0,

            created_at
            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # Timetable
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timetable (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            subject_code TEXT NOT NULL,

            day_of_week INTEGER NOT NULL,

            start_time TEXT,

            end_time TEXT,

            lectures_per_day INTEGER DEFAULT 1,

            FOREIGN KEY(subject_code)
            REFERENCES subjects(subject_code)
        )
    """)

    # --------------------------------------------------------
    # Semester settings
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS semester_settings (

            id INTEGER PRIMARY KEY CHECK(id = 1),

            semester_name TEXT DEFAULT 'Semester 1',

            start_date TEXT,

            end_date TEXT,

            total_working_days INTEGER DEFAULT 0
        )
    """)

    # --------------------------------------------------------
    # Default semester settings
    # --------------------------------------------------------

    cursor.execute("""
        SELECT id
        FROM semester_settings
        WHERE id = 1
    """)

    settings = cursor.fetchone()

    if settings is None:

        today = datetime.now().date()

        semester_start = today.replace(
            month=7,
            day=1
        )

        semester_end = today.replace(
            month=12,
            day=31
        )

        cursor.execute("""
            INSERT INTO semester_settings
            (
                id,
                semester_name,
                start_date,
                end_date
            )
            VALUES
            (
                1,
                'Semester 1',
                ?,
                ?
            )
        """, (
            semester_start.strftime("%Y-%m-%d"),
            semester_end.strftime("%Y-%m-%d")
        ))

    connection.commit()
    connection.close()


# ============================================================
# LOGIN
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if not session.get("logged_in"):

            return redirect("/login")

        return function(*args, **kwargs)

    return decorated_function


# ============================================================
# SEMESTER SETTINGS
# ============================================================

def get_semester_settings():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM semester_settings
        WHERE id = 1
    """)

    settings = cursor.fetchone()

    connection.close()

    return settings


# ============================================================
# DATE HELPERS
# ============================================================

def get_date_range(
    start_date,
    end_date
):

    dates = []

    current = start_date

    while current <= end_date:

        dates.append(current)

        current += timedelta(days=1)

    return dates


# ============================================================
# CALCULATE SCHEDULED LECTURES
# ============================================================

def calculate_scheduled_lectures(
    subject_code,
    start_date,
    end_date
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            day_of_week,
            lectures_per_day
        FROM timetable
        WHERE subject_code = ?
    """, (subject_code,))

    timetable_rows = cursor.fetchall()

    connection.close()

    if not timetable_rows:

        return 0

    schedule = {}

    for row in timetable_rows:

        schedule[
            row["day_of_week"]
        ] = row["lectures_per_day"]

    total = 0

    current = start_date

    while current <= end_date:

        weekday = current.weekday()

        if weekday in schedule:

            total += schedule[weekday]

        current += timedelta(days=1)

    return total


# ============================================================
# ATTENDANCE RISK CALCULATION
# ============================================================

def calculate_risk(
    attended,
    scheduled,
    total_semester_lectures
):

    if scheduled <= 0:

        return {
            "percentage": 0,
            "risk": "No Data",
            "risk_class": "secondary",
            "defaulter": False,
            "remaining": total_semester_lectures,
            "prediction": "No timetable data"
        }

    percentage = round(
        (attended / scheduled) * 100,
        1
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if percentage >= 75:

        risk = "Low"
        risk_class = "success"

    elif percentage >= 60:

        risk = "Medium"
        risk_class = "warning"

    else:

        risk = "High"
        risk_class = "danger"

    # --------------------------------------------------------
    # Defaulter
    # --------------------------------------------------------

    defaulter = percentage < 75

    # --------------------------------------------------------
    # Remaining lectures
    # --------------------------------------------------------

    remaining = max(
        total_semester_lectures - scheduled,
        0
    )

    # --------------------------------------------------------
    # Future prediction
    #
    # We calculate how many future lectures must be
    # attended continuously to reach 75%.
    # --------------------------------------------------------

    required_attendance = 0

    if total_semester_lectures > 0:

        required_attendance = math.ceil(
            0.75 * total_semester_lectures
        )

    additional_needed = max(
        required_attendance - attended,
        0
    )

    if additional_needed == 0:

        prediction = (
            "Student is on track for the "
            "minimum attendance requirement."
        )

    elif additional_needed <= remaining:

        prediction = (
            f"Student must attend the next "
            f"{additional_needed} lectures "
            f"to reach approximately 75%."
        )

    else:

        prediction = (
            "Student cannot reach 75% even "
            "by attending all remaining lectures."
        )

    return {
        "percentage": percentage,
        "risk": risk,
        "risk_class": risk_class,
        "defaulter": defaulter,
        "remaining": remaining,
        "prediction": prediction
    }


# ============================================================
# SUBJECT ATTENDANCE
# ============================================================

def get_subject_attendance(
    student_id,
    subject_code,
    start_date,
    current_date,
    total_semester_lectures
):

    scheduled = calculate_scheduled_lectures(
        subject_code,
        start_date,
        current_date
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM attendance
        WHERE student_id = ?
        AND subject_code = ?
        AND status = 'Present'
        AND date BETWEEN ? AND ?
    """, (
        student_id,
        subject_code,
        start_date.strftime("%Y-%m-%d"),
        current_date.strftime("%Y-%m-%d")
    ))

    result = cursor.fetchone()

    attended = result["total"]

    connection.close()

    return scheduled, attended


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
@login_required
def dashboard():

    connection = get_connection()
    cursor = connection.cursor()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # Total students
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM students
    """)

    total_students = cursor.fetchone()["total"]

    # --------------------------------------------------------
    # Present today
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(DISTINCT student_id) AS total
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (today,))

    present_today = cursor.fetchone()["total"]

    # --------------------------------------------------------
    # Attendance rate
    # --------------------------------------------------------

    if total_students > 0:

        attendance_rate = round(
            (
                present_today
                / total_students
            ) * 100,
            1
        )

    else:

        attendance_rate = 0

    # --------------------------------------------------------
    # Today's attendance
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            attendance.student_id,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status,
            attendance.subject_code
        FROM attendance
        LEFT JOIN students
        ON attendance.student_id =
           students.student_id
        WHERE attendance.date = ?
        ORDER BY attendance.time DESC
    """, (today,))

    today_attendance = cursor.fetchall()

    # --------------------------------------------------------
    # Last 7 days
    # --------------------------------------------------------

    chart_dates = []
    chart_present = []

    for i in range(6, -1, -1):

        current_date = (
            datetime.now()
            - timedelta(days=i)
        ).strftime("%Y-%m-%d")

        chart_dates.append(current_date)

        cursor.execute("""
            SELECT COUNT(DISTINCT student_id)
            AS total
            FROM attendance
            WHERE date = ?
            AND status = 'Present'
        """, (current_date,))

        count = cursor.fetchone()["total"]

        chart_present.append(count)

    # --------------------------------------------------------
    # Student ranking
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            students.name,
            COUNT(attendance.id) AS total
        FROM students
        LEFT JOIN attendance
        ON students.student_id =
           attendance.student_id
        AND attendance.status = 'Present'
        GROUP BY students.student_id
        ORDER BY total DESC
    """)

    student_ranking = cursor.fetchall()

    student_names = [
        row["name"]
        for row in student_ranking
    ]

    student_attendance = [
        row["total"]
        for row in student_ranking
    ]

    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM attendance
        WHERE status = 'Present'
    """)

    total_present_records = (
        cursor.fetchone()["total"]
    )

    connection.close()

    return render_template(
        "dashboard.html",
        total_students=total_students,
        present_today=present_today,
        attendance_rate=attendance_rate,
        today_attendance=today_attendance,
        chart_dates=chart_dates,
        chart_present=chart_present,
        student_names=student_names,
        student_attendance=student_attendance,
        total_present_records=total_present_records
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if session.get("logged_in"):

        return redirect("/")

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        admin_username = os.environ.get(
            "AI_ATTEND_ADMIN",
            "admin"
        )

        admin_password = os.environ.get(
            "AI_ATTEND_PASSWORD",
            "Admin@12345"
        )

        if (
            username == admin_username
            and password == admin_password
        ):

            session["logged_in"] = True
            session["username"] = username

            return redirect("/")

        flash(
            "Invalid username or password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ============================================================
# STUDENTS
# ============================================================

@app.route("/students")
@login_required
def students():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            student_id,
            name,
            created_at
        FROM students
        ORDER BY id DESC
    """)

    student_records = cursor.fetchall()

    connection.close()

    return render_template(
        "students.html",
        students=student_records
    )


# ============================================================
# ADD STUDENT
# ============================================================

@app.route(
    "/add-student",
    methods=["GET", "POST"]
)
@login_required
def add_student():

    if request.method == "POST":

        student_id = request.form.get(
            "student_id",
            ""
        ).strip()

        name = request.form.get(
            "name",
            ""
        ).strip()

        if not student_id or not name:

            flash(
                "Student ID and name are required."
            )

            return redirect(
                "/add-student"
            )

        connection = get_connection()
        cursor = connection.cursor()

        try:

            cursor.execute("""
                INSERT INTO students
                (
                    student_id,
                    name
                )
                VALUES (?, ?)
            """, (
                student_id,
                name
            ))

            connection.commit()

            flash(
                "Student added successfully."
            )

        except sqlite3.IntegrityError:

            flash(
                "Student ID already exists."
            )

        finally:

            connection.close()

        return redirect(
            "/students"
        )

    return render_template(
        "add_student.html"
    )


# ============================================================
# REGISTER FACE
# ============================================================

@app.route(
    "/register-face",
    methods=["GET", "POST"]
)
@login_required
def register_face_page():

    if request.method == "POST":

        student_id = request.form.get(
            "student_id",
            ""
        ).strip()

        student_name = request.form.get(
            "student_name",
            ""
        ).strip()

        if not student_id or not student_name:

            flash(
                "Student ID and name are required."
            )

            return redirect(
                "/register-face"
            )

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id
            FROM students
            WHERE student_id = ?
        """, (student_id,))

        existing_student = cursor.fetchone()

        if existing_student is None:

            try:

                cursor.execute("""
                    INSERT INTO students
                    (
                        student_id,
                        name
                    )
                    VALUES (?, ?)
                """, (
                    student_id,
                    student_name
                ))

                connection.commit()

            except sqlite3.IntegrityError:

                connection.close()

                flash(
                    "Unable to create student."
                )

                return redirect(
                    "/register-face"
                )

        else:

            cursor.execute("""
                UPDATE students
                SET name = ?
                WHERE student_id = ?
            """, (
                student_name,
                student_id
            ))

            connection.commit()

        connection.close()

        success = register_face(
            student_id,
            student_name
        )

        if success:

            flash(
                "Face registration completed successfully."
            )

        else:

            flash(
                "Face registration was cancelled or incomplete."
            )

        return redirect(
            "/student/" + student_id
        )

    return render_template(
        "register_face.html"
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

@app.route(
    "/student/<student_id>"
)
@login_required
def student_profile(student_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            student_id,
            name,
            created_at
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    if student is None:

        connection.close()

        flash(
            "Student not found."
        )

        return redirect(
            "/students"
        )

    # --------------------------------------------------------
    # Attendance history
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            date,
            time,
            status,
            subject_code
        FROM attendance
        WHERE student_id = ?
        ORDER BY date DESC, time DESC
    """, (student_id,))

    attendance_history = (
        cursor.fetchall()
    )

    # --------------------------------------------------------
    # Present days
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(DISTINCT date)
        AS total
        FROM attendance
        WHERE student_id = ?
        AND status = 'Present'
    """, (student_id,))

    present_days = cursor.fetchone()["total"]

    # --------------------------------------------------------
    # Total attendance days
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(DISTINCT date)
        AS total
        FROM attendance
        WHERE student_id = ?
    """, (student_id,))

    total_attendance_days = (
        cursor.fetchone()["total"]
    )

    if total_attendance_days > 0:

        attendance_percentage = round(
            (
                present_days
                / total_attendance_days
            ) * 100,
            1
        )

    else:

        attendance_percentage = 0

    # --------------------------------------------------------
    # Face registration
    # --------------------------------------------------------

    student_folder = os.path.join(
        "students",
        student_id
    )

    embedding_count = 0

    if os.path.exists(student_folder):

        embedding_count = len([
            filename
            for filename in os.listdir(
                student_folder
            )
            if filename.endswith(".npy")
        ])

    face_registered = (
        embedding_count > 0
    )

    connection.close()

    return render_template(
        "student_profile.html",
        student=student,
        attendance_history=attendance_history,
        present_days=present_days,
        total_attendance_days=total_attendance_days,
        attendance_percentage=attendance_percentage,
        face_registered=face_registered,
        embedding_count=embedding_count
    )


# ============================================================
# ATTENDANCE
# ============================================================

@app.route("/attendance")
@login_required
def attendance():

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    search = request.args.get(
        "search",
        ""
    ).strip()

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        SELECT
            attendance.student_id,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status,
            attendance.subject_code
        FROM attendance
        LEFT JOIN students
        ON attendance.student_id =
           students.student_id
        WHERE attendance.date = ?
    """

    parameters = [
        selected_date
    ]

    if search:

        query += """
            AND (
                attendance.student_id LIKE ?
                OR students.name LIKE ?
                OR attendance.subject_code LIKE ?
            )
        """

        search_value = (
            f"%{search}%"
        )

        parameters.extend([
            search_value,
            search_value,
            search_value
        ])

    query += """
        ORDER BY attendance.time DESC
    """

    cursor.execute(
        query,
        parameters
    )

    attendance_records = (
        cursor.fetchall()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM students
    """)

    total_students = (
        cursor.fetchone()["total"]
    )

    cursor.execute("""
        SELECT COUNT(DISTINCT student_id)
        AS total
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (selected_date,))

    present = (
        cursor.fetchone()["total"]
    )

    absent = max(
        total_students - present,
        0
    )

    connection.close()

    return render_template(
        "attendance.html",
        attendance_records=attendance_records,
        selected_date=selected_date,
        search=search,
        total_students=total_students,
        present=present,
        absent=absent
    )


# ============================================================
# LIVE CAMERA
# ============================================================

@app.route("/live-camera")
@login_required
def live_camera():

    return render_template(
        "live_camera.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video-feed")
@login_required
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ============================================================
# SUBJECT MANAGEMENT
# ============================================================

@app.route(
    "/subjects",
    methods=["GET", "POST"]
)
@login_required
def subjects():

    connection = get_connection()
    cursor = connection.cursor()

    if request.method == "POST":

        subject_code = request.form.get(
            "subject_code",
            ""
        ).strip().upper()

        subject_name = request.form.get(
            "subject_name",
            ""
        ).strip()

        total_lectures = request.form.get(
            "total_lectures",
            "45"
        ).strip()

        min_attendance = request.form.get(
            "min_attendance",
            "75"
        ).strip()

        try:

            total_lectures = int(
                total_lectures
            )

            min_attendance = float(
                min_attendance
            )

        except ValueError:

            total_lectures = 45
            min_attendance = 75

        if (
            not subject_code
            or not subject_name
        ):

            flash(
                "Subject code and name are required."
            )

        else:

            try:

                cursor.execute("""
                    INSERT INTO subjects
                    (
                        subject_code,
                        subject_name,
                        total_lectures,
                        min_attendance
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    subject_code,
                    subject_name,
                    total_lectures,
                    min_attendance
                ))

                connection.commit()

                flash(
                    "Subject added successfully."
                )

            except sqlite3.IntegrityError:

                flash(
                    "Subject code already exists."
                )

        connection.close()

        return redirect(
            "/subjects"
        )

    cursor.execute("""
        SELECT *
        FROM subjects
        ORDER BY subject_code
    """)

    subject_records = (
        cursor.fetchall()
    )

    connection.close()

    return render_template(
        "subjects.html",
        subjects=subject_records
    )


# ============================================================
# TIMETABLE
# ============================================================

@app.route(
    "/timetable",
    methods=["GET", "POST"]
)
@login_required
def timetable():

    connection = get_connection()
    cursor = connection.cursor()

    if request.method == "POST":

        subject_code = request.form.get(
            "subject_code",
            ""
        ).strip().upper()

        day_of_week = request.form.get(
            "day_of_week",
            ""
        ).strip()

        start_time = request.form.get(
            "start_time",
            ""
        ).strip()

        end_time = request.form.get(
            "end_time",
            ""
        ).strip()

        lectures_per_day = request.form.get(
            "lectures_per_day",
            "1"
        ).strip()

        try:

            day_of_week = int(
                day_of_week
            )

            lectures_per_day = int(
                lectures_per_day
            )

        except ValueError:

            day_of_week = 0
            lectures_per_day = 1

        if not subject_code:

            flash(
                "Please select a subject."
            )

        else:

            cursor.execute("""
                INSERT INTO timetable
                (
                    subject_code,
                    day_of_week,
                    start_time,
                    end_time,
                    lectures_per_day
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                subject_code,
                day_of_week,
                start_time,
                end_time,
                lectures_per_day
            ))

            connection.commit()

            flash(
                "Timetable entry added."
            )

        connection.close()

        return redirect(
            "/timetable"
        )

    cursor.execute("""
        SELECT *
        FROM subjects
        ORDER BY subject_code
    """)

    subject_records = (
        cursor.fetchall()
    )

    cursor.execute("""
        SELECT
            timetable.id,
            timetable.subject_code,
            subjects.subject_name,
            timetable.day_of_week,
            timetable.start_time,
            timetable.end_time,
            timetable.lectures_per_day
        FROM timetable
        LEFT JOIN subjects
        ON timetable.subject_code =
           subjects.subject_code
        ORDER BY
            timetable.day_of_week,
            timetable.start_time
    """)

    timetable_records = (
        cursor.fetchall()
    )

    connection.close()

    return render_template(
        "timetable.html",
        subjects=subject_records,
        timetable=timetable_records
    )


# ============================================================
# DELETE TIMETABLE ENTRY
# ============================================================

@app.route(
    "/delete-timetable/<int:entry_id>",
    methods=["POST"]
)
@login_required
def delete_timetable(entry_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM timetable
        WHERE id = ?
    """, (entry_id,))

    connection.commit()
    connection.close()

    flash(
        "Timetable entry deleted."
    )

    return redirect(
        "/timetable"
    )


# ============================================================
# SEMESTER SETTINGS PAGE
# ============================================================

@app.route(
    "/semester-settings",
    methods=["GET", "POST"]
)
@login_required
def semester_settings():

    if request.method == "POST":

        semester_name = request.form.get(
            "semester_name",
            "Semester 1"
        ).strip()

        start_date = request.form.get(
            "start_date",
            ""
        ).strip()

        end_date = request.form.get(
            "end_date",
            ""
        ).strip()

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE semester_settings
            SET
                semester_name = ?,
                start_date = ?,
                end_date = ?
            WHERE id = 1
        """, (
            semester_name,
            start_date,
            end_date
        ))

        connection.commit()
        connection.close()

        flash(
            "Semester settings updated."
        )

        return redirect(
            "/semester-settings"
        )

    settings = get_semester_settings()

    return render_template(
        "semester_settings.html",
        settings=settings
    )


# ============================================================
# SEMESTER RISK DASHBOARD
# ============================================================

@app.route(
    "/semester-attendance"
)
@login_required
def semester_attendance():

    settings = get_semester_settings()

    start_date = datetime.strptime(
        settings["start_date"],
        "%Y-%m-%d"
    ).date()

    semester_end = datetime.strptime(
        settings["end_date"],
        "%Y-%m-%d"
    ).date()

    today = datetime.now().date()

    current_date = min(
        today,
        semester_end
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM subjects
        ORDER BY subject_code
    """)

    subject_records = (
        cursor.fetchall()
    )

    cursor.execute("""
        SELECT
            student_id,
            name
        FROM students
        ORDER BY name
    """)

    student_records = (
        cursor.fetchall()
    )

    connection.close()

    reports = []

    # --------------------------------------------------------
    # Calculate every student / every subject
    # --------------------------------------------------------

    for student in student_records:

        student_subjects = []

        overall_scheduled = 0
        overall_attended = 0
        high_risk_count = 0

        for subject in subject_records:

            scheduled, attended = (
                get_subject_attendance(
                    student["student_id"],
                    subject["subject_code"],
                    start_date,
                    current_date,
                    subject["total_lectures"]
                )
            )

            risk = calculate_risk(
                attended,
                scheduled,
                subject["total_lectures"]
            )

            if risk["risk"] == "High":

                high_risk_count += 1

            overall_scheduled += scheduled
            overall_attended += attended

            student_subjects.append({

                "subject_code":
                    subject["subject_code"],

                "subject_name":
                    subject["subject_name"],

                "total_lectures":
                    subject["total_lectures"],

                "scheduled":
                    scheduled,

                "attended":
                    attended,

                "percentage":
                    risk["percentage"],

                "risk":
                    risk["risk"],

                "risk_class":
                    risk["risk_class"],

                "defaulter":
                    risk["defaulter"],

                "remaining":
                    risk["remaining"],

                "prediction":
                    risk["prediction"]
            })

        if overall_scheduled > 0:

            overall_percentage = round(
                (
                    overall_attended
                    / overall_scheduled
                ) * 100,
                1
            )

        else:

            overall_percentage = 0

        if high_risk_count > 0:

            overall_risk = "High"
            overall_class = "danger"

        elif overall_percentage < 75:

            overall_risk = "Medium"
            overall_class = "warning"

        else:

            overall_risk = "Low"
            overall_class = "success"

        reports.append({

            "student_id":
                student["student_id"],

            "name":
                student["name"],

            "subjects":
                student_subjects,

            "overall_scheduled":
                overall_scheduled,

            "overall_attended":
                overall_attended,

            "overall_percentage":
                overall_percentage,

            "overall_risk":
                overall_risk,

            "overall_class":
                overall_class
        })

    return render_template(
        "semester_attendance.html",
        settings=settings,
        reports=reports,
        subjects=subject_records
    )


# ============================================================
# STUDENT SEMESTER PROFILE
# ============================================================

@app.route(
    "/student/<student_id>/semester"
)
@login_required
def student_semester_profile(student_id):

    settings = get_semester_settings()

    start_date = datetime.strptime(
        settings["start_date"],
        "%Y-%m-%d"
    ).date()

    semester_end = datetime.strptime(
        settings["end_date"],
        "%Y-%m-%d"
    ).date()

    current_date = min(
        datetime.now().date(),
        semester_end
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    if student is None:

        connection.close()

        flash(
            "Student not found."
        )

        return redirect(
            "/students"
        )

    cursor.execute("""
        SELECT *
        FROM subjects
        ORDER BY subject_code
    """)

    subjects = cursor.fetchall()

    connection.close()

    subject_reports = []

    for subject in subjects:

        scheduled, attended = (
            get_subject_attendance(
                student_id,
                subject["subject_code"],
                start_date,
                current_date,
                subject["total_lectures"]
            )
        )

        risk = calculate_risk(
            attended,
            scheduled,
            subject["total_lectures"]
        )

        subject_reports.append({

            "subject_code":
                subject["subject_code"],

            "subject_name":
                subject["subject_name"],

            "total_lectures":
                subject["total_lectures"],

            "scheduled":
                scheduled,

            "attended":
                attended,

            "percentage":
                risk["percentage"],

            "risk":
                risk["risk"],

            "risk_class":
                risk["risk_class"],

            "defaulter":
                risk["defaulter"],

            "remaining":
                risk["remaining"],

            "prediction":
                risk["prediction"]
        })

    return render_template(
        "student_semester.html",
        student=student,
        subject_reports=subject_reports,
        settings=settings
    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
@login_required
def reports():

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM students
    """)

    total_students = (
        cursor.fetchone()["total"]
    )

    cursor.execute("""
        SELECT COUNT(DISTINCT student_id)
        AS total
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (selected_date,))

    present = (
        cursor.fetchone()["total"]
    )

    absent = max(
        total_students - present,
        0
    )

    if total_students >  0:

        attendance_rate = round(
            (
                present
                / total_students
            ) * 100,
            1
        )

    else:

        attendance_rate = 0

    cursor.execute("""
        SELECT
            students.student_id,
            students.name,
            COUNT(attendance.id)
            AS present_days
        FROM students
        LEFT JOIN attendance
        ON students.student_id =
           attendance.student_id
        AND attendance.status = 'Present'
        GROUP BY students.student_id
        ORDER BY students.name
    """)

    student_reports = (
        cursor.fetchall()
    )

    connection.close()

    return render_template(
        "reports.html",
        selected_date=selected_date,
        total_students=total_students,
        present=present,
        absent=absent,
        attendance_rate=attendance_rate,
        student_reports=student_reports
    )


# ============================================================
# EXPORT REPORT
# ============================================================

@app.route("/export-report")
@login_required
def export_report():

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            attendance.student_id,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status,
            attendance.subject_code
        FROM attendance
        LEFT JOIN students
        ON attendance.student_id =
           students.student_id
        WHERE attendance.date = ?
        ORDER BY attendance.time
    """, (selected_date,))

    records = cursor.fetchall()

    connection.close()

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Student ID",
        "Name",
        "Date",
        "Time",
        "Subject",
        "Status"
    ])

    for record in records:

        writer.writerow([
            record["student_id"],
            record["name"],
            record["date"],
            record["time"],
            record["subject_code"] or "",
            record["status"]
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers[
        "Content-Disposition"
    ] = (
        f"attachment; "
        f"filename=attendance_{selected_date}.csv"
    )

    return response


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        threaded=True
    )

