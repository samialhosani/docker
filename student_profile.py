import sqlite3
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class EnrolledCourse(BaseModel):
    course_id: str
    course_name: str
    department: str
    status: str


class UpcomingExam(BaseModel):
    exam_id: str
    course_id: str
    course_name: str
    exam_type: str
    exam_date: str
    max_grade: float
    days_until: int


class PendingAssignment(BaseModel):
    assignment_id: str
    course_id: str 
    course_name: str
    status: str
    deadline: str
    max_grade: float
    days_until: int


class StudentProfile(BaseModel):
    student_id: str
    full_name: str
    email: str
    system_type: str
    department: str
    academic_year: int
    gpa: float
    enrolled_courses: List[EnrolledCourse] = []
    upcoming_exams: List[UpcomingExam] = []
    pending_assignments: List[PendingAssignment] = []

    def get_context_string(self) -> str:
        """
        Generates a formatted string to inject into the LLM's system prompt.
        (We keep this focused on chat context, while the UI can use the exam/assignment lists).
        """
        courses = (
            ", ".join([c.course_name for c in self.enrolled_courses])
            or "No active enrollments"
        )
        return (
            f"Student Name: {self.full_name}\n"
            f"Academic Year: Year {self.academic_year}\n"
            f"Department: {self.department} ({self.system_type} System)\n"
            f"Current GPA: {self.gpa}\n"
            f"Enrolled Courses: {courses}"
        )


def parse_date(date_str: str) -> Optional[datetime]:
    """Tries to parse the CSV date format (e.g., 3/18/2025 or 2025-03-18)."""
    if not date_str:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_student_profile(identifier: str, db_path: str, reference_date: Optional[datetime] = None) -> Optional[StudentProfile]:
    if reference_date is None:
        reference_date = datetime.now()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Fetch Student Info
        cursor.execute("""
            SELECT Student_ID, student_full_name_english, student_email_address, 
                   student_system_type, student_department_english, student_academic_year, student_gpa
            FROM Dim_Students WHERE Student_ID = ? OR student_email_address = ?
        """, (identifier, identifier))
        student_row = cursor.fetchone()
        if not student_row: return None

        student_id = student_row["Student_ID"]

        # Fetch Enrollments
        cursor.execute("""
            SELECT e.Course_ID, c.course_name_english, c.course_department_english, e.status
            FROM Fact_Enrollment e
            JOIN Dim_Courses c ON e.Course_ID = c.Course_ID
            WHERE e.Student_ID = ? AND LOWER(e.status) = 'enrolled'
        """, (student_id,))
        
        enrolled_courses = []
        enrolled_course_ids = []
        for row in cursor.fetchall():
            enrolled_course_ids.append(row["Course_ID"])
            enrolled_courses.append(EnrolledCourse(
                course_id=row["Course_ID"], course_name=row["course_name_english"],
                department=row["course_department_english"], status=row["status"]
            ))

        upcoming_exams = []
        if enrolled_course_ids:
            placeholders = ",".join(["?"] * len(enrolled_course_ids))
            cursor.execute(f"""
                SELECT e.exam_id, e.course_unique_id, c.course_name_english, e.exam_type, e.exam_date, e.exam_max_grade
                FROM Fact_Exams e
                JOIN Dim_Courses c ON e.course_unique_id = c.Course_ID
                WHERE e.course_unique_id IN ({placeholders})
            """, enrolled_course_ids)

            for row in cursor.fetchall():
                exam_date_obj = parse_date(row["exam_date"])
                if exam_date_obj and exam_date_obj >= reference_date:
                    upcoming_exams.append(UpcomingExam(
                        exam_id=row["exam_id"], course_id=row["course_unique_id"],
                        course_name=row["course_name_english"], exam_type=row["exam_type"],
                        exam_date=row["exam_date"], max_grade=float(row["exam_max_grade"] or 0),
                        days_until=(exam_date_obj - reference_date).days
                    ))
        upcoming_exams.sort(key=lambda x: x.days_until)

        upcoming_assignments = []
        
        try:
            cursor.execute("""
                SELECT a.assignment_id, a.submission_status, a.deadline, a.assignment_max_grade,
                       c.Course_ID, c.course_name_english
                FROM Fact_Assignment_Submissions a
                JOIN Dim_Courses c ON a.course_unique_id = c.Course_ID
                WHERE a.student_unique_id = ? AND LOWER(a.submission_status) != 'submitted'
            """, (student_id,))

            for row in cursor.fetchall():
                deadline_obj = parse_date(row["deadline"])
                if deadline_obj and deadline_obj >= reference_date:
                    upcoming_assignments.append(PendingAssignment(
                        assignment_id=row["assignment_id"],
                        course_id=row["Course_ID"],
                        course_name=row["course_name_english"],
                        status=row["submission_status"],
                        deadline=row["deadline"],
                        max_grade=float(row["assignment_max_grade"] or 0),
                        days_until=(deadline_obj - reference_date).days
                    ))
            upcoming_assignments.sort(key=lambda x: x.days_until)
            
        except sqlite3.OperationalError as e:
            print(f"\n⚠️ Database warning: {e}")
            print("Please check your 'Fact_Assignment_Submissions.csv' file.")
            print("Make sure the column linking to the course is named 'course_unique_id'.")

        for row in cursor.fetchall():
            deadline_obj = parse_date(row["deadline"])
            if deadline_obj and deadline_obj >= reference_date:
                upcoming_assignments.append(PendingAssignment(
                    assignment_id=row["assignment_id"],
                    course_id=row["Course_ID"],          # Added
                    course_name=row["course_name_english"], # Added
                    status=row["submission_status"],
                    deadline=row["deadline"],
                    max_grade=float(row["assignment_max_grade"] or 0),
                    days_until=(deadline_obj - reference_date).days
                ))
        upcoming_assignments.sort(key=lambda x: x.days_until)

        return StudentProfile(
            student_id=student_id, full_name=student_row["student_full_name_english"],
            email=student_row["student_email_address"], system_type=student_row["student_system_type"],
            department=student_row["student_department_english"], academic_year=int(student_row["student_academic_year"]),
            gpa=float(student_row["student_gpa"]), enrolled_courses=enrolled_courses,
            upcoming_exams=upcoming_exams, pending_assignments=upcoming_assignments
        )
    finally:
        conn.close()