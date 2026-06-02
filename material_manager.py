import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel
from config import AppConfig


class MaterialFile(BaseModel):
    file_name: str
    file_path: str
    extension: str


class LectureInfo(BaseModel):
    lecture_id: str
    lecture_number: int
    title: str
    course_year: int
    available_files: List[MaterialFile] = []


class MaterialManager:
    def __init__(self, config: AppConfig):
        self.base_dir = Path(config.materials_dir)
        self.db_path = config.db_path
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_available_courses(self, student_profile) -> List[Dict[str, str]]:
        """
        Step 1: List courses the student can choose from.
        Takes the StudentProfile object we built previously.
        """
        if not student_profile or not student_profile.enrolled_courses:
            return []

        return [
            {"course_id": course.course_id, "course_name": course.course_name}
            for course in student_profile.enrolled_courses
        ]

    def get_course_lectures(self, course_id: str) -> List[LectureInfo]:
        """
        Step 2: List all available lectures for a selected course from the DB.
        We also fetch the course_academic_year to help construct the folder path later.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:

            cursor.execute(
                """
                SELECT l.lecture_unique_id, l.lecture_number, l.lecture_title_english, c.course_academic_year
                FROM Dim_Lectures l
                JOIN Dim_Courses c ON l.course_unique_id = c.Course_ID
                WHERE l.course_unique_id = ?
                ORDER BY l.lecture_number ASC
            """,
                (course_id,),
            )

            lectures = []
            for row in cursor.fetchall():
                lectures.append(
                    LectureInfo(
                        lecture_id=row["lecture_unique_id"],
                        lecture_number=int(row["lecture_number"]),
                        title=row["lecture_title_english"],
                        course_year=int(row["course_academic_year"]),
                    )
                )
            return lectures

        finally:
            conn.close()

    def get_lecture_materials(
        self, course_year: int, course_id: str, lecture_number: int
    ) -> List[MaterialFile]:
        """
        Step 3: Scan the local file system for materials belonging to this lecture.
        Expected structure: {base_dir}/{year}/{course_id}/{lecture_number}/
        """

        target_dir = self.base_dir / str(course_year) / course_id / str(lecture_number)

        if not target_dir.exists() or not target_dir.is_dir():

            return []

        materials = []

        for file_path in target_dir.iterdir():
            if file_path.is_file():
                materials.append(
                    MaterialFile(
                        file_name=file_path.name,
                        file_path=str(file_path.absolute()),
                        extension=file_path.suffix.lower(),
                    )
                )

        return materials

    def select_lecture(self, lecture: LectureInfo, course_id: str) -> LectureInfo:
        """
        Helper method to perform the selection and attach the files to the Lecture object.
        """
        files = self.get_lecture_materials(
            course_year=lecture.course_year,
            course_id=course_id,
            lecture_number=lecture.lecture_number,
        )
        lecture.available_files = files
        return lecture


if __name__ == "__main__":

    mock_path = Path("./materials/1/C001/1")
    mock_path.mkdir(parents=True, exist_ok=True)
    (mock_path / "Lecture_1_Slides.pdf").touch()
    (mock_path / "Lab_1_Code.py").touch()

    manager = MaterialManager()

    print("--- Step 1: User selects a course ---")

    selected_course_id = "C001"
    print(f"User selected: {selected_course_id}\n")

    print("--- Step 2: Fetching Available Lectures ---")
    lectures = manager.get_course_lectures(selected_course_id)

    if not lectures:
        print("No lectures found for this course in the database.")
    else:
        for idx, lec in enumerate(lectures):
            print(f"[{idx + 1}] Lecture {lec.lecture_number}: {lec.title}")

        print("\n--- Step 3: User selects a lecture to chat about ---")

        selected_lecture = lectures[0]
        print(f"User selected: {selected_lecture.title}\n")

        print("--- Step 4: Fetching Material Files for RAG ---")

        active_lecture_state = manager.select_lecture(
            selected_lecture, selected_course_id
        )

        if not active_lecture_state.available_files:
            print("No files found in the directory for this lecture.")
        else:
            print(
                f"Found {len(active_lecture_state.available_files)} files ready for the AI Agent:"
            )
            for file in active_lecture_state.available_files:
                print(f" -> {file.file_name} (Path: {file.file_path})")

            print(
                "\n✅ READY FOR STEP 3 (RAG). You can now pass these file paths to the Document Loaders!"
            )
