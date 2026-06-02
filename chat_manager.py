import sqlite3
import uuid
from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ChatDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Chat_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                course_id TEXT,
                student_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def save_message(self, session_id: str, course_id: str, student_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO Chat_History (session_id, course_id, student_id, role, content)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, course_id, student_id, role, content))
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, course_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """FIX: Retrieves only recent history for the SPECIFIC course to avoid cross-pollination & context overflow."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Using a subquery to get the last N records, then re-sorting them chronologically
        cursor = conn.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM Chat_History
                WHERE session_id = ? AND course_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
        """, (session_id, course_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

class ChatSession:
    def __init__(self, llm, profile, db, course_id: str, retriever=None, session_id: str = None):
        self.llm = llm
        self.profile = profile
        self.db = db
        self.course_id = course_id
        self.retriever = retriever
        self.session_id = session_id or str(uuid.uuid4())

        self.system_prompt_template = (
            "You are an AI teaching assistant for an educational platform.\n"
            "You are currently helping the student with the course: {course_id}.\n\n"
            "--- STUDENT PROFILE ---\n{profile_context}\n-----------------------\n\n"
            "--- COURSE MATERIAL CONTEXT ---\n{rag_context}\n-------------------------------\n\n"
            "Instructions:\n"
            "1. Use the provided Course Material Context to answer the user's question.\n"
            "2. If the answer is not in the material, say you don't know but try to guide them based on general academic knowledge.\n"
            "3. Keep your answers concise, encouraging, and directly helpful."
        )

    def send_message(self, user_input: str) -> str:
        # Save user message tied to the specific course
        self.db.save_message(self.session_id, self.course_id, self.profile.student_id, "user", user_input)

        rag_context = "No specific course material retrieved."
        if self.retriever:
            relevant_docs = self.retriever.invoke(user_input)
            if relevant_docs:
                rag_context = "\n\n".join([
                    f"Excerpt from {doc.metadata.get('source', 'Unknown')}:\n{doc.page_content}"
                    for doc in relevant_docs
                ])

        system_msg = SystemMessage(
            content=self.system_prompt_template.format(
                course_id=self.course_id,
                profile_context=self.profile.get_context_string(),
                rag_context=rag_context,
            )
        )

        # Limit fetched history to the last 10 messages of THIS course
        raw_history = self.db.get_history(self.session_id, self.course_id, limit=10)
        
        messages = [system_msg]
        for msg in raw_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        response = self.llm.invoke(messages)
        ai_reply = response.content

        # Save AI response
        self.db.save_message(self.session_id, self.course_id, self.profile.student_id, "ai", ai_reply)

        return ai_reply