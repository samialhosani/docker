import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# LangChain Agent Components
from langchain_core.tools import StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Local Modules
from student_profile import StudentProfile, get_student_profile
from rag_manager import RAGManager
from chat_manager import ChatDatabase
from providers import LLMProviderFactory
from config import load_config

# --- Tool Input Schemas ---
class SearchInput(BaseModel):
    query: str = Field(description="The specific academic question or topic to search for in the course materials.")
    course_id: str = Field(description="The ID of the course (e.g., 'C001').")

class EducationAgent:
    """An autonomous Agent that uses tools to fetch RAG context and database records."""
    
    def __init__(self, llm, profile: StudentProfile, db: ChatDatabase, rag_manager: RAGManager, session_id: str = None):
        self.llm = llm
        self.profile = profile
        self.db = db
        self.rag_manager = rag_manager
        self.session_id = session_id or str(uuid.uuid4())

        self.tools = [
            StructuredTool.from_function(
                func=self._search_course_materials,
                name="search_course_materials",
                description="Search the vector database for course materials, lecture notes, and syllabus info to answer academic questions.",
                args_schema=SearchInput
            ),
            StructuredTool.from_function(
                func=self._get_upcoming_deadlines,
                name="get_upcoming_deadlines",
                description="Fetch the student's upcoming exams and pending assignments.",
            ),
            StructuredTool.from_function(
                func=self._get_student_profile_info,
                name="get_student_profile_info",
                description="Get general profile information about the student like GPA, department, and enrolled courses.",
            )
        ]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an AI teaching assistant for an educational platform.\n"
             "You are currently helping the student: {student_name}.\n\n"
             "Guidelines:\n"
             "1. ALWAYS use the 'search_course_materials' tool when the student asks about course content, concepts, or lectures. Pass the specific course_id they are asking about.\n"
             "2. Use the 'get_upcoming_deadlines' tool when they ask about what's due, assignments, or exams.\n"
             "3. Use 'get_student_profile_info' for general questions about their academic status.\n"
             "4. Do not make up answers about deadlines or materials. If a tool returns no data, inform the student honestly.\n"
             "5. Keep responses encouraging, academic, and concise."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    
    def _search_course_materials(self, query: str, course_id: str) -> str:
        """Tool: Uses RAGManager to search for course concepts."""
        print(f"\n🔍 [Tool Execution] Searching RAG for: '{query}' in {course_id}...")
        try:
            retriever = self.rag_manager.get_retriever(course_id)
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant course material found in the database."
            
            context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nExcerpt: {doc.page_content}" for doc in docs])
            return context
        except Exception as e:
            return f"Error searching materials: {str(e)}"

    def _get_upcoming_deadlines(self) -> str:
        """Tool: Retrieves Exams and Assignments from StudentProfile."""
        print("\n📅 [Tool Execution] Fetching student deadlines...")
        lines = []
        if self.profile.upcoming_exams:
            lines.append("Upcoming Exams:")
            for ex in self.profile.upcoming_exams:
                lines.append(f"- {ex.course_name} ({ex.exam_type}) on {ex.exam_date} (in {ex.days_until} days)")
        else:
            lines.append("No upcoming exams.")
            
        lines.append("")
        if self.profile.pending_assignments:
            lines.append("Pending Assignments:")
            for pa in self.profile.pending_assignments:
                lines.append(f"- {pa.course_name}: Assignment {pa.assignment_id} due {pa.deadline} (in {pa.days_until} days) [Status: {pa.status}]")
        else:
            lines.append("No pending assignments.")
            
        return "\n".join(lines)

    def _get_student_profile_info(self) -> str:
        """Tool: Retrieves basic academic profile details."""
        print("\n🎓 [Tool Execution] Fetching student profile info...")
        return self.profile.get_context_string()


    def send_message(self, user_input: str, course_id: str = "GENERAL") -> str:
        """Sends a message to the agent, handles DB history, and returns the response."""
        
        self.db.save_message(self.session_id, course_id, self.profile.student_id, "user", user_input)
        
        raw_history = self.db.get_history(self.session_id, course_id, limit=10)
        chat_history = []
        for msg in raw_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "student_name": self.profile.full_name
            })
            ai_reply = response["output"]
        except Exception as e:
            print(f"\n❌ Agent Execution Error: {e}")
            ai_reply = "I'm sorry, but I encountered an error while processing your request. Please try again."

        self.db.save_message(self.session_id, course_id, self.profile.student_id, "ai", ai_reply)

        return ai_reply


if __name__ == "__main__":
    from datetime import datetime

    print("1. Loading Config and LLM...")
    config = load_config()
    
    llm = LLMProviderFactory.create_llm(config.active_llm_config)

    print("2. Initializing Databases & Managers...")
    db = ChatDatabase(config.db_path)
    rag = RAGManager(config.vector_db_path)
    
    print("3. Fetching Student Profile (C007)...")
    simulate_today = datetime(2025, 3, 10) 
    profile = get_student_profile("C007", db_path=config.db_path, reference_date=simulate_today)
    
    if not profile:
        print("❌ Student not found. Please ensure data.py has been run.")
        exit()

    print("\n" + "="*50)
    print("🤖 Education Agent Initialized!")
    print(f"👤 Logged in as: {profile.full_name}")
    print("💡 Try asking:")
    print("   - 'What is my GPA?'")
    print("   - 'Do I have any pending assignments?'")
    print("   - 'What is Course C001 about?'")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")

    agent = EducationAgent(llm=llm, profile=profile, db=db, rag_manager=rag)

    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() in ["quit", "exit"]:
            print("Ending session.")
            break

        ai_response = agent.send_message(user_msg, course_id="GENERAL")
        
        print(f"\n🎓 AI Assistant: {ai_response}")