import streamlit as st
import pdfplumber
import os
import json
import re
import tempfile
import subprocess
import io
import pandas as pd
import PyPDF2  # For extracting outlines
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY2")

# --- Initialize Gemini LLM via LangChain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

# --- Helper Function: Extract Text from PDF ---
def extract_text_from_pdf(pdf_file, start_page=0, num_pages=None):
    text = ""
    pdf_file.seek(0)  # Reset pointer
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        if num_pages is None:
            num_pages = total_pages - start_page
        for i in range(start_page, min(start_page + num_pages, total_pages)):
            page_text = pdf.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Helper Function: Parse JSON from Response ---
def parse_json_response(response_text):
    try:
        cleaned_text = re.sub(r'[\x00-\x1F]+', '', response_text)
        json_str = re.search(r'\{.*\}', cleaned_text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        st.error("Error parsing JSON: " + str(e))
        return None

# --- Function to Extract Outlines (Bookmarks) from a PDF using PyPDF2 ---
def extract_outlines(pdf_file):
    pdf_file.seek(0)
    reader = PyPDF2.PdfReader(pdf_file)
    try:
        # Use the new attribute "outline"
        outlines = reader.outline
    except Exception as e:
        st.error("Could not extract outlines: " + str(e))
        return []
    def flatten_outlines(outlines):
        items = []
        for item in outlines:
            if isinstance(item, list):
                items.extend(flatten_outlines(item))
            else:
                try:
                    page_num = reader.get_destination_page_number(item)
                except Exception:
                    page_num = None
                items.append((item.title, page_num))
        return items
    return flatten_outlines(outlines)

# --- Combined Chain for Generating Personalized Notes and Questions ---
combined_prompt = PromptTemplate(
    input_variables=["text", "style"],
    template=(
        "You are an educational AI assistant. Given the following content, generate personalized notes in the style specified and 5 educational questions based on it.\n\n"
        "Style: {style}\n\n"
        "Return ONLY a valid JSON object with no additional text. The JSON must have exactly two keys: 'notes' and 'questions'.\n"
        "The 'notes' value should be a string containing the personalized notes in the given style.\n"
        "The 'questions' value should be an array of exactly 5 strings, each a question.\n\n"
        "Content: {text}"
    )
)
combined_chain = LLMChain(llm=llm, prompt=combined_prompt)

def generate_content(text, style):
    response_text = combined_chain.run({"text": text, "style": style})
    parsed = parse_json_response(response_text)
    if parsed:
        notes = parsed.get("notes", "")
        questions = parsed.get("questions", [])
        return notes, questions
    else:
        st.error("Failed to parse JSON output from the AI. Please try again.")
        return None, None

# --- Chain for Objective Evaluation of Answers ---
evaluation_prompt = PromptTemplate(
    input_variables=["question", "student_answer"],
    template=(
        "You are an educational AI assistant. Evaluate the student's answer for the following question objectively.\n"
        "Question: {question}\n"
        "Student Answer: {student_answer}\n"
        "Also detect if the answer submitted by the student was AI generated; if yes, give a score of -1 with feedback as 'AI Generated Answer' if not then\n."
        "Provide a score between 0 and 5 and detailed feedback on how to improve the answer which is for the student.\n"
        "Return ONLY a valid JSON object with keys 'score' and 'feedback'. "
    )
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def evaluate_answer(question, student_answer):
    response_text = evaluation_chain.run({"question": question, "student_answer": student_answer})
    evaluation = parse_json_response(response_text)
    if evaluation:
        return evaluation
    else:
        st.error("Evaluation failed to produce valid JSON.")
        return None

# --- New Chain for Generating PlantUML Code ---
plantuml_prompt = PromptTemplate(
    input_variables=["notes"],
    template=(
        "You are an expert in UML diagramming. Given the following educational content, "
        "generate a diagram in standard PlantUML syntax. Do not include any external references "
        "or libraries (such as !include <C4/C4_Context>). Use only PlantUML's built-in elements. "
        "Return ONLY the PlantUML code with no additional explanation.\n\n"
        "Content: {notes}"
    )
)
plantuml_chain = LLMChain(llm=llm, prompt=plantuml_prompt)

def generate_plantuml_code(notes):
    return plantuml_chain.run({"notes": notes})

# --- Helper Function: Generate Diagram using PlantUML ---
def generate_uml(uml_code, output_filename="diagram.png"):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.uml', delete=False) as temp_file:
        uml_filepath = temp_file.name
        temp_file.write(uml_code)
    
    command = ['plantuml', '-tpng', uml_filepath]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        st.error("Error generating UML diagram: " + result.stderr.decode())
        os.remove(uml_filepath)
        return None
    
    generated_filepath = uml_filepath.replace('.uml', '.png')
    if os.path.exists(generated_filepath):
        os.rename(generated_filepath, output_filename)
        os.remove(uml_filepath)
        return output_filename
    else:
        st.error("Diagram file not found.")
        os.remove(uml_filepath)
        return None

# --- New Chain for Lesson Planning ---
lesson_planning_prompt = PromptTemplate(
    input_variables=["plan_type", "subject", "grade_level", "objectives", "num_days"],
    template=(
        "You are an AI that assists teachers in creating lesson materials.\n\n"
        "Plan Type: {plan_type}\n"
        "Subject: {subject}\n"
        "Grade Level: {grade_level}\n"
        "Objectives: {objectives}\n"
        "Number of Days: {num_days}\n\n"
        "Instructions:\n"
        "- If the plan type is 'Lesson Seed', provide a brief outline or idea that the teacher can expand.\n"
        "- If the plan type is 'Lesson Plan', provide a detailed lesson structure (objectives, activities, assessment).\n"
        "- If the plan type is 'Unit Plan', outline a multi-week or multi-topic approach, including subtopics and key activities.\n"
        "- If the plan type is 'Plan by Number of Days', break the plan into day-by-day sections.\n\n"
        "Return only the plan details with minimal additional text."
    )
)
lesson_planning_chain = LLMChain(llm=llm, prompt=lesson_planning_prompt)

def generate_lesson_plan(plan_type, subject, grade_level, objectives, num_days):
    response = lesson_planning_chain.run({
        "plan_type": plan_type,
        "subject": subject,
        "grade_level": grade_level,
        "objectives": objectives,
        "num_days": num_days
    })
    return response

# --- New Chain for Generating Flashcards ---
flashcard_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are an educational AI assistant. Based on the following content, generate a set of flashcards to help review key concepts. "
        "Return ONLY a valid JSON object with one key 'flashcards'. The value should be an array of objects, each with keys 'question' and 'answer'.\n\n"
        "Content: {text}"
    )
)
flashcard_chain = LLMChain(llm=llm, prompt=flashcard_prompt)

def generate_flashcards(text):
    response_text = flashcard_chain.run(text)
    parsed = parse_json_response(response_text)
    if parsed and "flashcards" in parsed:
        return parsed["flashcards"]
    else:
        st.error("Failed to generate flashcards.")
        return None

# --- Initialize session_state variables if not already set ---
if "notes" not in st.session_state:
    st.session_state["notes"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = {}
if "plantuml_code" not in st.session_state:
    st.session_state["plantuml_code"] = ""
if "diagram_path" not in st.session_state:
    st.session_state["diagram_path"] = ""
if "show_uml_code" not in st.session_state:
    st.session_state["show_uml_code"] = False
if "lesson_plan" not in st.session_state:
    st.session_state["lesson_plan"] = ""
if "flashcards" not in st.session_state:
    st.session_state["flashcards"] = []

# --- Navigation: Sidebar for Multi-Page Layout ---
page = st.sidebar.radio("Navigation", [
    "Generate Content",
    "Notes",
    "Flashcards",
    "Questionnaire",
    "Dashboard",
    "Visual Insights",
    "Lesson Planning"
])

if page == "Generate Content":
    st.title("📚 Generate Content")
    st.subheader("Upload a PDF or Notes File")
    
    # Radio selection for PDF type
    pdf_type = st.radio("Select PDF Type:", ["Small PDF", "Textbook"])
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    style_options = ["Story-telling", "Case Study", "Bullet Points", "Formal Summary", "As complete explanation"]
    selected_style = st.selectbox("Choose a style for personalized notes:", style_options)
    
    if uploaded_file:
        if pdf_type == "Textbook":
            # For textbooks, try to extract outlines and use selected section
            outlines = extract_outlines(uploaded_file)
            if outlines:
                sections = [(title, page_num) for title, page_num in outlines if page_num is not None]
                if sections:
                    section_titles = [f"{title} (Page {page_num+1})" for title, page_num in sections]
                    selected_section = st.selectbox("Select the section to generate notes from:", section_titles)
                    sel_index = section_titles.index(selected_section)
                    start_page = sections[sel_index][1]
                    # Extract text from the first two pages starting at the selected section
                    extracted_text = extract_text_from_pdf(uploaded_file, start_page=start_page, num_pages=2)
                else:
                    st.warning("No selectable sections found. Extracting entire text.")
                    extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                st.info("No outlines found. Extracting entire text from PDF.")
                extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            # For small PDFs, simply extract all text
            extracted_text = extract_text_from_pdf(uploaded_file)
            
        if extracted_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", extracted_text[:1000], height=200)
            if st.button("Generate Content"):
                with st.spinner("Generating personalized notes and questions..."):
                    notes, questions = generate_content(extracted_text, selected_style)
                if notes and questions:
                    st.session_state["notes"] = notes
                    st.session_state["questions"] = questions
                    # Reset flashcards if new content is generated
                    st.session_state["flashcards"] = []
                    st.success("Content generated successfully!")
                    
elif page == "Notes":
    st.title("📝 Notes")
    if st.session_state["notes"]:
        st.write(st.session_state["notes"])
    else:
        st.info("No notes available. Please generate content first.")

elif page == "Flashcards":
    st.title("🃏 Flashcards")
    if not st.session_state["notes"]:
        st.warning("No content generated. Please generate content first.")
    else:
        st.subheader("Flashcards Generated from Content")
        flashcards = generate_flashcards(st.session_state["notes"])
        if flashcards:
            st.session_state["flashcards"] = flashcards
            for idx, card in enumerate(flashcards, start=1):
                with st.expander(f"Flashcard {idx}: {card.get('question', 'No Question Provided')}"):
                    st.write("*Answer:*", card.get("answer", "No Answer Provided"))
        else:
            st.error("Flashcards could not be generated.")

elif page == "Questionnaire":
    st.title("📝 Questionnaire")
    if not st.session_state["questions"]:
        st.warning("No content generated. Please generate content first.")
    else:
        st.subheader("Answer the Following Questions")
        for idx, question in enumerate(st.session_state["questions"], start=1):
            st.markdown(f"**Question {idx}:** {question}")
            answer_key = f"q{idx}"
            student_answer = st.text_area(f"Your Answer for Question {idx}", key=answer_key)
            if st.button(f"Submit Answer for Question {idx}", key=f"btn_{idx}"):
                if not student_answer.strip():
                    st.error("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating answer..."):
                        evaluation = evaluate_answer(question, student_answer)
                    if evaluation:
                        st.success(f"Score: {evaluation.get('score')}")
                        st.info(f"Feedback: {evaluation.get('feedback')}")
                        st.session_state["evaluations"][answer_key] = {
                            "score": evaluation.get("score"),
                            "feedback": evaluation.get("feedback")
                        }
        if st.button("Show All Evaluations"):
            st.subheader("Evaluations Summary")
            if st.session_state["evaluations"]:
                for key, eval_data in st.session_state["evaluations"].items():
                    st.write(f"**{key}**: Score: {eval_data.get('score')}, Feedback: {eval_data.get('feedback')}")
            else:
                st.info("No evaluations available.")

elif page == "Dashboard":
    st.title("📊 Dashboard")
    if not st.session_state["evaluations"]:
        st.warning("No evaluations available. Please submit some answers first.")
    else:
        st.subheader("Evaluations Overview")
        evaluations = st.session_state["evaluations"]
        scores = []
        ai_generated_count = 0
        for key, eval_data in evaluations.items():
            score = eval_data.get("score")
            if score is not None:
                scores.append(score)
                if score == -1:
                    ai_generated_count += 1
        total_evaluations = len(scores)
        average_score = sum(scores) / total_evaluations if total_evaluations > 0 else 0
        
        st.markdown(f"**Total Evaluations:** {total_evaluations}")
        st.markdown(f"**Average Score:** {average_score:.2f}")
        st.markdown(f"**AI Generated Answers:** {ai_generated_count}")
        
        st.subheader("Evaluations Details")
        for key, eval_data in evaluations.items():
            st.write(f"**{key}**: Score: {eval_data.get('score')}, Feedback: {eval_data.get('feedback')}")
        
        st.subheader("Scores Bar Chart")
        df = pd.DataFrame({
            "Question": list(evaluations.keys()),
            "Score": [eval_data.get("score") for eval_data in evaluations.values()]
        }).set_index("Question")
        st.bar_chart(df)

elif page == "Visual Insights":
    st.title("🎨 Visual Insights")
    if not st.session_state["notes"]:
        st.warning("No content generated. Please generate content first to create a diagram.")
    else:
        if not st.session_state["plantuml_code"]:
            with st.spinner("Generating PlantUML code from content..."):
                plantuml_code = generate_plantuml_code(st.session_state["notes"])
                st.session_state["plantuml_code"] = plantuml_code
        else:
            plantuml_code = st.session_state["plantuml_code"]
        
        if not st.session_state["diagram_path"]:
            with st.spinner("Generating diagram using PlantUML..."):
                diagram_path = generate_uml(plantuml_code, output_filename="diagram.png")
                if diagram_path:
                    st.session_state["diagram_path"] = diagram_path
        
        if st.session_state["diagram_path"]:
            st.image(st.session_state["diagram_path"], caption="Generated Diagram", use_column_width=True)
            st.success("Diagram generated successfully.")
        
        if st.button("Show/Hide UML Code"):
            st.session_state["show_uml_code"] = not st.session_state["show_uml_code"]
        
        if st.session_state["show_uml_code"]:
            st.code(st.session_state["plantuml_code"], language="plantuml")

elif page == "Lesson Planning":
    st.title("📅 Lesson Planning")
    st.subheader("Create AI-generated lesson materials")

    plan_type = st.radio(
        "Select Plan Type",
        ["Lesson Seed", "Lesson Plan", "Unit Plan", "Plan by Number of Days"]
    )

    subject = st.text_input("Subject (e.g., Math, English)")
    grade_level = st.text_input("Grade Level (e.g., Grade 5)")
    objectives = st.text_area("Learning Objectives (optional)")

    if plan_type == "Plan by Number of Days":
        num_days = st.number_input("Number of Days", min_value=1, max_value=10, value=5)
    else:
        num_days = 1

    if st.button("Generate Lesson Plan"):
        with st.spinner("Generating lesson plan..."):
            lesson_text = generate_lesson_plan(
                plan_type,
                subject,
                grade_level,
                objectives,
                num_days
            )
        st.subheader("Your AI-Generated Plan")
        st.write(lesson_text)
        st.session_state["lesson_plan"] = lesson_text
