import json
import os
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

AGENTS_FILE = Path(__file__).with_name("agents.json")
DEFAULT_AGENTS = [
    {"role": "Research Analyst", "goal": "Find reliable information and turn it into clear, useful insights", "backstory": "You are a careful researcher who compares sources, spots patterns, and explains complex topics plainly."},
    {"role": "Content Strategist", "goal": "Shape information into focused content for a specific audience", "backstory": "You are an experienced editor who understands audience needs, structure, tone, and persuasive communication."},
    {"role": "Data Analyst", "goal": "Interpret data, identify meaningful trends, and communicate practical conclusions", "backstory": "You are a methodical analyst who checks assumptions, looks for outliers, and presents evidence-driven recommendations."},
    {"role": "Quality Reviewer", "goal": "Check work for accuracy, completeness, clarity, and actionable improvements", "backstory": "You are a constructive reviewer who catches gaps and inconsistencies while keeping the final result easy to use."},
]


def load_saved_agents():
    if not AGENTS_FILE.exists():
        return [dict(agent) for agent in DEFAULT_AGENTS]
    try:
        with AGENTS_FILE.open("r", encoding="utf-8") as file:
            saved_agents = json.load(file)
        return saved_agents if isinstance(saved_agents, list) else []
    except (OSError, json.JSONDecodeError):
        return [dict(agent) for agent in DEFAULT_AGENTS]


def save_agents(agents):
    with AGENTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(agents, file, indent=2)


def extract_document_text(uploaded_file):
    """Extract text from the supported document formats."""
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith((".txt", ".md", ".csv", ".json")):
        return file_bytes.decode("utf-8", errors="replace")

    if file_name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if file_name.endswith(".docx"):
        from docx import Document

        document = Document(BytesIO(file_bytes))
        return "\n\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    raise ValueError("Unsupported file type. Use TXT, MD, CSV, JSON, PDF, or DOCX.")


def create_docx_download(text):
    from docx import Document

    document = Document()
    for paragraph in text.split("\n"):
        document.add_paragraph(paragraph)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def parse_json_from_text(raw_text):
    if raw_text is None:
        raise ValueError("No response from the model.")

    if isinstance(raw_text, dict):
        return raw_text

    text = str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON.")
        data = json.loads(match.group(0))

    return data


def normalize_workflow_payload(payload):
    if isinstance(payload, dict):
        agents = payload.get("agents") or payload.get("roles") or []
        tasks = payload.get("tasks") or payload.get("workflow") or []
    elif isinstance(payload, list):
        agents = []
        tasks = payload
    else:
        raise ValueError("Workflow payload must be a JSON object with agents and tasks.")

    cleaned_agents = []
    for item in agents:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or item.get("name") or "").strip()
        goal = (item.get("goal") or item.get("objective") or "").strip()
        backstory = (item.get("backstory") or item.get("style") or "").strip()
        if role and goal and backstory:
            cleaned_agents.append({"role": role, "goal": goal, "backstory": backstory})

    cleaned_tasks = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        description = (item.get("description") or item.get("task") or "").strip()
        expected_output = (item.get("expected_output") or item.get("deliverable") or "").strip()
        agent_role = (item.get("agent_role") or item.get("assigned_role") or item.get("agent") or "").strip()

        if description and expected_output:
            cleaned_tasks.append({
                "description": description,
                "expected_output": expected_output,
                "agent_role": agent_role or (cleaned_agents[0]["role"] if cleaned_agents else "Research Analyst"),
            })

    if not cleaned_agents and cleaned_tasks:
        unique_roles = []
        seen = set()
        for task in cleaned_tasks:
            role = task["agent_role"]
            if role and role not in seen:
                unique_roles.append({
                    "role": role,
                    "goal": f"Complete the task: {task['description'][:80]}",
                    "backstory": "You are a specialist agent created from the workflow description."
                })
                seen.add(role)
        cleaned_agents = unique_roles

    if cleaned_agents and not cleaned_tasks:
        cleaned_tasks = [{
            "description": f"Execute the workflow for: {cleaned_agents[0]['goal']}",
            "expected_output": "A concise, actionable outcome that fulfills the requested workflow.",
            "agent_role": cleaned_agents[0]["role"]
        }]

    return cleaned_agents, cleaned_tasks


def generate_workflow_from_brief(workflow_brief, llm):
    if not llm:
        raise ValueError("Add the provider API key before generating a workflow.")

    prompt = f"""
You are a workflow architect. Create a realistic multi-agent work plan from this brief.

USER BRIEF:
{workflow_brief}

Important: The workflow must be suitable for a resume and LinkedIn career growth use case.
Focus on extracting facts from the resume, identifying the current profile, recommending the next role, and producing updated profile/resume language.

Return a valid JSON object with exactly these keys:
{{
  "agents": [
    {{"role": "Resume Analyst", "goal": "Extract years of experience, current location, and current role from a resume.", "backstory": "You are detail-oriented and careful with factual extraction from career documents."}},
    {{"role": "Career Strategist", "goal": "Identify the most relevant next job based on present role and location.", "backstory": "You reason about role transitions, market fit, and realistic career progression."}},
    {{"role": "LinkedIn Profile Writer", "goal": "Draft profile content for the next role and location.", "backstory": "You write concise, professional, high-impact profile language tailored to a target job."}},
    {{"role": "Resume Writer", "goal": "Draft updated resume content that matches the target role.", "backstory": "You rewrite experience and achievements into recruiter-friendly, evidence-based resume language."}}
  ],
  "tasks": [
    {{"description": "Read the attached resume and identify years of experience, present location, and current role.", "expected_output": "A clear summary of the candidate's current experience, location, and role.", "agent_role": "Resume Analyst"}},
    {{"description": "Determine the next related job opportunity from LinkedIn based on the current role and current location.", "expected_output": "A realistic target job title and a short explanation of why it matches the candidate profile.", "agent_role": "Career Strategist"}},
    {{"description": "Write the LinkedIn profile content needed to position the candidate for the target next role.", "expected_output": "Updated headline, summary, and profile positioning content for LinkedIn.", "agent_role": "LinkedIn Profile Writer"}},
    {{"description": "Write the resume experience and summary content to update the resume for the next role.", "expected_output": "Updated resume summary and achievement-oriented content tailored to the target role.", "agent_role": "Resume Writer"}}
  ]
}}

Rules:
- Use 2 to 5 agents.
- Use 2 to 5 tasks.
- Match each task to an existing agent role.
- Keep the response valid JSON only; no markdown fences, no narration, no comments.
- Make the workflow realistic and actionable for a resume and career progression scenario.
"""
    response = llm.call(prompt)
    payload = parse_json_from_text(response)
    return normalize_workflow_payload(payload)


st.set_page_config(page_title="Auto Workflow Builder", layout="wide")
st.title("Workflow Studio Auto Builder")
st.subheader("Describe the workflow once; let the system propose the roles and tasks")

with st.sidebar:
    st.header("Model settings")
    api_key_input = st.text_input(
        "Provider API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Your provider API key is used only when a workflow runs or generates tasks."
    )

    gemini_model = st.selectbox(
        "Select model",
        options=["gemini/gemini-3.5-flash", "gemini/gemini-3.5-pro", "gemini/gemini-3.5-flash-lite"],
        index=0,
    )

    process_type = st.radio(
        "Workflow mode",
        options=["Sequential", "Hierarchical"],
        index=0,
        help="Choose whether tasks run in order or are coordinated by a manager."
    )

if api_key_input:
    gemini_llm = LLM(model=gemini_model, api_key=api_key_input)
else:
    gemini_llm = None
    st.warning("Add a provider API key in the sidebar to generate or run a workflow.")

if "saved_agents" not in st.session_state:
    st.session_state.saved_agents = load_saved_agents()
if "generated_agents" not in st.session_state:
    st.session_state.generated_agents = []
if "generated_tasks" not in st.session_state:
    st.session_state.generated_tasks = []
if "workflow_accepted" not in st.session_state:
    st.session_state.workflow_accepted = False

st.divider()
st.header("1. Describe your workflow")
workflow_brief = st.text_area(
    "What should this workflow do?",
    height=150,
    placeholder="Example: Build a customer research workflow for a new fintech app. The flow should compare competitors, summarize customer pain points, and recommend feature priorities.",
)

if st.button("Generate workflow", type="primary"):
    if not workflow_brief.strip():
        st.error("Please describe the workflow you want to build.")
    elif gemini_llm is None:
        st.error("Add a provider API key before generating the workflow.")
    else:
        try:
            with st.spinner("Generating agents and tasks from your brief..."):
                generated_agents, generated_tasks = generate_workflow_from_brief(workflow_brief, gemini_llm)
                st.session_state.generated_agents = generated_agents
                st.session_state.generated_tasks = generated_tasks
                st.session_state.workflow_accepted = False
            st.success("A draft workflow was generated.")
        except Exception as exc:
            st.error(f"Could not generate the workflow: {exc}")

if st.session_state.generated_agents or st.session_state.generated_tasks:
    st.divider()
    st.header("2. Review generated plan")

    st.subheader("Suggested agents")
    for agent in st.session_state.generated_agents:
        st.write(f"**{agent['role']}**")
        st.caption(agent["goal"])
        st.caption(agent["backstory"])

    st.subheader("Suggested tasks")
    for index, task in enumerate(st.session_state.generated_tasks, start=1):
        st.write(f"**Task {index}:** {task['agent_role']}")
        st.write(task["description"])
        st.caption(f"Expected output: {task['expected_output']}")

    if st.button("Use this generated workflow"):
        st.session_state.saved_agents = st.session_state.saved_agents + [
            agent for agent in st.session_state.generated_agents
            if agent not in st.session_state.saved_agents
        ]
        save_agents(st.session_state.saved_agents)
        st.session_state.workflow_accepted = True
        st.success("The generated workflow is ready to run.")

st.divider()
st.header("3. Run workflow")

uploaded_document = st.file_uploader(
    "Attach a source document for the workflow",
    type=["txt", "md", "csv", "json", "pdf", "docx"],
    help="Optional context to include in every task.",
)

document_text = ""
if uploaded_document:
    try:
        document_text = extract_document_text(uploaded_document)
        st.success(f"Attached {uploaded_document.name} ({len(document_text):,} characters).")
        with st.expander("Preview attached document"):
            st.text(document_text[:5000] or "No readable text was found.")
    except Exception as exc:
        st.error(f"Could not read {uploaded_document.name}: {exc}")

active_agents = st.session_state.generated_agents if st.session_state.workflow_accepted else []
active_tasks = st.session_state.generated_tasks if st.session_state.workflow_accepted else []

if not active_agents or not active_tasks:
    st.info("Generate a workflow first and accept it to start the run.")
else:
    if st.button("Run workflow", type="primary"):
        if not gemini_llm:
            st.error("Add a provider API key before running.")
        else:
            try:
                with st.spinner("Running your generated workflow..."):
                    created_agents = {}
                    crewai_agents = []
                    for ag in active_agents:
                        agent_obj = Agent(
                            role=ag["role"],
                            goal=ag["goal"],
                            backstory=ag["backstory"],
                            llm=gemini_llm,
                            verbose=False,
                        )
                        created_agents[ag["role"]] = agent_obj
                        crewai_agents.append(agent_obj)

                    crewai_tasks = []
                    for task in active_tasks:
                        assigned_agent = created_agents.get(task["agent_role"])
                        if assigned_agent is None:
                            assigned_agent = next(iter(created_agents.values()))

                        task_description = task["description"]
                        if document_text:
                            task_description = (
                                f"{task_description}\n\n"
                                "SOURCE DOCUMENT (Use this as reference material):\n"
                                f"{document_text}"
                            )

                        task_obj = Task(
                            description=task_description,
                            expected_output=task["expected_output"],
                            agent=assigned_agent,
                        )
                        crewai_tasks.append(task_obj)

                    selected_process = Process.sequential if process_type == "Sequential" else Process.hierarchical
                    crew = Crew(
                        agents=crewai_agents,
                        tasks=crewai_tasks,
                        process=selected_process,
                        manager_llm=gemini_llm if selected_process == Process.hierarchical else None,
                        verbose=False,
                    )
                    result = crew.kickoff()

                st.success("Workflow completed.")
                st.markdown("### Final result")
                st.markdown(result.raw)
                st.download_button(
                    "Download result as Markdown",
                    data=result.raw,
                    file_name="workflow_result.md",
                    mime="text/markdown",
                )
                try:
                    st.download_button(
                        "Download result as Word document",
                        data=create_docx_download(result.raw),
                        file_name="workflow_result.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                except ImportError:
                    st.caption("Install python-docx to enable Word document downloads.")
            except Exception:
                st.error("The workflow could not be completed. Check your settings and try again.")
