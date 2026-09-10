import sys

import streamlit as st

st.set_page_config(page_title="Dynamic CrewAI Builder", layout="wide")
st.title("🤖 No-Code Dynamic CrewAI Workflow")

if sys.version_info >= (3, 13):
    st.warning(
        "CrewAI is currently unstable on Python 3.13. Please switch to Python 3.11.x for this app to run reliably."
    )
    st.stop()

# -------------------------------------------------------------
# STEP 1: Dynamic Agent Configuration via UI
# -------------------------------------------------------------
st.sidebar.header("1. Define Agents")
num_agents = st.sidebar.number_input("How many agents?", min_value=1, max_value=5, value=2)

agents_config = []
for i in range(num_agents):
    st.sidebar.subheader(f"Agent {i+1}")
    role = st.sidebar.text_input(f"Role #{i+1}", value=f"Researcher" if i==0 else "Writer")
    goal = st.sidebar.text_area(f"Goal #{i+1}", value="Gather key insights" if i==0 else "Draft summary report")
    backstory = st.sidebar.text_area(f"Backstory #{i+1}", value="Expert researcher" if i==0 else "Professional technical writer")
    
    agents_config.append({
        "role": role,
        "goal": goal,
        "backstory": backstory
    })

# -------------------------------------------------------------
# STEP 2: Dynamic Task Configuration via UI
# -------------------------------------------------------------
st.sidebar.header("2. Define Tasks")
num_tasks = st.sidebar.number_input("How many tasks?", min_value=1, max_value=5, value=2)

tasks_config = []
agent_roles = [a["role"] for a in agents_config]

for j in range(num_tasks):
    st.sidebar.subheader(f"Task {j+1}")
    desc = st.sidebar.text_area(f"Task Description #{j+1}", value="Analyze market trends for {topic}")
    expected_output = st.sidebar.text_input(f"Expected Output #{j+1}", value="Detailed bullet points of market trends")
    assigned_role = st.sidebar.selectbox(f"Assign to Agent #{j+1}", options=agent_roles)
    
    tasks_config.append({
        "description": desc,
        "expected_output": expected_output,
        "assigned_role": assigned_role
    })

# -------------------------------------------------------------
# STEP 3: Task Inputs & Execution Trigger
# -------------------------------------------------------------
st.header("3. Execution Settings")
topic_input = st.text_input("Enter Topic Input ({topic} placeholder variable):", value="Artificial Intelligence in Healthcare")

if st.button("🚀 Run Crew Workflow"):
    try:
        from crewai import Agent, Task, Crew, Process
    except Exception as exc:
        st.error("CrewAI could not be imported. Please use Python 3.11.x and reinstall the dependencies.")
        st.exception(exc)
        st.stop()

    st.info("Instantiating Agents and Tasks dynamically...")

    # A. Dynamically Instantiate Agents
    created_agents = {}
    for cfg in agents_config:
        agent_obj = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            verbose=True
        )
        created_agents[cfg["role"]] = agent_obj

    # B. Dynamically Instantiate Tasks and Link to assigned Agents
    created_tasks = []
    for cfg in tasks_config:
        assigned_agent = created_agents[cfg["assigned_role"]]
        task_obj = Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=assigned_agent
        )
        created_tasks.append(task_obj)

    # C. Assemble Crew and Run
    dynamic_crew = Crew(
        agents=list(created_agents.values()),
        tasks=created_tasks,
        process=Process.sequential,
        verbose=True
    )

    with st.spinner("Crew is executing tasks..."):
        # Pass inputs dynamically to kickoff
        result = dynamic_crew.kickoff(inputs={"topic": topic_input})

    st.success("Execution Complete!")
    st.subheader("Final Output:")
    st.markdown(result.raw)