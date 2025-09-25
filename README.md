# Agent Simulation Project

This project is a baseline scaffold for simulating agents in an environment.  
The system is designed from UML diagrams and split into modular Python files.

---

## 📂 Project Structure
```
project/
│── agents.py # Defines Agent class
│── agent_step.py # Logic for agent step (state update per tick)
│── agent_decide.py # Logic for agent decision-making
│── simulator.py # Simulator class: manages agents and environment
│── simulation_loop.py # Runs the main simulation loop
│── main.py # Entry point for running the simulation
```
---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd project
   ```

2.	Run the simulation
	```bash
	python main.py
	```

## ⚙️ Requirements
- Python 3.9+
- (Optional) Virtual environment for development:
    ```bash
	python -m venv venv
	source venv/bin/activate   # Mac/Linux
	venv\Scripts\activate      # Windows
	```
	
At this stage, no external dependencies are required.
Future logic can be added into agent_decide.py and agent_step.py.

## 🛠 Contribution Guide
Create a new branch for each feature (e.g., feature/agent-decision)
Add or modify logic inside the appropriate file
Submit a pull request for code review before merging to main

## 📌 Next Steps
Implement decision-making in agent_decide.py
Implement step logic in agent_step.py
Expand Simulator to support more complex environments