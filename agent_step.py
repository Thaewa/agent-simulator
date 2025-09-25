# agent_step.py
# Executes actions decided by agent_decide.py

from agents import Wasp, Larvae, WaspRole

def execute_action(agent, action: dict, simulator=None) -> None:
    """
    Execute the action returned from agent_decide.decide_action().
    simulator: pass simulator reference for logging movements
    """

    act = action.get("action", "idle")

    # Larvae behavior
    if isinstance(agent, Larvae):
        if act == "ask_food":
            agent.askForFood()
        else:
            agent.storedEvents.append(f"{agent.id} stayed idle")

    # Wasp behavior
    elif isinstance(agent, Wasp):
        if act == "forage":
            agent.forage()
        elif act == "feed":
            target = action.get("target")
            if target:
                agent.feed(target)
        elif act == "transfer":
            target = action.get("target")
            if target:
                agent.transfer_food(target)
        elif act == "move_to":
            target = action.get("target")
            if target:
                agent.move_towards(target, simulator)
        elif act == "idle":
            agent.storedEvents.append(f"{agent.id} idled")
        else:
            agent.storedEvents.append(f"{agent.id} unknown action: {act}")
