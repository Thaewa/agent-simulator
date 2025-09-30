# agent_decide.py
# Contains the logic for how an agent decides what to do (rule-based)

from agents import Wasp, Larvae, WaspRole

def decide_action(agent, environment):
    """
    Rule-based decision logic for each agent.
    environment: dict that contains references to other agents.
    """
    # Wasp
    if isinstance(agent, Wasp):

        # Case 1: Forager
        if agent.role == WaspRole.FORAGER:
            if agent.food == 0:
                return {"action": "forage"}
            else:
                feeders = [a for a in environment["agents"] if isinstance(a, Wasp) and a.role == WaspRole.FEEDER]
                if feeders:
                    target = feeders[0]   # simple: pick first feeder
                    if agent.getPosition() == target.getPosition():
                        return {"action": "transfer", "target": target}
                    else:
                        return {"action": "move_to", "target": target}
                return {"action": "idle"}

        # Case 2: Feeder
        elif agent.role == WaspRole.FEEDER:
            if agent.food == 0:
                return {"action": "idle"}
            else:
                larvae = [a for a in environment["agents"] if isinstance(a, Larvae)]
                if larvae:
                    target = max(larvae, key=lambda l: l.hunger)
                    if agent.getPosition() == target.getPosition():
                        return {"action": "feed", "target": target}
                    else:
                        return {"action": "move_to", "target": target}
                return {"action": "idle"}

    # Default: if no condition matched
    return {"action": "idle"}
