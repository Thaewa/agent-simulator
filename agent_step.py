class Agent(ABC):
    ...
    @abstractmethod
    def step(self, t: int) -> None:
        """
        Perform one simulation step.
        To be implemented by subclasses.
        """
        pass


class Wasp(Agent):
    ...
    def step(self, t: int) -> None:
        """Wasp step: decideAction, then askForFood."""
        self.decideAction()
        self.askForFood()


class Larvae(Agent):
    ...
    def step(self, t: int) -> None:
        """Larvae step: only askForFood (no decideAction)."""
        self.askForFood()
