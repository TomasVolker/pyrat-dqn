from pyrat.components import State


class Agent:

    def prepare(self, state: State) -> None:
        pass

    def act(self, state: State) -> str:
        raise NotImplementedError()





