from state import State


class FSM:
    def __init__(self):
        self.begin = State('0', 0)

    def add_state(self, letter, timestamp, parent=None):
        state = State(letter, timestamp)
        if parent is None:
            self.begin.add_child(state)
        else:
            parent.add_child(state)

        return state

    def generate_seq(self):
        current = self.begin
        seq = [current]
        while current.has_children():
            current = current.pick_child()
            seq.append(current)

        return seq
