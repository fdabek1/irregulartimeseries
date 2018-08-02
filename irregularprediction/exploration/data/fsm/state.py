import random


class State:
    def __init__(self, letter, timestamp):
        self.letter = letter
        self.timestamp = timestamp
        self.children = []

    def add_child(self, state):
        self.children.append(state)

    def has_children(self):
        return len(self.children) > 0

    def pick_child(self):
        return random.choice(self.children)
