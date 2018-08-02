from fsm import FSM

fsm = FSM()
# Top Path
top = fsm.add_state('1', 2, None)
top = fsm.add_state('2', 4, top)
top = fsm.add_state('3', 6, top)
top = fsm.add_state('4', 8, top)

# Bottom Path
bottom = fsm.add_state('1', 5, None)
bottom = fsm.add_state('3', 8, bottom)
bottom = fsm.add_state('2', 11, bottom)
bottom = fsm.add_state('4', 14, bottom)

file_index = 1
num_seqs = {
    'train': 1000,
    'test': 200,
}

for data_type in ['train', 'test']:
    seqs = [fsm.generate_seq() for _ in range(num_seqs[data_type])]
    with open('event-' + str(file_index) + '-' + data_type + '.txt', 'w') as w_events:
        for seq in seqs:
            w_events.write(' '.join([state.letter for state in seq]) + '\n')

    with open('time-' + str(file_index) + '-' + data_type + '.txt', 'w') as w_times:
        for seq in seqs:
            w_times.write(' '.join([str(state.timestamp) for state in seq]) + '\n')
