class GridWorld687:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.actions = ['AU', 'AD', 'AL', 'AR']
        self.actionsDir = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }
        self.forbidden_furniture = [(2, 2), (3, 2)]
        self.monsters = [(0, 2)]
        self.food = (4, 4)
    
    def valid(self, state):
        r, c = state
        return 0 <= r < 5 and 0 <= c < 5 and state not in self.forbidden_furniture
    
    def reward(self, nextS):
        if nextS == self.food:
            return 10.0
        if nextS in self.monsters:
            return -10.0
        return 0.0
    
    def move(self, state, delta):
        new_r, new_c = state[0] + delta[0], state[1] + delta[1]
        if self.valid((new_r, new_c)):
            return (new_r, new_c)
        else:
            return state
    
    def p(self, state, action):
        intendedMove = self.actionsDir[action]
        if action == 'AU':
            left_move = (0, -1)
            right_move = (0, 1)
        elif action == 'AD':
            left_move = (0, 1)
            right_move = (0, -1)
        elif action == 'AL':
            left_move = (1, 0)
            right_move = (-1, 0)
        elif action == 'AR':
            left_move = (-1, 0)
            right_move = (1, 0)
        return [
            self.move(state, intendedMove), 
            self.move(state, left_move),
            self.move(state, right_move),
            state,
        ], [
            0.8,
            0.05,
            0.05,
            0.1
        ]