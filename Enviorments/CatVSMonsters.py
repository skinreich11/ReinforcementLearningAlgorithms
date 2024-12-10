class CatVSMonsters:
    def __init__(self, gamma=0.925, catnip=False, catnipTerminal=False, catnipReward=5.0):
        self.gamma = gamma
        self.catnip = catnip
        self.catnipTerminal = catnipTerminal
        self.catnipReward = catnipReward
        self.actions = ['AU', 'AD', 'AL', 'AR']
        self.actionsDir = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }
        self.forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
        self.monsters = [(0, 3), (4, 1)]
        self.food = (4, 4)

    def valid(self, state):
        r, c = state
        return 0 <= r < 5 and 0 <= c < 5 and state not in self.forbidden_furniture

    def reward(self, nextS):
        if nextS == self.food:
            return 10.0
        if nextS in self.monsters:
            return -8.0
        if self.catnip and nextS == (0, 1):
            return self.catnipReward
        return -0.05

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
            0.7,
            0.12,
            0.12,
            0.06
        ]

    def results(self, V, pi):
        print("Final Value Function:")
        print(V)
        print("\nFinal Policy:")
        for i in range(5):
            arrowRow = ""
            for j in range(5):
                state = (i, j)
                match pi[state]:
                    case "AU":
                        arrowRow += u'\u2191'
                        arrowRow += " "
                    case "AR":
                        arrowRow += u'\u2192'
                        arrowRow += " "
                    case "AD":
                        arrowRow += u'\u2193'
                        arrowRow += " "
                    case "AL":
                        arrowRow += u'\u2190'
                        arrowRow += " "
                    case "G ":
                        arrowRow += "G "
                    case _:
                        arrowRow += "  "
            print(arrowRow)