from pycuber import Cube, Centre, Corner, Edge, Square
import random
from visualization import *

ACTION = ["U", "U'",
          "L", "L'",
          "F", "F'",
          "R", "R'",
          "B", "B'",
          "D", "D'"]


def inv(a):
    return a - 1 if a % 2 else a + 1


def generate_face(face):
    map_f = [['' for _ in range(3)] for _ in range(3)]
    for i, c in enumerate(face):
        row = i // 3
        col = i % 3
        map_f[row][col] = c
    return map_f


def generate_map(cuber_dict):
    map = [[] for _ in range(5)]
    map[0] = generate_face(cuber_dict['U'])
    map[1] = generate_face(cuber_dict['L'])
    map[2] = generate_face(cuber_dict['F'])
    map[3] = generate_face(cuber_dict['R'])
    map[4] = generate_face(cuber_dict['B'])
    if 'D' in cuber_dict.keys():
        map.append(generate_face(cuber_dict['D']))
    return map


class CuberEnv(Cube):
    '''
    Rubik's Cuber:
         U          Upward
        LFRB  Left  Front  Right  Behind
         D          Downward

    Goal:
        Given a picture of all side's blocks as initial state.
        Solve the cuber, with blocks of the same color in each side.
        If step >= max_step without solved, game failed.

    State(6 x 3 x 3):
        3x3(solved):
           000
           000
           000
        111222333444
        111222333444
        111222333444
           555
           555
           555

    Action(12):
        F: Front clockwise
        F': Front anticlockwise
        L: Left clockwise
        L': Left anticlockwise
        R: Right clockwise
        R': Right anticlockwise
        U: Upside clockwise
        U': Upside anticlockwise
        D: Downside clockwise
        D': Downside anticlockwise
        B: Behind clockwise
        B': Behind anticlockwise

    Reward:
        For each step:
            reward = -1
        If visited state:
            reward = -2
    '''

    def __init__(self, map=None, rand=0, max_step=200):
        self.nA = len(ACTION)
        self.max_step = max_step
        self.init_actions = []
        self.action_his = []
        self.state_his = []
        if map:
            if isinstance(map, dict):
                map = generate_map(map)
            assert len(map) in [5, 6]

            def m(pos):
                i1 = (pos % 1000) // 100
                i2 = (pos % 100) // 10
                i3 = pos % 10
                return map[i1][i2][i3]

            if len(map) == 5:
                D_face = [['' for _ in range(3)] for _ in range(3)]
                corners = ['ygr', 'ygo', 'ybo', 'ybr',
                           'wgr', 'wgo', 'wbo', 'wbr']
                edges = ['yr', 'yg', 'yo', 'yb',
                         'rg', 'go', 'ob', 'br',
                         'wr', 'wg', 'wo', 'wb']
                centers = ['yw', 'gb', 'ro']
                c11 = m(11)
                for ct in centers:
                    if c11 in ct:
                        D_face[1][1] = ct[1] if ct[0] == c11 else ct[0]
                        break
                existed_corners = [m(20) + m(102) + m(200),
                                   m(22) + m(202) + m(300),
                                   m(0) + m(100) + m(402),
                                   m(2) + m(302) + m(400)]
                for cn in existed_corners:
                    cn_ = 'y' if 'y' in cn else 'w'
                    cn_ += 'g' if 'g' in cn else 'b'
                    cn_ += 'r' if 'r' in cn else 'o'
                    corners.remove(cn_)

                def find_corner(c):
                    for cn in corners:
                        if c[0] in cn and c[1] in cn:
                            for c_ in cn:
                                if c_ not in c:
                                    return c_

                D_face[0][0] = find_corner(m(122) + m(220))
                D_face[2][0] = find_corner(m(120) + m(422))
                D_face[2][2] = find_corner(m(322) + m(420))
                D_face[0][2] = find_corner(m(222) + m(320))

                existed_edges = [m(10) + m(101),
                                 m(21) + m(201),
                                 m(12) + m(302),
                                 m(1) + m(401),
                                 m(210) + m(112),
                                 m(212) + m(310),
                                 m(312) + m(410),
                                 m(412) + m(110)]

                for e in existed_edges:
                    e_ = e[1] + e[0]
                    if e in edges:
                        edges.remove(e)
                    elif e_ in edges:
                        edges.remove(e_)

                def find_edge(c):
                    for e in edges:
                        if c in e:
                            return e[1] if e[0] == c else e[0]

                D_face[0][1] = find_edge(m(221))
                D_face[1][2] = find_edge(m(321))
                D_face[2][1] = find_edge(m(421))
                D_face[1][0] = find_edge(m(121))

                map.append(D_face)

            self.map = map
            cubies = set()
            colours = {"r": "red", "y": "yellow", "g": "green", "w": "white", "o": "orange", "b": "blue"}

            def add(loc, pos):
                if len(loc) == 3:
                    cubies.add(Corner(**{loc[i]: Square(colours[m(pos[i])]) for i in range(3)}))
                elif len(loc) == 2:
                    cubies.add(Edge(**{loc[i]: Square(colours[m(pos[i])]) for i in range(2)}))
                else:
                    cubies.add(Centre(**{loc[0]: Square(colours[m(pos[0])])}))

            add('LDB', [120, 520, 422])
            add('LDF', [122, 500, 220])
            add('LUB', [100, 0, 402])
            add('LUF', [102, 20, 200])
            add('RDB', [322, 522, 420])
            add('RDF', [320, 502, 222])
            add('RUB', [302, 2, 400])
            add('RUF', [300, 22, 202])
            add('LB', [110, 412])
            add('LF', [112, 210])
            add('LU', [101, 10])
            add('LD', [121, 510])
            add('DB', [521, 421])
            add('DF', [501, 221])
            add('UB', [1, 401])
            add('UF', [21, 201])
            add('RB', [312, 410])
            add('RF', [310, 212])
            add('RU', [301, 12])
            add('RD', [321, 512])
            add('L', [111])
            add('R', [311])
            add('U', [11])
            add('D', [511])
            add('F', [211])
            add('B', [411])
            super().__init__(cubies=cubies)
        else:
            super().__init__()
            if rand > 0:
                for _ in range(rand):
                    self.step(self.sample())
                if self.solved():
                    self.step(self.sample())
            self.init_actions = self.action_his.copy()
            self.action_his = []
            self.state_his = []
            map = self.map.copy()
        self.state_his.append(self.map_str())
        self.init_map = map

    def reset(self):
        self.action_his = []
        self.state_his = []
        map = self.init_map

        def m(pos):
            i1 = (pos % 1000) // 100
            i2 = (pos % 100) // 10
            i3 = pos % 10
            return map[i1][i2][i3]

        self.map = map
        cubies = set()
        colours = {"r": "red", "y": "yellow", "g": "green", "w": "white", "o": "orange", "b": "blue"}

        def add(loc, pos):
            if len(loc) == 3:
                cubies.add(Corner(**{loc[i]: Square(colours[m(pos[i])]) for i in range(3)}))
            elif len(loc) == 2:
                cubies.add(Edge(**{loc[i]: Square(colours[m(pos[i])]) for i in range(2)}))
            else:
                cubies.add(Centre(**{loc[0]: Square(colours[m(pos[0])])}))

        add('LDB', [120, 520, 422])
        add('LDF', [122, 500, 220])
        add('LUB', [100, 0, 402])
        add('LUF', [102, 20, 200])
        add('RDB', [322, 522, 420])
        add('RDF', [320, 502, 222])
        add('RUB', [302, 2, 400])
        add('RUF', [300, 22, 202])
        add('LB', [110, 412])
        add('LF', [112, 210])
        add('LU', [101, 10])
        add('LD', [121, 510])
        add('DB', [521, 421])
        add('DF', [501, 221])
        add('UB', [1, 401])
        add('UF', [21, 201])
        add('RB', [312, 410])
        add('RF', [310, 212])
        add('RU', [301, 12])
        add('RD', [321, 512])
        add('L', [111])
        add('R', [311])
        add('U', [11])
        add('D', [511])
        add('F', [211])
        add('B', [411])
        super().__init__(cubies=cubies)
        self.state_his.append(self.map_str())

    def map_str(self):
        map_s = ''
        for face in self.map:
            for row in face:
                for c in row:
                    map_s += c
        return map_s

    def step(self, action):
        if isinstance(action, int):
            assert 0 <= action < self.nA
            a = ACTION[action]
        else:
            assert action in ACTION
            a = action
        self.perform_step(a)
        self.map = self.set_map()
        state_str = self.map_str()
        solved = self.solved()
        done = solved
        self.action_his.append(a)
        reward = -2.0 if state_str in self.state_his else -1.0
        self.state_his.append(state_str)
        if len(self.state_his) >= self.max_step:
            done = True
        return self.map, reward, done, solved

    def solved(self):
        for face in self.map[:4]:
            c = face[0][0]
            for c_ in (np.array(face, dtype=str)).ravel():
                if c != c_:
                    return False
        return True

    def sample(self):
        return int(random.random() * self.nA)

    def set_map(self):
        map_str = self.__str__()
        map = [[['' for _ in range(3)] for _ in range(3)] for _ in range(6)]
        face, row, col = 0, 0, 0
        for c in map_str:
            if c in 'rygwob':
                map[face][row][col] = c
                col += 1
                if col == 3:
                    col = 0
                    if 0 < face < 5:
                        face += 1
                        if face == 5:
                            if row < 2:
                                face = 1
                                row += 1
                            else:
                                row = 0
                    else:
                        row += 1
                        if row == 3:
                            row = 0
                            face += 1
        return map

    def render(self, plot=False, save=None):
        # if plot:
        #     print(self.__repr__())
        return make_img(self.map, save_p=save, plot=plot)


if __name__ == '__main__':
    # c = CuberEnv(rand=20)
    map = {
        'U': 'yyyyyyyyy',
        'L': 'rrrrrrbbb',
        'F': 'ggggggrrr',
        'R': 'ooooooggg',
        'B': 'bbbbbbooo'
    }
    c = CuberEnv(map=map)
    solved = c.solved()
    model = './model/best_10_003.dat'
    c.render(plot=True)
    print(c.solved())
    c.step('D\'')
    c.render(plot=True)
    # print(c.solved())

    # for _ in range(20):
    #     c.step(c.sample())
    #
    # c.render(plot=True)
    # print(c.map)
    # print(c.action_his)
    pass
