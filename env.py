from enum import Enum


class Pointer(Enum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3


def transform(state: Pointer):
    next_state_value = (state.value + 1) % 4
    return Pointer(next_state_value)


class Env:

    def __init__(self, height, width, max_step=10):
        self._height = height
        self._width = width
        self._max_step = 10
        self._step = 0

        self._grids = [[Pointer.Up for _ in range(width)] for _ in range(height)]
    
    def reset(self):
        self._step = 0
        self._grids = [[Pointer.Up for _ in range(self._width)] for _ in range(self._height)]
        return self._digitizing_grids()
    
    def _get_pointed_grid(self, x, y, pointer: Pointer):
        if pointer == Pointer.Up:
            x = x - 1
        elif pointer == Pointer.Right:
            y = y + 1
        elif pointer == Pointer.Down:
            x = x + 1
        else:
            y = y - 1
        return x, y
    
    def _is_in_grids(self, x, y):
        return 0 <= x < self._height and 0 <= y < self._width
    
    def render(self):
        print()
        for i in range(self._height):
            for j in range(self._width):
                print(self._grids[i][j].value, end=" ")
            print()
    
    def _digitizing_grids(self):
        grids = [[self._grids[i][j].value for j in range(self._width)] for i in range(self._height)]
        return grids
    
    def step(self, x, y):
        assert 0 <= x < self._height
        assert 0 <= y < self._width
        state = transform(self._grids[x][y])
        self._grids[x][y] = state
        next_x, next_y = self._get_pointed_grid(x, y, state)
        reward = 90
        while self._is_in_grids(next_x, next_y):
            state = transform(self._grids[next_x][next_y])
            self._grids[next_x][next_y] = state
            next_x, next_y = self._get_pointed_grid(next_x, next_y, state)
            reward += 90
        self._step += 1
        done = False
        if self._step >= self._max_step:
            done = True
        return self._digitizing_grids(), reward, done, {}

    