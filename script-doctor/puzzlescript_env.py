import os

from javascript import require
# ps = require('puzzlescript')
ps = require('./node_modules/puzzlescript/lib/index.js')

from config import Moves

class PuzzlescriptEnvironment():
    def __init__(self, path_or_gamestr):

        if os.path.isfile(path_or_gamestr):
            game = open(path_or_gamestr).read()
        else:
            game = path_or_gamestr

        self.game_data = ps.Parser.parse(game).data
        self.engine = ps.GameEngine(self.game_data, ps.EmptyGameEngineHandler())
        self.engine.setLevel(0)

    def step(self, action: Moves):
        self.engine.press(action)
        delta = self.engine.tick()
        return delta
    
    def get_snapshot(self):
        '''
        Returns a snapshot of the current game state converted
        into a nested set of tuples (which allows us to hash the
        state)
        '''
        snapshot_proxy = self.engine.saveSnapshotToJSON()
        processed = self._process_snapshot(snapshot_proxy)
        
        return processed

    def _process_snapshot(self, snapshot):
        if isinstance(snapshot, str):
            return snapshot

        else:
            container = []
            for item in snapshot:
                container.append(self._process_snapshot(item))

            return tuple(container)
        
    def load_snapshot(self, snapshot):
        '''
        Load a snapshotted game state
        '''
        self.engine.loadSnapshotFromJSON(snapshot)
    
if __name__ == "__main__":
    import time

    env = PuzzlescriptEnvironment("games/sokoban_basic.txt")
    snapshot = env.get_snapshot()
    
    # Test a winning sequence of moves
    MOVE_SEQUENCE = [
        Moves.RIGHT,
        Moves.RIGHT,
        Moves.DOWN,
        Moves.LEFT,
        Moves.UP,
        Moves.LEFT,
        Moves.LEFT,
        Moves.DOWN,
        Moves.DOWN,
        Moves.RIGHT,
        Moves.UP,
        Moves.UP,
        Moves.UP,
    ]

    for move in MOVE_SEQUENCE:
        delta = env.step(move)
        print(f"didWinGame: {delta['didWinGame']}\tdidLevelChange: {delta['didLevelChange']}")

    # Do some time trials
    env.load_snapshot(snapshot)
    
    N = 2500
    start = time.time()
    for _ in range(N):
        env.step(Moves.RIGHT)
        env.step(Moves.LEFT)
        env.step(Moves.UP)
        env.step(Moves.DOWN)
    end = time.time()

    print(f"Steps / sec: {(4 * N / (end - start)):.2f}")

    # print(env.engine.myFunc())
    start = time.time()
    env.engine.myFunc(N)
    end = time.time()
    print(f"myFunc steps / sec: {(4 * N / (end - start)):.2f}")
    

