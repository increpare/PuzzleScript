from javascript import require

ps = require('puzzlescript')

handler = ps.EmptyGameEngineHandler()
game = open("games/blockfaker.txt").read()

game_data = ps.Parser.parse(game).data
engine = ps.GameEngine(game_data, handler)

# Set the first level and perform a couple actions
engine.setLevel(0)
engine.press("RIGHT")
delta = engine.tick()
print(f"Delta: {delta}")
print(f"Level snapshot: {engine.saveSnapshotToJSON()}")