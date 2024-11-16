import os

from javascript import require
# ps = require('puzzlescript')
ps = require('./node_modules/puzzlescript/lib/index.js')

game = open("games/sokoban_match3.txt").read()
game_data = ps.Parser.parse(game).data
engine = ps.GameEngine(game_data, ps.EmptyGameEngineHandler())
# print(engine.bfs(0))
print(engine.bfs(1))