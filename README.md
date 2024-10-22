ScriptDoctor
============

To install requirements: `pip install -r requirements.txt`. To run a local server: `python server.py`. Then, open the local IP address. You should see the PuzzleScript editor page displayed, and some GPT slop appearing in the PuzzleScript terminal and level code editor. 

The main function is run client-side, from inside `ScriptDoctor.js`, which is included in the `editor.html` (which is served by the server). This JS process asks for a game from the server (which makes an OpenAI API call), then throws it in the editor.

Next: playtesting. Making generated games not suck.

Notes:
- We made a single edit to `compile.js` to fix an issue importing gzipper, but we don't actually use the compressed version of the engine at the moment (the one in `bin`---instead just using the one in `src`).

TODO (feel free to claim a task---they're relatively standalone):
- Submodule focused solely on adding new levels to functioning games
- Ask for natural language descriptions of game semantics (e.g. "a game about a hungry caterpillar eating leaves while avoiding predatory birds"), then use these as prompts for generating games
- Save gifs of solutions being played out (there is some existing functionality for saving gifs in the js codebase---use it)
- Feed screenshots of generated levels to GPT-4o to iterate on sprites
- Some kind of evolutionary loop that will generate a bunch of games for us, diverse/novel along some axis

PuzzleScript
============

Open Source HTML5 Puzzle Game Engine

Try it out at https://www.puzzlescript.net.

-----

If you're interested in recompiling/modifing/hacking the engine, there is [development setup info here](DEVELOPMENT.md).  If you're just interested in learning how to use the engine/make games in it, [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).
