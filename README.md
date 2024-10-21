ScriptDoctor
============

Some new requirements for ScriptDoctor:

`npm install openai`

We made a single edit to `compile.js` to fix an issue importing gzipper, but we don't actually use the compressed version of the engine at the moment (the one in `bin`---instead just using the one in `src`).

To run a local server we use `python server.py`. The main function is run client-side, from inside `ScriptDoctor.js`, which is included in the `editor.html` (which is served by the server). This JS process asks for a game from the server (which makes an OpenAI API call), then throws it in the editor.

Next: compilation, iterative feedback, playtesting. Also fewshot prompting.


PuzzleScript
============

Open Source HTML5 Puzzle Game Engine

Try it out at https://www.puzzlescript.net.

-----

If you're interested in recompiling/modifing/hacking the engine, there is [development setup info here](DEVELOPMENT.md).  If you're just interested in learning how to use the engine/make games in it, [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).
