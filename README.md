ScriptDoctor
============

## Scraping and parsing games

```
python scrape_pedro.py
python parse_lark.py
python gen_trees.py
```
I'm still working on `gen_trees.py`
- Convert games to canonical (graph-like) form
- Determine if two games are functionally equivalent via this form
- Reimplement game mechanics etc. in numpy and jax, as an RL environment

## Fine-tuning a model

```
python finetune.py
```

## Evolving games with OpenAI queries

Put your OpenAI API key in a file called `.env`, which will have a single line that reads `OPENAI_API_KEY=yourkeyhere`.

To install requirements: `pip install -r requirements.txt`. To run a local server: `python server.py`. Then, open the local IP address. You should see the PuzzleScript editor page displayed, and some GPT slop appearing in the PuzzleScript terminal and level code editor. 

The main function is run client-side, from inside `ScriptDoctor.js`, which is included in the `editor.html` (which is served by the server). This JS process asks for a game from the server (which makes an OpenAI API call), then throws it in the editor.

Next: playtesting. Making generated games not suck.

Notes:
- We made a single edit to `compile.js` to fix an issue importing gzipper, but we don't actually use the compressed version of the engine at the moment (the one in `bin`---instead just using the one in `src`).

TODO (feel free to claim a task---they're relatively standalone):
- Submodule focused solely on adding new levels to functioning games
- Save gifs of solutions being played out (there is some existing functionality for saving gifs in the js codebase---use it)
- Feed screenshots of generated levels to GPT-4o to iterate on sprites
- Some kind of evolutionary loop that will generate a bunch of games for us, diverse/novel along some axis (implemented, need to debug)

## Running experiments

To sweep over fewshot and chain of thought prompting, uncomment `sweep()` in `src/js/ScriptDoctor.js`, launch the server with `python server.py` and open the webpage at `127.0.0.1:8000` (or whatever pops up in the terminal where you've launched the server). Then the javascript code, and the `sweep()` function, will be run. Once this is done, run `python compute_edit_distances.py` then `python eval_fewshot_cot_sweep.py` to generate tables of results.

To generate game ideas, run `python brainstorm.py`, then uncomment `fromIdeaSweep()` in `src/js/ScriptDoctor.js`, launch the server and open the webpage, then run `python compute_from_idea_edit_distances.py` and `python eval_from_idea_sweep.py`.

PuzzleScript
============

Open Source HTML5 Puzzle Game Engine

Try it out at https://www.puzzlescript.net.

-----

If you're interested in recompiling/modifing/hacking the engine, there is [development setup info here](DEVELOPMENT.md).  If you're just interested in learning how to use the engine/make games in it, [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).
