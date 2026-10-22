# PuzzleScript Development Guide

## Hello

This document is about recompiling puzzlescript from source.  If you're just interested in learning about using the engine, rather than developing it, the documentation is [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).

## Structure
The structure of PuzzleScript is the following:

* In the `./src/` directory you have the 'raw' version of PuzzleScript, which is itself runnable, just not compressed/optimised.
* When you run `node ./compile.js` it generates a compressed/optimized version of PuzzleScript, which is what people see on [puzzlescript.net](https://www.puzzlescript.net/).  Running this also updates the `./src/standalone_inlined.txt` file, which is the template that is used for exported standalone PuzzleScript games.

Other directories:

* `src/runserver.bat` and `src/runserver.sh` start a local server; the build copies them into `bin/` to serve the compiled site. `upload_website.sh` is the maintainer's deployment helper at the repository root.
* `server/` contains the Python CGI services, which are deployed separately from the static website.
* `src/tests/` contains the browser and Node test runners and regression fixtures.
* `.build/buildnumber.txt` is the tracked build counter. The separate `build/` directory holds ignored build diagnostics, including image-compression logs.

Keep local configuration outside `src/`. The build excludes tests, local assistant configuration, and `.bak`/`.orig` files, while retaining the server launchers, `.htaccess`, and JavaScript sources used for debugging.

`src/js/codemirror/codemirror.js.orig` is the unmodified `lib/codemirror.js` from the [CodeMirror 5.65.21 npm release](https://www.npmjs.com/package/codemirror/v/5.65.21), kept for comparing our changes in `codemirror.js`.

## Getting compilation working

`./compile.js` uses [node](https://nodejs.org). So first off you have to install that.  Then you need to install the packages that it uses, with this command:

```
npm install
```

This installs the packages listed in `package.json`.

Then you should be able to compile the site (outputted to the `./bin/` directory) with 

```
node compile.js
```

`npm run build` is an alias for `node compile.js`. Compilation updates the tracked files `.build/buildnumber.txt` and `src/standalone_inlined.txt` as well as rebuilding `bin/`. Review these generated changes before committing.

## Running locally

With Python 3 available as `python`, run `npm start` from the repository root, or run `src/runserver.bat` on Windows. On macOS/Linux, run `bash src/runserver.sh` (uses `python3`).

These commands serve `src/`, trying port 8000 first and choosing a free port if it is unavailable. The server prints the localhost URL and LAN URLs, and accepts connections from other devices on the LAN. Open `/editor.html` for the editor or `/tests/tests.html` for the browser tests. Stop the server with Ctrl+C.

To preview a compiled build, run `bin/runserver.bat` on Windows or `bash bin/runserver.sh` on macOS/Linux. Each launcher uses `runserver.py` in its own directory, so the source and compiled servers can run together on different ports.

## Standalone-exporting

If you load `./src/editor.html` directly, by double-clicking it or whatever, exporting won't work because the browser sandboxing prevents the `XMLHttpRequest` for `standalone_inlined.txt` from working.  To get it to work you need to run a local http server - see [Running locally](#running-locally) above.

Also, remember you need to run `./compile.js` to generate the updated `./src/standalone_inlined.txt` template (which is generated from `./src/standalone.html`).

## Deployment helpers

`upload_website.sh` is the site maintainer's personal deployment script at the repository root. Run it with Bash; it asks for `UPLOAD`, builds the site, then uploads `bin/` using rsync and SSH.

`server/access_token.py` and `server/gist_proxy.py` are deployed separately to the server's CGI directory. Configure their Python dependencies, credentials, and allowed origins as described in their comments. Keep live credentials out of tracked source. Their public CGI URLs in `src/auth.html` and `src/js/github.js` are unchanged; the local static server does not execute CGI scripts.

## Tests

The Node tests can be run with `npm test` (equivalent to `node src/tests/run_tests_node.js`).

The tests can be run by opening `./src/tests/tests.html`.  There are two kinds of tests:

* Tests based on short play-sessions recorded in the editor - it checks for a given start state and input state that a particular end-state will be reached.   
* Tests based on error messages.  This is not based on input, but records all the error messages and checks that they are still present in the current version of the engine.  If new errors are generated, that's also ok, so long as the old ones are still there.  Note that if you change the wording of an error message in PuzzleScript, you'll also need to change it in the test data.

The two kinds of tests are stored in `./src/tests/resources/testdata.js` and `./src/tests/resources/errormessage_testdata.js` respectively.  

Here's how you make a new test: Press *Ctrl/Cmd+J* in the editor to generate test data in the console (Be sure to have compiled/launched the game first).  You'll see something like this:

```
Compilation error/warning data (for error message tests - errormessage_testdata.js):

[
    "Name of project",
    [big long array]
],

Recorded play session data (for play session tests - testdata.js):

[
    "Name of test",
    [big long array]
]
```

If you're at the title screen you won't get the second bit.  The recorded play session records all the input from when the level was loaded up until the moment of generation.  Yeah anyway just paste whichever bit you want as an entry in  `./src/tests/resources/testdata.js` or `./src/tests/resources/errormessage_testdata.js` and you're away.
