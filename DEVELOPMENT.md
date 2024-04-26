# PuzzleScript Development Guide

## Hello

This document is about recompiling puzzlescript from source.  If you're just interested in learning about using the engine, rather than developing it, the documentation is [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).

## Structure
The structure of PuzzleScript is the following:

* In the `./src/` directory you have the 'raw' version of PuzzleScript, which is itself runnable, just not compressed/optimised.
* When you run `node ./compile.js` it generates a compressed/optimized version of PuzzleScript, which is what people see on [puzzlescript.net](https://www.puzzlescript.net/).  Running this also updates the `./src/standalone_inlined.txt` file, which is the template that is used for exported standalone PuzzleScript games.

## Getting compilation working

`./compile.js` uses [node](https://nodejs.org). So first off you have to install that.  Then you need to install the packages that it uses:

```
npm i rimraf compress-images web-resource-inliner ncp gifsicle@5.3.0 concat ycssmin terser gzipper html-minifier-terser glob@8   
```

Then you should be able to compile the site (outputted to the `./bin/` directory) with 

```
node compile.js
```

## Standalone-exporting

If you load `./src/editor.html` directly, by double-clicking it or whatever, exporting won't work because the browser sandboxing prevents the `XMLHttpRequest` for `standalone_inlined.txt` from working.  To get it to work you need to run a local http server - see for instance [this](http://www.linuxjournal.com/content/tech-tip-really-simple-http-server-python) for an example of how to set one up with python.

Also, remember you need to run `./compile.js` to generate the updated `./src/standalone_inlined.txt` template (which is generated from `./src/standalone.html`).

## Tests

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
