# Startup functions called

play.html: "calls the following:"
compiler: compile
engine: setGameState
engine: loadLevelFromLevelDat
engine: drawMessageScreen
graphics: canvasResize
graphics: redraw


# Steps to regain sanity (and encapsulation)

- convert files into modules and import functions (search replace \nfunction with export function)
- add webpack
  - will find unused functions and vars


# Unused JS files

deleted:    js/Blob.js
deleted:    js/FileSaver.js
deleted:    js/_gist.js
deleted:    js/_gist2.js
deleted:    js/addlisteners.js
deleted:    js/addlisteners_editor.js
deleted:    js/altfont.js
deleted:    js/buildStandalone.js
deleted:    js/codemirror/active-line.js
deleted:    js/codemirror/dialog.js
deleted:    js/codemirror/match-highlighter.js
deleted:    js/codemirror/search.js
deleted:    js/codemirror/searchcursor.js
deleted:    js/console.js
deleted:    js/debug.js
deleted:    js/editor.js
deleted:    js/gamedat.js
deleted:    js/jsgif/GIFEncoder.js
deleted:    js/jsgif/LZWEncoder.js
deleted:    js/jsgif/NeuQuant.js
deleted:    js/jsgif/b64.js
deleted:    js/layout.js
deleted:    js/makegif.js
deleted:    js/riffwave.js
deleted:    js/soundbar.js
deleted:    js/toolbar.js

# Server

These are the packages the server uses:

- http://koajs.com/
- https://github.com/koajs/koa
- https://github.com/koajs/static
- https://github.com/koajs/route
- https://github.com/koajs/bodyparser
- https://github.com/octokit/rest.js
- https://www.npmjs.com/package/dotenv
