PuzzleScript
============

Open Source HTML5 Puzzle Game Engine

Try it out at https://www.puzzlescript.net

-----

Dev instructions here - https://groups.google.com/forum/#!searchin/puzzlescript/development/puzzlescript/yptIpY9hlng/cjfrOPy_4jcJ


Server Development
==================

1. Run `npm install` to install the [nodejs](https://nodejs.org) dependencies
1. Run `npm start` to start the server
1. Visit http://localhost:3000 to see the website

To configure Gist uploading:

1. copy `.env.example` to `.env`
1. visit https://github.com/settings/tokens/new , select the `gist` checkbox, and click `[Generate token]`
1. paste the token into `.env`
1. run `npm start` to start the server


Server Packages
---------------

These are the packages the server uses:

- http://koajs.com/
- https://github.com/koajs/koa
- https://github.com/koajs/static
- https://github.com/koajs/route
- https://github.com/koajs/bodyparser
- https://github.com/octokit/rest.js
- https://www.npmjs.com/package/dotenv


-----

The MIT License (MIT)

Copyright (c) 2013 Stephen Lavelle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
