#!/usr/bin/env node

/* 

creates a highly compressed release build in bin of the contents of src

packages used:

npm i tar  html-minifier-terser ycssmin  google-closure-compiler concat  pngcrush-bin inliner ncp rimraf gifsicle

*/


//#node-qunit-phantomjs  tests/tests.html --timeout 40
console.log("===========================");
var start = new Date()

// console.log("clearing whitepsace from demos")
// cd demo
// find . -type f \( -name "*.txt" \) -exec perl -p -i -e "s/[ \t]*$//g" {} \;
// cd ..

console.log("removing bin")

var fs = require("fs");

fs.rmdirSync("./bin", { recursive: true });

fs.mkdirSync('./bin');

console.log("inlining standalone template")

var Inliner = require('inliner');

new Inliner('./src/standalone.html', function (error, html) {
  // compressed and inlined HTML page
  fs.writeFileSync("./src/standalone_inlined.txt",html,'utf8');

  console.log("Copying files")
  var ncp = require('ncp').ncp;
  ncp.limit = 16;
  ncp("./src", "./bin/", function (err) {
    if (err) {
      return console.error(err);
    }
    console.log("echo optimizing pngs");

    const rimraf = require('rimraf');
    rimraf.sync('./bin/images/*.png');

    const imagemin = require('imagemin');
    const imageminPngcrush = require('imagemin-pngcrush');

    (async () => {
        await imagemin(['./src/images/*.png'], {
            destination: './bin/images/',
            plugins: [
                imageminPngcrush(["-brute","-reduce","-rem allb"])
            ]
        });
    


        const {execFileSync} = require('child_process');
        const gifsicle = require('gifsicle');
        
        console.log('Optimizing documentation gifs');
        
        var glob = require("glob")
        glob("./bin/Documentation/images/*.gif", {}, async function (er, files) {
            for (filename of files){
                execFileSync(gifsicle, ['-O2','-o', filename, filename]);
            }

            
            console.log('Images optimized');


                        
            fs.rmdirSync("./bin/js", { recursive: true });
            fs.mkdirSync('./bin/js');
            fs.rmdirSync("./bin/css", { recursive: true });
            fs.mkdirSync('./bin/css');
            fs.rmdirSync("./bin/tests", { recursive: true });
            fs.rmdirSync("./bin/Levels", { recursive: true });

            console.log('compressing css');

            const concat = require('concat');
            await concat(["./src/css/docs.css", 
                "./src/css/codemirror.css", 
                "./src/css/midnight.css", 
                "./src/css/console.css", 
                "./src/css/gamecanvas.css", 
                "./src/css/soundbar.css", 
                "./src/css/layout.css", 
                "./src/css/toolbar.css", 
                "./src/css/dialog.css", 
                "./src/css/show-hint.css"], 
                "./bin/css/combined.css");

            console.log('css files concatenated')


            var cssmin = require('ycssmin').cssmin;
            var css = fs.readFileSync("./bin/css/combined.css", encoding='utf8');
            var min = cssmin(css);
            fs.writeFileSync("./bin/css/combined.css",min,encoding="utf8");
            

            var css = fs.readFileSync("./bin/Documentation/css/bootstrap.css", encoding='utf8');
            var min = cssmin(css);
            fs.writeFileSync("./bin/Documentation/css/bootstrap.css",min,encoding="utf8");

            console.log("running google closure compiler");


            const ClosureCompiler = require('google-closure-compiler').compiler;

            const closureCompiler = new ClosureCompiler({
                js: [  
                    "./src/js/Blob.js",
                    "./src/js/FileSaver.js",
                    "./src/js/jsgif/b64.js",
                    "./src/js/jsgif/LZWEncoder.js",
                    "./src/js/jsgif/NeuQuant.js",
                    "./src/js/jsgif/GIFEncoder.js",
                    "./src/js/debug.js",
                    "./src/js/globalVariables.js",
                    "./src/js/font.js",
                    "./src/js/rng.js",
                    "./src/js/riffwave.js",
                    "./src/js/sfxr.js",
                    "./src/js/codemirror/codemirror.js",
                    "./src/js/codemirror/active-line.js",
                    "./src/js/codemirror/dialog.js",
                    "./src/js/codemirror/search.js",
                    "./src/js/codemirror/searchcursor.js",
                    "./src/js/codemirror/match-highlighter.js",
                    "./src/js/codemirror/show-hint.js",
                    "./src/js/codemirror/anyword-hint.js",
                    "./src/js/colors.js",
                    "./src/js/graphics.js",
                    "./src/js/inputoutput.js",
                    "./src/js/mobile.js",
                    "./src/js/buildStandalone.js",
                    "./src/js/engine.js",
                    "./src/js/parser.js",
                    "./src/js/editor.js",
                    "./src/js/compiler.js",
                    "./src/js/console.js",
                    "./src/js/soundbar.js",
                    "./src/js/toolbar.js",
                    "./src/js/layout.js",
                    "./src/js/addlisteners.js",
                    "./src/js/addlisteners_editor.js",
                    "./src/js/makegif.js"],
                js_output_file:'./bin/js/scripts_compiled.js',
                compilation_level: 'SIMPLE'
            });
            const compilerProcess = closureCompiler.run((exitCode, stdOut, stdErr) => {
                

                          
                const closureCompiler2 = new ClosureCompiler({
                    js: [  
                        "./src/js/globalVariables.js",
                        "./src/js/debug_off.js",
                        "./src/js/font.js",
                        "./src/js/rng.js",
                        "./src/js/riffwave.js",
                        "./src/js/sfxr.js",
                        "./src/js/codemirror/codemirror.js",
                        "./src/js/colors.js",
                        "./src/js/graphics.js",
                        "./src/js/engine.js",
                        "./src/js/parser.js",
                        "./src/js/compiler.js",
                        "./src/js/inputoutput.js",
                        "./src/js/mobile.js"],
                    js_output_file:'./bin/js/scripts_play_compiled.js',
                    compilation_level: 'SIMPLE'
                });
                const compilerProcess2 = closureCompiler2.run((exitCode, stdOut, stdErr) => {
                    console.log("compilation done");

                    var editor = fs.readFileSync("./bin/editor.html", encoding='utf8');
                    editor = editor.replace(/<script src="js\/[A-Za-z0-9_\/-]*\.js"><\/script>/g, "");
                    editor = editor.replace(/<!--TOREPLACE-->/g, '<script src="js\/scripts_compiled.js"><\/script>');
                    editor = editor.replace(/<link rel="stylesheet" href="[A-Za-z0-9_\/-]*\.css">/g, '');
                    editor = editor.replace(/<!--CSSREPLACE-->/g, '<link rel="stylesheet" href="css\/combined.css">');
                    fs.writeFileSync("./bin/editor.html",editor, encoding='utf8');

                    var player = fs.readFileSync("./bin/play.html", encoding='utf8');
                    player = player.replace(/<script src="js\/[A-Za-z0-9_\/-]*\.js"><\/script>/g, "");
                    player = player.replace(/<!--TOREPLACE-->/g, '<script src="js\/scripts_play_compiled.js"><\/script>');
                    fs.writeFileSync("./bin/play.html",player, encoding='utf8');

                    console.log("compressing html");
                    
                    var minify = require('html-minifier-terser').minify;
                    
                    glob("./bin/*.html", {}, async function (er, files) {
                        for (filename of files){
                           var lines=fs.readFileSync(filename, encoding='utf8');
                           var result = minify(lines);
                           fs.writeFileSync(filename,result);
                        }
                    });

                })

              });

          })


    })();

    });


});


/*
echo gzipping site
cd ../bin
./gzipper
rm README.md
rm gzipper
rm commit
cd ../src
end=`date +%s`
runtime=$((end-start))
time=`date "+%H:%M:%S"`
echo script end time : $time 
echo script took $runtime seconds
echo ===========================
*/