#!/usr/bin/env node

/* 

creates a highly compressed release build in bin of the contents of src

(See DEVELOPMENT.md for information on how to set up/use this script)

*/

const fs = require("fs");
const path = require('path');
const { execFileSync } = require('child_process');

const rimraf = require('rimraf');
const compress_images = require("compress-images");
let webResourceInliner = require("web-resource-inliner");
const ncp = require('ncp').ncp;
const gifsicle = require('gifsicle').default || require('gifsicle');
const concat = require('concat');
const { transform } = require('lightningcss');
const { minify } = require('@swc/core');
const { TraceMap, eachMapping } = require('@jridgewell/trace-mapping');
const { GenMapping, addMapping, setSourceContent, toEncodedMap } = require('@jridgewell/gen-mapping');
const { Compress } = require('gzipper');
const htmlminify = require('html-minifier-terser').minify;
const glob = require("glob")

// Resolve build inputs relative to this script, including when invoked elsewhere.
process.chdir(__dirname);

function cssmin(css) {
    return transform({ code: Buffer.from(css), minify: true }).code.toString();
}

// Source maps still include source files, but local tooling and backups are not
// website content. Keep deployment files such as .htaccess.
const developmentNames = new Set(['.git', '.claude', '.codex', '.cursor', '.local-ai', 'AGENTS.md']);
function includeInRelease(filename) {
    const parts = path.relative(path.join(__dirname, 'src'), filename).split(path.sep);
    return parts[0] !== 'tests' &&
        !parts.some(part => developmentNames.has(part) || /\.(bak|orig)$/i.test(part));
}

//print all paths to all modules above
let lines = fs.readFileSync(".build/buildnumber.txt", encoding = 'utf-8');
let buildnum = parseInt(lines);
buildnum++;
fs.writeFileSync(".build/buildnumber.txt", buildnum.toString(), encoding = 'utf-8');

console.log("===========================");
console.log('build number ' + buildnum)

let start = new Date()

console.log("removing bin")

rimraf.sync("./bin");
fs.mkdirSync('./bin');

console.log("Copying files")
ncp.limit = 16;
ncp("./src", "./bin/", { filter: includeInRelease }, function (err) {
    if (err) {
        return console.error(err);
    }
    console.log("echo optimizing pngs");

    rimraf.sync('./bin/images/*.png');

    (async () => {

        compress_images(
            "./src/images/*.png",
            "./bin/images/",
            { compress_force: false, statistic: false, autoupdate: true, pathLog: './build/log/compress-images' }, false,
            { jpg: { engine: "mozjpeg", command: ["-quality", "60"] } },
            { png: { engine: "pngcrush", command: ["-reduce", "-brute"] } },
            { svg: { engine: "svgo", command: "--multipass" } },
            { gif: { engine: "gifsicle", command: ["--colors", "64", "--use-col=web"] } },

            function (error, completed, statistic) {
                // console.log("-------------");
                // console.log(error);
                // console.log(completed);
                // console.log(statistic);
                // console.log("-------------");
            }
        );

        console.log('Optimizing gallery gifs');

        const galGifDir = "./bin/Gallery/gifs";

        fs.readdirSync(galGifDir).forEach(file => {
            if (fs.lstatSync(path.resolve(galGifDir, file)).isDirectory()) {
            } else {
                if (path.extname(file).toLowerCase() === ".gif") {
                    execFileSync(gifsicle, ['--batch', '-O2', galGifDir + "/" + file])
                }
            }
        });

        console.log('Optimizing documentation gifs');

        glob("./bin/Documentation/images/*.gif", {}, async function (er, files) {
            for (filename of files) {
                execFileSync(gifsicle, ['-O2', '-o', filename, filename]);
            }

            console.log('Images optimized');

            //remove ".bin/js dir if it exists"
            rimraf.sync('./bin/js');
            rimraf.sync('./bin/css');
            rimraf.sync('./bin/tests');

            fs.mkdirSync('./bin/js');
            fs.mkdirSync('./bin/css');

            console.log('compressing css');

            await concat(["./src/css/docs.css",
                "./src/css/codemirror.css",
                "./src/css/midnight.css",
                "./src/css/console.css",
                "./src/css/gamecanvas.css",
                "./src/css/soundbar.css",
                "./src/css/layout.css",
                "./src/css/toolbar.css",
                "./src/css/dialog.css",
                "./src/css/show-hint.css",
                "./src/css/player.css",
                "./src/css/editorplayer.css"],
                "./bin/css/combined.css");

            console.log('css files concatenated')




            let css = fs.readFileSync("./bin/css/combined.css", encoding = 'utf8');
            let min = cssmin(css);
            fs.writeFileSync("./bin/css/combined.css", min, encoding = "utf8");

            css = fs.readFileSync("./src/css/player.css", encoding = 'utf8');
            min = cssmin(css);
            fs.writeFileSync("./bin/css/player.css", min, encoding = "utf8");

            css = fs.readFileSync("./bin/Documentation/css/bootstrap.css", encoding = 'utf8');
            min = cssmin(css);
            fs.writeFileSync("./bin/Documentation/css/bootstrap.css", min, encoding = "utf8");

            console.log("running js minification");

            async function generateFrom(toinclude, output_src, output_bin) {
                let lineOffset = 0;
                const sources = toinclude.map(fpath => {
                    const code = fs.readFileSync(fpath, 'utf8');
                    const source = { name: "source/" + fpath.slice(9), code, lineOffset };
                    lineOffset += (code.match(/\n/g) || []).length + 1;
                    return source;
                });
                const result = await minify(sources.map(source => source.code).join('\n'), {
                    // Inline HTML handlers and generated engine functions share these globals.
                    module: false,
                    toplevel: false,
                    compress: { toplevel: false, passes: 3 },
                    mangle: { toplevel: false },
                    format: { comments: 'some', inlineScript: true },
                    sourceMap: true,
                    inlineSourcesContent: false
                });

                // SWC maps the concatenated input. Restore the original file boundaries
                // so browser debugging still opens the individual source files.
                const map = new GenMapping({ file: output_src });
                sources.forEach(source => setSourceContent(map, source.name, source.code));
                eachMapping(new TraceMap(result.map), mapping => {
                    const generated = { line: mapping.generatedLine, column: mapping.generatedColumn };
                    if (mapping.originalLine === null) {
                        addMapping(map, { generated });
                        return;
                    }
                    const source = sources.find((source, index) =>
                        index === sources.length - 1 || mapping.originalLine <= sources[index + 1].lineOffset);
                    addMapping(map, {
                        generated,
                        source: source.name,
                        original: { line: mapping.originalLine - source.lineOffset, column: mapping.originalColumn },
                        name: mapping.name || undefined
                    });
                });
                fs.writeFileSync(output_bin, result.code + "\n//# sourceMappingURL=" + output_src + ".map\n");
                fs.writeFileSync(output_bin + ".map", JSON.stringify(toEncodedMap(map)));
            };

            let includes_editor = [
                "./src/js/Blob.js",
                "./src/js/FileSaver.js",
                "./src/js/jsgif/LZWEncoder.js",
                "./src/js/jsgif/NeuQuant.js",
                "./src/js/jsgif/GIFEncoder.js",
                "./src/js/storagewrapper.js",
                "./src/js/debug.js",
                "./src/js/bitvec.js",
                "./src/js/level.js",
                "./src/js/languageConstants.js",
                "./src/js/globalVariables.js",
                "./src/js/font.js",
                "./src/js/rng.js",
                "./src/js/riffwave.js",
                "./src/js/sfxr.js",
                "./src/js/colorhelpers.js",
                "./src/js/codemirror/codemirror.js",
                "./src/js/codemirror/panel.js",
                "./src/js/codemirror/active-line.js",
                "./src/js/codemirror/dialog.js",
                "./src/js/codemirror/searchcursor.js",
                "./src/js/codemirror/search.js",
                "./src/js/codemirror/match-highlighter.js",
                "./src/js/codemirror/show-hint.js",
                "./src/js/codemirror/rule-transform.js",
                "./src/js/codemirror/anyword-hint.js",
                "./src/js/codemirror/comment.js",
                "./src/js/colors.js",
                "./src/js/graphics.js",
                "./src/js/inputoutput.js",
                "./src/js/mobile.js",
                "./src/js/buildStandalone.js",
                "./src/js/engine.js",
                "./src/js/parser.js",
                "./src/js/github.js",
                "./src/js/imagepaste.js",
                "./src/js/editor.js",
                "./src/js/compiler.js",
                "./src/js/console.js",
                "./src/js/soundbar.js",
                "./src/js/toolbar.js",
                "./src/js/layout.js",
                "./src/js/addlisteners.js",
                "./src/js/addlisteners_editor.js",
                "./src/js/makegif.js"];
                
            await generateFrom(
                includes_editor,
                "scripts_compiled.js",
                "./bin/js/scripts_compiled.js");


            let includes_play = [
                "./src/js/storagewrapper.js",
                "./src/js/bitvec.js",
                "./src/js/level.js",
                "./src/js/languageConstants.js",
                "./src/js/globalVariables.js",
                "./src/js/debug_off.js",
                "./src/js/font.js",
                "./src/js/rng.js",
                "./src/js/riffwave.js",
                "./src/js/sfxr.js",
                "./src/js/codemirror/stringstream.js",
                "./src/js/colors.js",
                "./src/js/graphics.js",
                "./src/js/engine.js",
                "./src/js/parser.js",
                "./src/js/github.js",
                "./src/js/compiler.js",
                "./src/js/inputoutput.js",
                "./src/js/mobile.js",
                "./src/js/play.js"];

            await generateFrom(
                includes_play,
                "scripts_play_compiled.js",
                "./bin/js/scripts_play_compiled.js");

            // Standalone has its own script list, without the web player's GitHub loader.
            let standalone_raw = fs.readFileSync("./src/standalone.html", 'utf8');
            await generateFrom(
                Array.from(standalone_raw.matchAll(/<script src="([^"]+)"><\/script>/g), match => "./src/" + match[1]),
                "scripts_standalone_compiled.js",
                "./bin/js/scripts_standalone_compiled.js");

            await ncp("./src/js", "./bin/js/source", { filter: includeInRelease }, function (err) {
                if (err) {
                    return console.error(err);
                }
            });

            console.log("compilation done");

            let editor = fs.readFileSync("./bin/editor.html", encoding = 'utf8');
            editor = editor.replace(/<script src="js\/[A-Za-z0-9_\/-]*\.js"><\/script>/g, "");
            editor = editor.replace(/<!--___SCRIPTINSERT___-->/g, '<script src="js\/scripts_compiled.js"><\/script>');
            editor = editor.replace(/<link rel="stylesheet" href="[A-Za-z0-9_\/-]*\.css">/g, '');
            editor = editor.replace(/<!--CSSREPLACE-->/g, '<link rel="stylesheet" href="css\/combined.css">');
            d = new Date();
            const monthname = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
            editor = editor.replace(/<!--BUILDNUMBER-->/g, `build ${buildnum.toString()}, ${d.getDate()}-${monthname[d.getMonth()]}-${d.getFullYear()}`);
            fs.writeFileSync("./bin/editor.html", editor, encoding = 'utf8');

            let player = fs.readFileSync("./bin/play.html", encoding = 'utf8');
            player = player.replace(/<script src="js\/[A-Za-z0-9_\/-]*\.js"><\/script>/g, "");
            player = player.replace(/<!--___SCRIPTINSERT___-->/g, '<script src="js\/scripts_play_compiled.js"><\/script>');
            fs.writeFileSync("./bin/play.html", player, encoding = 'utf8');

            console.log("inlining standalone template")

            //src one first:
            webResourceInliner.html({
                fileContent: standalone_raw,
                relativeTo: 'src/',
            },
                function (err, inlined) {
                    if (err) {
                        console.log(err)
                    } else {
                        fs.writeFileSync("./src/standalone_inlined.txt", inlined);
                    }
                });

            //then bin one:
            standalone_raw = standalone_raw.replace(/<script src="js\/[A-Za-z0-9_\/-]*\.js"><\/script>/g, "");
            standalone_raw = standalone_raw.replace(/<!--___SCRIPTINSERT___-->/g, '<script src="js\/scripts_standalone_compiled.js"><\/script>');
            webResourceInliner.html({
                fileContent: standalone_raw,
                relativeTo: 'bin/',
            },
                async function (err, inlined) {
                    if (err) {
                        console.log(err)
                    } else {
                        let minified = await htmlminify(inlined,
                            {
                                collapseBooleanAttributes: true,
                                collapseWhitespace: true,
                                conservativeCollapse: true,
                                removeComments: true,
                                removeEmptyAttributes: true,
                                removeOptionalTags: true,
                                useShortDoctype: true,
                                minifyCSS: true,
                                minifyURLs: true,
                                // Placeholders such as __GAMETITLE__ are replaced later and may contain spaces.
                                removeAttributeQuotes: false
                            });
                        fs.writeFileSync("./bin/standalone_inlined.txt", minified);
                    }

                    //delete ./bin/standalone.html
                    fs.unlinkSync("./bin/standalone.html");

                    console.log("compressing html");

                    glob("./bin/*.html", {}, async function (er, files) {
                        for (filename of files) {
                            let lines = fs.readFileSync(filename, encoding = 'utf8');
                            let result = await htmlminify(lines, {
                                collapseBooleanAttributes: true,
                                collapseWhitespace: true,
                                conservativeCollapse: true,
                                removeComments: true,
                                removeEmptyAttributes: true,
                                removeOptionalTags: true,
                                useShortDoctype: true,
                                removeAttributeQuotes: false
                            });
                            fs.writeFileSync(filename, result);
                        }
                    });

                    (async function a() {

                        files = glob.sync("./bin/**/*.js");
                        files = files.concat(glob.sync("./bin/**/*.html"));
                        files = files.concat(glob.sync("./bin/**/*.css"));
                        files = files.concat(glob.sync("./bin/**/*.txt"));

                        for (let i = 0; i < files.length; i++) {
                            let file = files[i];
                            let comp = new Compress(file);
                            await comp.run();
                        }

                        console.log("Files compressed. All good!");

                    })();
                });

        });

    })();

});
