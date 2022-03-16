//use file library
var fs = require('fs');
//include exec
var exec = require('child_process').exec;

var files = [
    "./src/fuzz_header.js",
	"../tests/resources/wrapper.js",
    "../js/storagewrapper.js",
    "../js/globalVariables.js",
    "../js/debug.js",
    "../js/font.js",
    "../js/rng.js",
    "../js/riffwave.js",
    "../js/sfxr.js",
    "../js/codemirror/stringstream.js",
    "../js/colors.js",
    "../js/engine.js",
    "../js/parser.js",
    "../js/compiler.js",
    "../js/soundbar.js",
    "./src/fuzz_runner.js",
];

var concatenated = files.map(function(file) {
  return fs.readFileSync(file, 'utf8');
}).join('\n');

//write concatenated file to disk
fs.writeFileSync("./generated/concatenated.js", concatenated);

//execute concatenated.js
exec('node ./generated/concatenated.js', function(error, stdout, stderr) {
    if (error) {
        console.log(error);
        console.log(stderr);
        return;
    }
    console.log(stdout);
    });

