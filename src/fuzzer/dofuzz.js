//use file library
var fs = require('fs');

var testdata = fs.readFileSync('../tests/resources/testdata.js', 'utf8');
//exec testdata
eval(testdata);
var errormessage_testdata = fs.readFileSync('../tests/resources/errormessage_testdata.js', 'utf8');
//exec testdata
eval(errormessage_testdata);

console.log(testdata.length);
console.log(errormessage_testdata.length);

var samples=[];
for (var i=0;i<testdata.length;i++){
    var src=testdata[i][1][0];
    fs.writeFileSync('./corpus/testdata'+i+'.txt', src);
}
for (var i=0;i<errormessage_testdata.length;i++){
    var src=errormessage_testdata[i][1][0];
    fs.writeFileSync('./corpus/errormessage_testdata'+i+'.txt', src);
}

//include exec
var {spawn, exec} = require('child_process');

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


var files_test = [
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
    "./src/fuzz_runner_test.js",
];

var concatenated = files.map(function(file) {
  return fs.readFileSync(file, 'utf8');
}).join('\n');

var concatenated_test = files_test.map(function(file) {
    return fs.readFileSync(file, 'utf8');
  }).join('\n');
  
//write concatenated file to disk
fs.writeFileSync("./generated/concatenated.js", concatenated);
fs.writeFileSync("./generated/concatenated_test.js", concatenated_test);

//execute concatenated.js
var spawn = spawn('jsfuzz.cmd',['--versifier','--only-ascii','--timeout=10','./generated/concatenated.js','corpus'])
spawn.stdout.on('data', function(msg){         
    console.log(msg.toString().trim())
});

