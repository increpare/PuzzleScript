
var fs = require('fs');

var test_src = fs.readFileSync("../demo/sokoban_basic.txt", 'utf8');
//remove all '\r's from test_src
test_src = test_src.replace(/\r/g, "");

console.log(test_src);

unitTesting=true;

compile(["restart"],test_src);

console.log(state);
console.log(level);
console.log("done")