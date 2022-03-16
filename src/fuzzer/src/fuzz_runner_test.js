
var fs = require('fs');


//read file from supplied argument
var input = fs.readFileSync(process.argv[2], 'utf8');
//remove \r from file
input = input.replace(/\r/g, '');
    
compile(["restart"],input);
console.log(state);

return errorCount===0?1:0;