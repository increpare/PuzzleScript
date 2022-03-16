
var fs = require('fs');

module.exports = {
  fuzz: input => {
    input = input.toString('utf8') 
    
    var compileerror=false
    compile(["restart"],input);

    return errorCount===0?1:0;
  }
}