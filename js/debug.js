var canSetHTMLColors=false;
var canDump=true;
var canYoutube=false;
inputHistory=[];
var compiledText;
var canOpenEditor=true;
dumpTraceHooks=[];

function pushInput(inp) {
	if (canDump===true) {
    inputHistory.push(inp);
  }
}

function clearInputs() {
	if (canDump===true) {
		inputHistory=[];
	}
}

function randomDirAvailable() { 
    return false; 
}

function popRandomDir() {
    throw new Exception("No choices available"); 
}

function randomEntIdxAvailable() {
    return false; 
}

function popRandomEntIdx() {
    throw new Exception("No choices available"); 
}

function randomRuleIdxAvailable() {
    return false; 
}

function popRandomRuleIdx() {
    throw new Exception("No choices available"); 
}

function addDumpTraceHook(fn) {
    dumpTraceHooks.push(fn);
}

function dumpTrace() {
    var title = state.metadata.title || state.title;
    for(var i=0; i < dumpTraceHooks.length; i++) {
        dumpTraceHooks[i](
            title, 
            curlevel, 
            inputHistory
        );
    }
}

function convertLevelToString() {
	return JSON.stringify(level);
}

function dumpTestCase() {
	var levelDat = compiledText;
	var input = inputHistory.concat([]);
	var outputDat = convertLevelToString();

	var resultarray = [levelDat,input,outputDat,curlevel];
	var resultstring = JSON.stringify(resultarray);
	consolePrint("<br><br><br>"+resultstring+"<br><br><br>");
}