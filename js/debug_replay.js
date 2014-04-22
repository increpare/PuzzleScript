var canSetHTMLColors=false;
var canDump=true;
var canYoutube=false;
inputHistory=[];
replayQueue=null;
var compiledText;
var canOpenEditor=true;
dumpTraceHooks=[];
var IDE=true;

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
    if(!replayQueue || !replayQueue.length) { return false; }
    var last = replayQueue[replayQueue.length-1];
    if(!isNaN(last)) { return false; }
    return last.substr(0,9) == "randomDir";
}

function popRandomDir() {
    if(!randomDirAvailable()) {
        throw new Exception("No direction choices available"); 
    }
    return parseInt(replayQueue.pop().substr(10));
}

function randomEntIdxAvailable() {
    if(!replayQueue || !replayQueue.length) { return false; }
    var last = replayQueue[replayQueue.length-1];
    if(!isNaN(last)) { return false; }
    return last.substr(0,12) == "randomEntIdx";
}

function popRandomEntIdx() {
    if(!randomEntIdxAvailable()) {
        throw new Exception("No entity index choices available"); 
    }
    return parseInt(replayQueue.pop().substr(13));
}

function randomRuleIdxAvailable() {
    if(!replayQueue || !replayQueue.length) { return false; }
    var last = replayQueue[replayQueue.length-1];
    if(!isNaN(last)) { return false; }
    return last.substr(0,13) == "randomRuleIdx";
}

function popRandomRuleIdx() {
    if(!randomRuleIdxAvailable()) {
        throw new Exception("No rule index choices available"); 
    }
    return parseInt(replayQueue.pop().substr(14));
}

function addDumpTraceHook(fn) {
    dumpTraceHooks.push(fn);
}

function dumpTrace() {
    if(replayQueue) { return; }
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