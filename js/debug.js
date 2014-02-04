var canSetHTMLColors=false;
var canDump=true;
var canYoutube=false;
inputHistory=[];
replayQueue=null;
var compiledText;
var canOpenEditor=true;

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
  //TODO
  return false; 
}

function popRandomDir() {
  //TODO
  throw new Exception("No choices available"); 
}

function randomEntIdxAvailable() {
  //TODO
  return false; 
}

function popRandomEntIdx() {
  //TODO
  throw new Exception("No choices available"); 
}

function randomRuleIdxAvailable() {
  //TODO
  return false; 
}

function popRandomRuleIdx() {
  //TODO
  throw new Exception("No choices available"); 
}

function convertLevelToString() {
	return  JSON.stringify(level);
}

function dumpTestCase() {
	var levelDat = compiledText;
	var input = inputHistory.concat([]);
	var outputDat = convertLevelToString();

	var resultarray = [levelDat,input,outputDat,curlevel];
	var resultstring = JSON.stringify(resultarray);
	consolePrint("<br><br><br>"+resultstring+"<br><br><br>");
}