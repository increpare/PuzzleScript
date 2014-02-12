var canSetHTMLColors=true;
var canDump=false;
var canOpenEditor=false;
var canYoutube=true;

function stripTags(str) {
	var div = document.createElement("div");
	div.innerHTML = str;
	var result = div.textContent || div.innerText || "";
	return result;
}

function pushInput(inp) {
    //nop
}

function clearInputs() {
    //nop
}

function randomDirAvailable() { return false; }
function popRandomDir() { throw new Exception("No choices available"); }
function randomEntIdxAvailable() { return false; }
function popRandomEntIdx() { throw new Exception("No choices available"); }
function randomRuleIdxAvailable() { return false; }
function popRandomRuleIdx() { throw new Exception("No choices available"); }

function dumpTrace() {
    //nop
}

function consolePrint(str){
/*	var errorText = document.getElementById("errormessage");
	
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";*/
}

function consoleError(str,lineNumber){
	var errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";
}

function logErrorNoLine(str){
	var errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";
}

function logBetaMessage(str){
	var errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";	
}