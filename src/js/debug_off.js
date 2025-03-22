let canSetHTMLColors=true;
let canDump=false;
let canOpenEditor=false;
let IDE=false;
const diffToVisualize=null;

function stripTags(str) {
	let div = document.createElement("div");
	div.innerHTML = str;
	let result = div.textContent || div.innerText || "";
	return result;
}

function consolePrint(linenumber,inspect_ID){
/*	let errorText = document.getElementById("errormessage");
	
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";*/
}

function consolePrintFromRule(str,rule,urgent){
/*	let errorText = document.getElementById("errormessage");
	
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";*/
}

function consoleCacheDump(str){
	
}

function UnitTestingThrow(error){}

function consoleError(str,lineNumber){
	let errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";
}

function logErrorNoLine(str){
	let errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";
}

function clearInputHistory() {}
function pushInput(inp) {}
function pushSoundToHistory(seed) {}
