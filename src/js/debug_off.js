var canSetHTMLColors=true;
var canDump=false;
var canOpenEditor=false;
var canYoutube=true;
var IDE=false;
const diffToVisualize=null;

function stripTags(str) {
	var div = document.createElement("div");
	div.innerHTML = str;
	var result = div.textContent || div.innerText || "";
	return result;
}

function consolePrint(str,urgent){
/*	var errorText = document.getElementById("errormessage");
	
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";*/
}

function consolePrintFromRule(str,rule,urgent){
/*	var errorText = document.getElementById("errormessage");
	
	str=stripTags(str);
	errorText.innerHTML+=str+"<br>";*/
}

function consoleCacheDump(str){
	
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

function clearInputHistory() {}
function pushInput(inp) {}