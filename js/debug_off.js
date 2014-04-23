var canSetHTMLColors=true;
var canDump=false;
var canOpenEditor=false;
var canYoutube=true;
var IDE=false;

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

function dumpTrace() {
    //nop
}

function consolePrint(str){
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