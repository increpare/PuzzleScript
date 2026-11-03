'use strict';

const canSetHTMLColors=true;
const canDump=false;
const canOpenEditor=false;
const IDE=false;
const diffToVisualize=null;

function stripTags(str) {
	let template = document.createElement("template");
	template.innerHTML = str;
	let result = template.content.textContent || "";
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
	errorText.appendChild(document.createTextNode(str));
	errorText.appendChild(document.createElement("br"));
}

function logErrorNoLine(str){
	let errorText = document.getElementById("errormessage");
	str=stripTags(str);
	errorText.appendChild(document.createTextNode(str));
	errorText.appendChild(document.createElement("br"));
}

function clearInputHistory() {}
function pushInput(inp) {}
function pushSoundToHistory(seed) {}
