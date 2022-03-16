var lastDownTarget = null;
var canvas = null;
var input = document.createElement('TEXTAREA');

function canvasResize() {

}

function redraw() {

}

function forceRegenImages(){

}

var levelString;
var inputString;
var outputString;

function consolePrintFromRule(text){}
function consolePrint(text,urgent,linenumber,turnIndex) {}
function consoleError(text) {
//	window.console.log(text);
}

function consoleCacheDump() {}
var editor = {
	getValue : function () { return levelString }
}

function addToDebugTimeline(level, lineNumber){}

var code = {
	editorreference:editor
}

window.form1 = {
	code: code,
}