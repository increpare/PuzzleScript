var lastDownTarget = null;
var canvas = null;
var input = document.createElement('TEXTAREA');
let consolecache = [];
function canvasResize() {

}

function redraw() {

}

function forceRegenImages(){

}

var levelString;
var inputString;
var outputString;

function consolePrintFromRule(text){
	window.console.log(text);
}
function consolePrint(text,urgent,linenumber,turnIndex) {
	window.console.log(text);
}
function consoleError(text) {
	window.console.log(text);
}

function consoleCacheDump(scrolldown=true) {
	
	for (let i = 0; i < consolecache.length-1; i++) {
		console.log(consolecache[i]);
	}
}
var editor = {
	getValue : function () { return levelString }
}

function addToDebugTimeline(level, lineNumber){}

function UnitTestingThrow(error){

	QUnit.push(false,false,false,error.message+"\n"+error.stack);
	console.error(error);
}