var canDump=true;
inputHistory=[];
var compiledText;
var canOpenEditor=true;

function convertLevelToString() {
	return  JSON.stringify(level);
}

function dumpTestCase() {
	var levelDat = compiledText;
	var input = inputHistory.concat([]);
	var outputDat = convertLevelToString();

	var resultarray = [levelDat,input,outputDat];
	var resultstring = JSON.stringify(resultarray);
	consolePrint("<br><br><br>"+resultstring+"<br><br><br>");
}