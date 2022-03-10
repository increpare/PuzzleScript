var canSetHTMLColors=false;
var canDump=true;
var recordingStartsFromLevel=0;
var inputHistory=[];
var compiledText;
var canOpenEditor=true;
var IDE=true;

var debugger_turnIndex=0;
var debug_visualisation_array=[];
var diffToVisualize=null;

function convertLevelToString() {
	var out = '';
	var seenCells = {};
	var i = 0;
	for (var y = 0; y < level.height; y++) {
		for (var x = 0; x < level.width; x++) {
			var bitmask = level.getCell(x + y * level.width);
			var objs = [];
			for (var bit = 0; bit < 32 * STRIDE_OBJ; ++bit) {
				if (bitmask.get(bit)) {
					objs.push(state.idDict[bit])
				}
			}
			objs.sort();
			objs = objs.join(" ");
			/* replace repeated object combinations with numbers */
			if (!seenCells.hasOwnProperty(objs)) {
				seenCells[objs] = i++;
				out += objs + ":";
			}
			out += seenCells[objs] + ",";
		}
		out += '\n';
	}
	return out;
}

function stripHTMLTags(html_str){
	var div = document.createElement("div");
	div.innerHTML = html_str;
	var text = div.textContent || div.innerText || "";
	return text.trim();
}

function dumpTestCase() {
	//compiler error data
	var levelDat = compiledText;
	var errorStrings_stripped = errorStrings.map(stripHTMLTags);
	var resultarray = [levelDat,errorStrings_stripped,errorCount];
	var resultstring = JSON.stringify(resultarray);
	resultstring = `<br>
	[<br>
		"${state.metadata.title||"untitled test"}",<br>
		${resultstring}<br>
	],`;
	selectableint++;
	var tag = 'selectable'+selectableint;
	consolePrint("<br>Compilation error/warning data (for error message tests - errormessage_testdata.js):<br><br><br><span id=\""+tag+"\" onclick=\"selectText('"+tag+"',event)\">"+resultstring+"</span><br><br><br>",true);

	
	//if the game is currently running and not on the title screen, dump the recording data
	if (!titleScreen) {
		//normal session recording data
		var levelDat = compiledText;
		var input = inputHistory.concat([]);
		var outputDat = convertLevelToString();

		var resultarray = [levelDat,input,outputDat,recordingStartsFromLevel,loadedLevelSeed];
		var resultstring = JSON.stringify(resultarray);
		resultstring = `<br>
		[<br>
			"${state.metadata.title||"untitled test"}",<br>
			${resultstring}<br>
		],`;
		
		selectableint++;
		var tag = 'selectable'+selectableint;
		
		consolePrint("<br>Recorded play session data (for play session tests - testdata.js):<br><br><br><span id=\""+tag+"\" onclick=\"selectText('"+tag+"',event)\">"+resultstring+"</span><br><br><br>",true);
	}

}

function clearInputHistory() {
	if (canDump===true) {
		inputHistory=[];
		recordingStartsFromLevel = curlevel;
	}
}

function pushInput(inp) {
	if (canDump===true) {
		inputHistory.push(inp);
	}
}