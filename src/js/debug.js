let canSetHTMLColors=false;
let canDump=true;
let recordingStartsFromLevel=0;
let inputHistory=[];
let soundHistory=[];
let compiledText;
let canOpenEditor=true;
let IDE=true;

let debugger_turnIndex=0;
let debug_visualisation_array=[];
let diffToVisualize=null;

function convertLevelToString() {
	let out = '';
	let seenCells = {};
	let i = 0;
	for (let y = 0; y < level.height; y++) {
		for (let x = 0; x < level.width; x++) {
			let bitmask = level.getCell(x + y * level.width);
			let objs = [];
			for (let bit = 0; bit < 32 * STRIDE_OBJ; ++bit) {
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
	let div = document.createElement("div");
	div.innerHTML = html_str;
	let text = div.textContent || div.innerText || "";
	return text.trim();
}

function dumpTestCase() {
	//compiler error data
	let levelDat = compiledText;
	let errorStrings_stripped = errorStrings.map(stripHTMLTags);
	let resultarray = [levelDat,errorStrings_stripped,errorCount];
	let resultstring = JSON.stringify(resultarray);
	let escapedtitle = (state.metadata.title||"untitled test").replace(/"/g, '\\"');
	resultstring = `<br>
	[<br>
		"${escapedtitle}",<br>
		${resultstring}<br>
	],`;
	selectableint++;
	let tag = 'selectable'+selectableint;
	consolePrint("<br>Compilation error/warning data (for error message tests - errormessage_testdata.js):<br><br><br><span id=\""+tag+"\" onclick=\"selectText('"+tag+"',event)\">"+resultstring+"</span><br><br><br>",true);

	
	//if the game is currently running and not on the title screen, dump the recording data
	if (!titleScreen) {
		//normal session recording data
		let levelDat = compiledText;
		let input = inputHistory.concat([]);
		let sounds = soundHistory.concat([]);
		let outputDat = convertLevelToString();

		let resultarray = [levelDat,input,outputDat,recordingStartsFromLevel,loadedLevelSeed,sounds];
		let resultstring = JSON.stringify(resultarray);
		resultstring = `<br>
		[<br>
			"${state.metadata.title||"untitled test"}",<br>
			${resultstring}<br>
		],`;
		
		selectableint++;
		let tag = 'selectable'+selectableint;
		
		consolePrint("<br>Recorded play session data (for play session tests - testdata.js):<br><br><br><span id=\""+tag+"\" onclick=\"selectText('"+tag+"',event)\">"+resultstring+"</span><br><br><br>",true);
	}

}

function clearInputHistory() {
	if (canDump===true) {
		inputHistory=[];
		soundHistory=[];
		recordingStartsFromLevel = curlevel;
	}
}

function pushInput(inp) {
	if (canDump===true) {
		inputHistory.push(inp);
	}
}

function pushSoundToHistory(seed) {
	if (canDump===true) {
		soundHistory.push(seed);
	}
}