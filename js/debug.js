var canSetHTMLColors=false;
var canDump=true;
var canYoutube=false;
var inputHistory=[];
var compiledText;
var canOpenEditor=true;
var IDE=true;

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
			out += objs + ",";
		}
		out += '\n';
	}
	return out;
}

function compressLevelString(s) {
	var seenCells = {};
	var i = 0;

	/* replace repeated object combinations with numbers */
	function deflater(match, objs) {
		if (!seenCells.hasOwnProperty(objs)) {
			seenCells[objs] = i++;
			return objs + ":" + seenCells[objs] + ",";
		}
		return seenCells[objs] + ",";
	}

	return s.replace(/([^,\n]+),/g, deflater);
}

/* inverse of compressLevelString */
function decompressLevelString(s) {
	var cellMappings = {};

	function inflater(match, ref, objs, assignment) {
		if (ref !== undefined) {
			return cellMappings[+ref] + ",";
		}
		cellMappings[assignment] = objs;
		return objs + ",";
	}

	return s.replace(/(\d+),|([^:,\n]+):(\d+),/g, inflater);
}

function dumpTestCase() {
	var levelDat = compiledText;
	var input = inputHistory.concat([]);
	var outputDat = compressLevelString(convertLevelToString());

	var resultarray = [levelDat,input,outputDat,curlevel,loadedLevelSeed];
	var resultstring = JSON.stringify(resultarray);
	consolePrint("<br><br><br>"+resultstring+"<br><br><br>",true);
}

function clearInputHistory() {
	if (canDump===true) {
		inputHistory=[];
	}
}

function pushInput(inp) {
	if (canDump===true) {
		inputHistory.push(inp);
	}
}
