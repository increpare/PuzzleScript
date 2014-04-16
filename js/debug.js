var canSetHTMLColors=false;
var canDump=true;
var canYoutube=false;
inputHistory=[];
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

function dumpTestCase() {
	var levelDat = compiledText;
	var input = inputHistory.concat([]);
	var outputDat = convertLevelToString();

	var resultarray = [levelDat,input,outputDat,curlevel,loadedLevelSeed];
	var resultstring = JSON.stringify(resultarray);
	consolePrint("<br><br><br>"+resultstring+"<br><br><br>",true);
}
