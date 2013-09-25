function runTest(dataarray) {
	unitTesting=true;
	levelString=dataarray[0];

	for (var i=0;i<errorStrings.length;i++) {
		var s = errorStrings[i];
		throw s;
	}

	var inputDat = dataarray[1];
	compile(["loadLevel",0],levelString);
	for(var i=0;i<inputDat.length;i++) {
		var val=inputDat[i];
		if (val==="undo") {
			DoUndo();
		} else if (val==="restart") {
			DoRestart();
		} else {
			processInput(val);
		}
	}
	var calculatedOutput = JSON.stringify(level.dat);
	var preparedOutput = dataarray[2];
	var preparedLevel;
	eval("preparedLevel = " + preparedOutput);
	preparedOutput = JSON.stringify(preparedLevel.dat);
	unitTesting=false;
	return calculatedOutput === preparedOutput;
}