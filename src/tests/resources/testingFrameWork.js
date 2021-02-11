function runTest(dataarray) {
	unitTesting=true;
	levelString=dataarray[0];
	errorStrings = [];
	errorCount=0;

	for (var i=0;i<errorStrings.length;i++) {
		var s = errorStrings[i];
		throw s;
	}

	var inputDat = dataarray[1];
	var targetlevel = dataarray[3];
	
	var randomseed = dataarray[4]!==undefined ? dataarray[4] : null;

	if (targetlevel===undefined) {
		targetlevel=0;
	}
	compile(["loadLevel",targetlevel],levelString,randomseed);

	while (againing) {
		againing=false;
		processInput(-1);			
	}
	
	for(var i=0;i<inputDat.length;i++) {
		var val=inputDat[i];
		if (val==="undo") {
			DoUndo(false,true);
		} else if (val==="restart") {
			DoRestart();
		} else if (val==="tick") {
			processInput(-1);
		} else {
			processInput(val);
		}
		while (againing) {
			againing=false;
			processInput(-1);			
		}
	}

	unitTesting=false;
	var levelString = convertLevelToString();
	var success = levelString == dataarray[2];
	if (success) {
		return true;
	} else {
		return false;
	}
}


function runCompilationTest(dataarray) {
	unitTesting=true;
	levelString=dataarray[0];
	var recordedErrorStrings=dataarray[1];
	var recordedErrorCount=dataarray[2];
	errorStrings = [];
	errorCount=0;

	try{
		compile(["restart"],levelString);
	} catch (error){
		console.log(error);
	}

	var strippedErrorStrings = errorStrings.map(stripHTMLTags);
	if (errorCount!==recordedErrorCount){
		return false;
	}

	var i_recorded=0;
	var i_simulated=0;
	for (i_simulated=0;i_simulated<strippedErrorStrings.length && i_recorded<recordedErrorStrings.length;i_simulated++){
		var simulated_error = strippedErrorStrings[i_simulated].replace(/\s/g, '');
		var recorded_error = recordedErrorStrings[i_recorded].replace(/\s/g, '');//html replaces '  ' with ' ', so I'm just comparing stripping all spaces lol.
		if (simulated_error===recorded_error){
			i_recorded++;
		}
	}

	if (i_recorded<recordedErrorStrings.length){
		return false;
	}
	return true;
}
