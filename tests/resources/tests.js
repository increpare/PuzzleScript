var inputVals = {0 : "UP",1: "LEFT",2:"DOWN",3:"RIGHT",4:"ACTION",tick:"TICK",undo:"UNDO",restart:"RESTART"};

for (var i=0;i<testdata.length;i++) {
	test(
		testdata[i][0], 
		function(td){
			return function(assert){
				runTest(assert, td[1]);
			};
		}(testdata[i])
	);
}

function runTest(assert, dataarray) {
	levelString = dataarray[0];
	var inputDat = dataarray[1];
	var testresult = dataarray[2];

	var input="";
	for (var j=0;j<inputDat.length;j++) {
		if (j>0) {
			input+=", ";
		}
		input += inputVals[inputDat[j]];
	}
	var errormessage = levelString + "\n\n\ninput : "+input;

	unitTesting=true;
	errorStrings = [];
	errorCount=0;

	for (var i=0;i<errorStrings.length;i++) {
		var s = errorStrings[i];
		throw s;
	}

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
			DoUndo();
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

	var origLevelString = convertLevelToString();
	var levelString = compressLevelString(origLevelString);
	assert.equal(decompressLevelString(levelString), origLevelString);
	assert.equal(origLevelString, decompressLevelString(testresult), errormessage);
}
