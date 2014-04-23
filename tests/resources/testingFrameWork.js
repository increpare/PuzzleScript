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
    replayQueue = inputDat.slice().reverse();
    while(replayQueue.length) {
        var val=replayQueue.pop();
        if (val==="undo") {
            pushInput("undo");
	        DoUndo();
        } else if (val==="restart") {
            pushInput("restart");
	    	DoRestart();
	    } else if (val==="tick") {
            autoTickGame();
        } else if (val==="quit" || val==="win") {
            continue;
        } else {
            pushInput(val);
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
		console.log("Got");
		console.log(levelString);
		console.log("Expected");
		console.log(dataarray[2]);
		return false;
	}
}
