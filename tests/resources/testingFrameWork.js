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
	if (targetlevel===undefined) {
		targetlevel=0;
	}
	compile(["loadLevel",targetlevel],levelString);
  replayQueue = inputDat.slice().reverse();
  while(replayQueue.length) {
    var val=replayQueue.pop();
    if(isNaN(val) && val.substr(0,6) == "random") {
      throw new Exception("Replay queue has unconsumed random "+val);
    }
		if (val==="undo") {
			DoUndo();
		} else if (val==="restart") {
			DoRestart();
		} else if (val==="wait") {
      processInput(-1);
    } else {
			processInput(val);
		}
		while (againing) {
			againing=false;
			processInput(-1);			
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