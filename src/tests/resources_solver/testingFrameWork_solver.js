function runTest_Solver(testname,script_string) {
	console.log("Running test "+testname);
	unitTesting=true;
	lazyFunctionGeneration=false;
	levelString=script_string;
	errorStrings = [];
	errorCount=0;
	textmode=false;

	try
	{
		compile(["loadLevel",0],levelString,0);
		for (let i=0;i<errorStrings.length;i++) {
			const s = errorStrings[i];
			console.log(s);
		}
		console.log("level count: "+state.levels.length);
		var solvedCount=0;
		//for each level, load it and solve it
		for (let i=1;i<state.levels.length;i++) {
			console.log("solving level "+i);
			loadLevelFromState(state,i);
			//time the solve
			const startTime = performance.now();
			let solved = solve(true);
			const endTime = performance.now();
			const timeTaken = (endTime - startTime).toFixed(0);
			QUnit.assert.ok(solved,"solved level "+i+" in "+timeTaken+"ms");
			if (solved) {
				solvedCount++;
			} else {
				console.log("level "+i+" not solved");
			}

		}
		console.log("solved count: "+solvedCount);
	} 
	catch (error){
		QUnit.push(false,false,false,error.message+"\n"+error.stack);
		console.error(error);
	}
	return true;
}
