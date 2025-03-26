'use strict';
function runTest_Solver(testname,script_string) {
	unitTesting=true;
	lazyFunctionGeneration=false;
	levelString=script_string;
	errorStrings = [];
	errorCount=0;

	console.log("Running test "+testname);
	
	compile(["compile"],script_string,0);

	console.log(state);
	console.log("Level Count: "+levels.levelCount);
}


function runCompilationTest(dataarray,testname) {
	console.log("Running test "+testname);
	unitTesting=true;
	lazyFunctionGeneration=false;
	levelString=dataarray[0];
	var recordedErrorStrings=dataarray[1];
	var recordedErrorCount=dataarray[2];
	errorStrings = [];
	errorCount=0;

	try{
		compile(["restart"],levelString);
	} catch (error){
		QUnit.push(false,false,false,error.message+"\n"+error.stack);
		console.error(error);
	}

	
	unitTesting=true;
	lazyFunctionGeneration=false;

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
		var simulated_summary = strippedErrorStrings.join("\n");
		var recorded_summary = recordedErrorStrings.join("\n");
		QUnit.assert.equal(simulated_summary,recorded_summary)
		return false;
	}
	return true;
}
