async function runtests(){
	for (var i=0;i<testdata.length;i++) {
		//testdata is an array of filenames - i wnat to load them first
		const testname = testdata[i];
		const response = await fetch("./resources_solver/solver_tests/"+testname);		
		const script_string = await response.text();
		console.log(testname);
		test(
			testname, 
			function(){					
					ok(runTest_Solver(testname,script_string),script_string);
				}
		);
	}
}

runtests();