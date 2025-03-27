async function runtests(){
	//load up the index html file "resources_solver/solver_tests/"
	const index_html = await fetch("./resources_solver/solver_tests/");
	const index_text = await index_html.text();
	//parse it as a dom
	const index_dom = new DOMParser().parseFromString(index_text, "text/html");
	//extract the names of all the links in the index html file
	const links = index_dom.querySelectorAll("a");
	//extract the names of all the links in the index html file
	const testdata = [];
	for (let i = 0; i < links.length; i++) {
		testdata.push(links[i].innerText);
	}


	for (var i=0;i<testdata.length;i++) {
		//testdata is an array of filenames - i wnat to load them first
		const testname = testdata[i];
		const response = await fetch("./resources_solver/solver_tests/"+testname);		
		let script_string = await response.text();
		//change crlf to lf
		script_string = script_string.replace(/\r\n/g, "\n");
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