var inputNames = {
	0: "U",
	1: "L",
	2: "D",
	3: "R",
	4: "A",
	tick: "T",
	undo: "UNDO",
	restart: "RESTART"
};

function formatInputs(inputs) {
	var groups = [];
	var group = "";
	var input;
	var i;
	if (inputs.length === 0) {
		return "(none)";
	}
	for (i = 0; i < inputs.length; i++) {
		input = inputNames[inputs[i]] || String(inputs[i]);
		if (input.length > 1) {
			if (group) {
				groups.push(group);
				group = "";
			}
			groups.push(input);
		} else {
			group += input;
			if (group.length === 5) {
				groups.push(group);
				group = "";
			}
		}
	}
	if (group) {
		groups.push(group);
	}
	return groups.join(" ");
}

function simulationSetup(data) {
	var setup = [
		{
			label: "Starting level index",
			value: String(data[3] === undefined ? 0 : data[3])
		},
		{ label: "Input", value: formatInputs(data[1]) }
	];
	if (data[4] !== undefined) {
		setup.push({ label: "Random seed", value: String(data[4]) });
	}
	if (data[5] !== undefined) {
		setup.push({
			label: "Expected audio",
			value: data[5].length === 0 ? "(none)" : data[5].join(";")
		});
	}
	return setup;
}

for (var i=0;i<testdata.length;i++) {
	test(
		testdata[i][0],
		{
			group: "simulation",
			source: testdata[i][1][0],
			setup: simulationSetup(testdata[i][1]),
			output: testdata[i][1][2]
		},
		function(num){
			return function(){
				var td = testdata[num];
				return runTest(td[1],td[0]);
			};
		}(i)
	);
}

for (var j=0;j<errormessage_testdata.length;j++) {
	test(
		"🐛"+errormessage_testdata[j][0],
		{
			group: "compiler-messages",
			source: errormessage_testdata[j][1][0],
			output: errormessage_testdata[j][1][1].join("\n")
		},
		function(num){
			return function(){
				var td = errormessage_testdata[num];
				if (td[1].length!==3){
					throw new Error(
						"Error/Warning message testdata has the wrong number of fields. " +
						"Accidentally pasted in level recording data?"
					);
				}
				return runCompilationTest(td[1],td[0]);
			};
		}(j)
	);
}

PuzzleScriptTests.start();
