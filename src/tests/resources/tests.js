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

// Big game sources and end states are stored as {file:"test_scripts/..."}
// references. Served over http, read the .txt files themselves; opened from
// disk, use the copies run_tests_node.js writes to external_testdata.js.
function collectExternalRefs() {
	var refs = [];
	var lists = [testdata, errormessage_testdata];
	for (var l = 0; l < lists.length; l++) {
		for (var t = 0; t < lists[l].length; t++) {
			var fields = lists[l][t][1];
			for (var f = 0; f < fields.length; f++) {
				var value = fields[f];
				if (value !== null && typeof value === "object" && !Array.isArray(value) && typeof value.file === "string") {
					refs.push({ fields: fields, index: f, file: value.file });
				}
			}
		}
	}
	return refs;
}

function loadExternalRef(file) {
	if (location.protocol === "http:" || location.protocol === "https:") {
		return fetch(file, { cache: "no-store" }).then(function(response) {
			if (!response.ok) {
				throw new Error(response.status + " " + response.statusText);
			}
			return response.text();
		});
	}
	if (typeof external_testdata !== "undefined" && external_testdata.hasOwnProperty(file)) {
		return Promise.resolve(external_testdata[file]);
	}
	return Promise.reject(new Error("not in resources/external_testdata.js (run node run_tests_node.js to rebuild it)"));
}

function resolveExternalRefs() {
	var missing = [];
	return Promise.all(collectExternalRefs().map(function(ref) {
		return loadExternalRef(ref.file).then(function(text) {
			ref.fields[ref.index] = text;
		}, function(error) {
			missing.push(ref.file + ": " + error.message);
			ref.fields[ref.index] = "";
		});
	})).then(function() {
		return missing;
	});
}

function registerTests(missingExternalFiles) {
if (missingExternalFiles.length > 0) {
	test(
		"external test files",
		{ group: "simulation", source: "", output: missingExternalFiles.join("\n") },
		function() {
			throw new Error("Couldn't load external test files:\n" + missingExternalFiles.join("\n"));
		}
	);
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
}

resolveExternalRefs().then(registerTests);
