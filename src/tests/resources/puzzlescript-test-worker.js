// The engine's test wrapper needs only these non-rendering browser hooks.
var window = self;
var document = {
	title: "PuzzleScript Tests",
	URL: self.location.href,
	createElement: function () { return {}; },
	getElementById: function () { return null; },
	body: { classList: { contains: function () { return false; } } }
};
var workerStorage = Object.create(null);
var localStorage = {
	getItem: function (key) { return key in workerStorage ? workerStorage[key] : null; },
	setItem: function (key, value) { workerStorage[key] = String(value); },
	removeItem: function (key) { delete workerStorage[key]; }
};

// Keep the engine script list in sync with tests.html.
importScripts(
	"testingFrameWork.js",
	"wrapper.js",
	"../../js/storagewrapper.js",
	"../../js/bitvec.js",
	"../../js/level.js",
	"../../js/languageConstants.js",
	"../../js/globalVariables.js",
	"../../js/debug.js",
	"../../js/font.js",
	"../../js/rng.js",
	"../../js/riffwave.js",
	"../../js/sfxr.js",
	"../../js/colorhelpers.js",
	"../../js/codemirror/stringstream.js",
	"../../js/colors.js",
	"../../js/engine.js",
	"../../js/parser.js",
	"../../js/compiler.js",
	"../../js/soundbar.js",
	"puzzlescript-test-runner.js"
);

self.onmessage = function (event) {
	var request = event.data;
	var compilation = null;
	var testRecord = {
		callback: function () {
			if (request.group === "compiler-messages") {
				if (request.data.length !== 3) {
					throw new Error("Error/Warning message testdata has the wrong number of fields. " +
						"Accidentally pasted in level recording data?");
				}
				compilation = compileTest(request.data, request.name);
				return compilation !== null;
			}
			return runTest(request.data, request.name);
		}
	};
	PuzzleScriptTests.execute(testRecord);
	self.postMessage({
		status: testRecord.status,
		duration: testRecord.duration,
		comparisons: testRecord.comparisons,
		error: testRecord.error,
		compilation: compilation
	});
};
self.postMessage({ ready: true });
