var abortSolver = false;
var solving = false;
var showingSolution = false;
var stopShowingSolution = false;

const timeout = ms => new Promise(res => setTimeout(res, ms))

function byScoreAndLength(a, b) {
	// if (a[2] != b[2]) {
	// 	return a[2] < b[2];
	// } else {
	// 	return a[0] < b[0];
	// }
	
	if (a[0] != b[0]) {
		return a[0] < b[0];
	} else {
		return a[2] < b[2];
	}
}

var distanceTable;

var act2str = "uldrx";
var exploredStates;

async function solve() {
	if (levelEditorOpened) return;
	if (showingSolution) return;
	if (solving) return;
	if (textMode || state.levels.length === 0) return;

	precalcDistances();
	abortSolver = false;
	muted = true;
	solving = true;
	// restartTarget = backupLevel();
	DoRestart();
	hasUsedCheckpoint = false;
	backups = [];
	var oldDT = deltatime;
	deltatime = 0;
	var actions = [0, 1, 2, 3, 4];
	if ('noaction' in state.metadata) {
		actions = [0, 1, 2, 3];
	}
	exploredStates = {};
	exploredStates[level.objects] = [level.objects.slice(0), -1];
	var queue;
	queue = new FastPriorityQueue(byScoreAndLength);
	queue.add([0, level.objects.slice(0), 0]);
	consolePrint("searching...");
	// var solvingProgress = document.getElementById("solvingProgress");
	// var cancelLink = document.getElementById("cancelClickLink");
	// cancelLink.hidden = false;
	// console.log("searching...");
	var iters = 0;
	var size = 1;

	var startTime = performance.now();

	while (!queue.isEmpty()) {
		if (abortSolver) {
			consolePrint("solver aborted");
			// cancelLink.hidden = true;
			break;
		}
		iters++;
		if (iters > 500) {
			iters = 0;
			// consolePrint("searched: " + size + " queue: " + discovered);
			// console.log(discovered, size);
			// solvingProgress.innerHTML = "searched: " + size;
			redraw();
			await timeout(1);
		}
		var temp = queue.poll();
		var parentState = temp[1];
		var numSteps = temp[2];
		// console.log(numSteps);
		shuffleALittle(actions);
		for (var i = 0, len = actions.length; i < len; i++) {
			for (var k = 0, len2 = parentState.length; k < len2; k++) {
				level.objects[k] = parentState[k];
			}
			var changedSomething = processInput(actions[i]);
			while (againing) {
				changedSomething = processInput(-1) || changedSomething;
			}

			if (changedSomething) {
				if (level.objects in exploredStates) {
					continue;
				}
				exploredStates[level.objects] = [parentState, actions[i]];
				if (winning || hasUsedCheckpoint) {
					muted = false;
					solving = false;
					winning = false;
					hasUsedCheckpoint = false;
					var solution = MakeSolution(level.objects);
					var chunks = chunkString(solution, 5).join(" ");
					var totalTime = (performance.now() - startTime) / 1000;
					consolePrint("solution found: (" + solution.length + " steps, " + size + " positions explored in " + totalTime + " seconds)");
					console.log("solution found:\n" + chunks);
					// solvingProgress.innerHTML = "";
					deltatime = oldDT;
					playSound(13219900);
					DoRestart();
					redraw();
					// cancelLink.hidden = true;
					consolePrint("<a href=\"javascript:ShowSolution('" + solution + "');\">" + chunks + "</a>");
					consolePrint("<br>");
					consolePrint("<a href=\"javascript:StopSolution();\"> stop showing solution </a>");
					consolePrint("<br>");
					ShowSolution(solution);
					return;
				}
				size++;
				queue.add([getScore(), level.objects.slice(0), numSteps + 1]);
			}
		}
	}
	muted = false;
	solving = false;
	DoRestart();
	consolePrint("no solution found (" + size + " positions explored)");
	console.log("no solution found");
	// solvingProgress.innerHTML = "";
	deltatime = oldDT;
	playSound(52291704);
	redraw();
	// cancelLink.hidden = true;
}

function MakeSolution(state) {
	var sol = "";
	while (true) {
		var p = exploredStates[state];
		if (p[1] == -1) {
			break;
		} else {
			sol = act2str[p[1]] + sol;
			state = p[0];
		}
	}
	return sol;
}

function StopSolution() {
	stopShowingSolution = true;
}

async function ShowSolution(sol) {
	if (levelEditorOpened) return;
	if (showingSolution) return;
	if (solving) return;
	if (textMode || state.levels.length === 0) return;

	showingSolution = true;
	stopShowingSolution = false;
	keybuffer = [];
	DoRestart();
	await timeout(repeatinterval);
	for (var i = 0; i < sol.length; i++) {
		if (stopShowingSolution) {
			stopShowingSolution = false;
			DoRestart();
			redraw();
			break;
		}
		var act = act2str.indexOf((sol[i]));
		var changedSomething = processInput(act, true);
		redraw();
		while (againing) {
			await timeout(againinterval);
			changedSomething = processInput(-1, true) || changedSomething;
			redraw();
		}
		await timeout(repeatinterval);
	}
	showingSolution = false;
}

function stopSolving() {
	abortSolver = true;
}

function chunkString(str, length) {
	return str.match(new RegExp('.{1,' + length + '}', 'g'));
}

function shuffleALittle(array) {
	randomIndex = 1 + Math.floor(Math.random() * (array.length - 1));
	temporaryValue = array[0];
	array[0] = array[randomIndex];
	array[randomIndex] = temporaryValue;
}

function distance(index1, index2) {
	return Math.abs(Math.floor(index1 / level.height) - Math.floor(index2 / level.height)) + Math.abs((index1 % level.height) - (index2 % level.height));
}

function precalcDistances() {
	distanceTable = [];
	for (var i = 0; i < level.n_tiles; i++) {
		ds = [];
		for (var j = 0; j < level.n_tiles; j++) {
			ds.push(distance(i, j));
		}
		distanceTable.push(ds);
	}
}

function getScore() {
	var score = 0.0;
	var maxDistance = level.width + level.height;
	if (state.winconditions.length > 0) {
		for (var wcIndex = 0; wcIndex < state.winconditions.length; wcIndex++) {
			var wincondition = state.winconditions[wcIndex];
			var filter1 = wincondition[1];
			var filter2 = wincondition[2];
			if (wincondition[0] == -1) {
				// "no" conditions
				for (var i = 0; i < level.n_tiles; i++) {
					var cell = level.getCellInto(i, _o10);
					if ((!filter1.bitsClearInArray(cell.data)) && (!filter2.bitsClearInArray(cell.data))) {
						score += 1.0; // penalization for each case
					}
				}
			} else {
				// "some" or "all" conditions
				for (var i = 0; i < level.n_tiles; i++) {
					if (!filter1.bitsClearInArray(level.getCellInto(i, _o10).data)) {
						var minDistance = maxDistance;
						for (var j = 0; j < level.n_tiles; j++) {
							if (!filter2.bitsClearInArray(level.getCellInto(j, _o10).data)) {
								var dist = distanceTable[i][j];
								if (dist < minDistance) {
									minDistance = dist;
								}
							}
						}
						score += minDistance;
					}
				}
			}
		}
	}
	// console.log(score);
	return score;
}

function getScoreNormalized() {
	var score = 0.0;
	var maxDistance = level.width + level.height;
	var normal_value = 0.0;
	if (state.winconditions.length > 0) {
		for (var wcIndex = 0; wcIndex < state.winconditions.length; wcIndex++) {
			var wincondition = state.winconditions[wcIndex];
			var filter1 = wincondition[1];
			var filter2 = wincondition[2];
			if (wincondition[0] == -1) {
				// "no" conditions
				for (var i = 0; i < level.n_tiles; i++) {
					var cell = level.getCellInto(i, _o10);
					if ((!filter1.bitsClearInArray(cell.data)) && (!filter2.bitsClearInArray(cell.data))) {
						score += 1.0; // penalization for each case
						normal_value += maxDistance;
					}
					
				}
			} else {
				// "some" or "all" conditions
				for (var i = 0; i < level.n_tiles; i++) {
					if (!filter1.bitsClearInArray(level.getCellInto(i, _o10).data)) {
						var minDistance = maxDistance;
						for (var j = 0; j < level.n_tiles; j++) {
							if (!filter2.bitsClearInArray(level.getCellInto(j, _o10).data)) {
								var dist = distanceTable[i][j];
								if (dist < minDistance) {
									minDistance = dist;
								}
							}
						}
						score += minDistance;
						normal_value += maxDistance;
					}
				}
			}
		}
	}
	// console.log(score);
	return score / normal_value;
}