// const { Deque } = import('./collections'); // Use a deque for efficient pop/push

function getConsoleText() {
  // This probably exists somewhere else already?
  var consoleOut = document.getElementById('consoletextarea');

  // Initialize an empty array to store the extracted text
  var textContentArray = [];

  // Iterate over all child divs inside the consoletextarea
  consoleOut.querySelectorAll('div').forEach(function(div) {
      // Push the plain text content of each div into the array
      textContentArray.push(div.textContent.trim());
  });

  // Join the array elements with line breaks (or other delimiter)
  var plainTextOutput = textContentArray.join('\n');

  return plainTextOutput
}

class GameIndividual {
  constructor(code, minCode, fitness, compiledIters, solvedIters, anySolvedIters, skipped) {
    this.code = code;
    this.minCode = minCode;
    this.fitness = fitness;
    this.compiledIters = compiledIters;
    this.solvedIters = solvedIters;
    this.anySolvedIters = anySolvedIters;
    this.skipped = skipped;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function playTest() {
  // const game = 'sokoban_match3';
  const game = 'sokoban_basic';
  const n_level = 0;

  response = await fetch('/load_game_from_file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      'game': game,
    }),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  code = await response.text();
  loadFile(code);
  console.log(code)

  editor.setValue(code);
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], code);

  console.log('Playtesting...');
  compile(['loadLevel', n_level], editor.getValue());
  console.log('Solving level:', n_level, ' with A*');
  var [sol_a, n_search_iters_a] = await solveLevelAStar(level_i=n_level);


  editor.setValue(code);
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  console.log('Solving level:', n_level, ' with BFS');
  [sol_a, n_search_iters_a] = await solveLevelBFS(n_level);
  // const [sol, n_search_iters] = await solveLevelBFS(n_level);
  // gameToLoad = '/demo/sokoban_match3.txt';
  // gameToLoad = '/misc/3d_sokoban.txt';
  // sol = await solveLevel(0);

  // Load the the text file demo/sokoban_match3.txt
  // tryLoadFile('sokoban_match3');
  // var client = new XMLHttpRequest();
  // client.open('GET', gameToLoad);
  // client.onreadystatechange = async function() {
  //   console.log('Ready state:', client.readyState);
  //   console.log('Response', client.responseText);
  //   editor.setValue(client.responseText);
  //   sol = await solveLevel(0);
  //   console.log('Solution:', sol);
  // }
  // await client.send();
  // console.log('Loaded level:', editor.getValue());
}


function serialize(val) {
  return JSON.stringify(val);
}

class Queue {
  constructor() {
    this.inStack = [];
    this.outStack = [];
  }

  enqueue(value) {
    this.inStack.push(value);
  }

  dequeue() {
    if (this.outStack.length === 0) {
      while (this.inStack.length > 0) {
        this.outStack.push(this.inStack.pop());
      }
    }
    return this.outStack.pop();
  }

  isEmpty() {
    return this.inStack.length === 0 && this.outStack.length === 0;
  }

  size() {
    return this.inStack.length + this.outStack.length;
  }
}

function byScoreAndLength2(a, b) {
	// if (a[2] != b[2]) {
	// 	return a[2] < b[2];
	// } else {
	// 	return a[0] < b[0];
	// }
	
	if (a[0] != b[0]) {
		return a[0] < b[0];
	} else {
		return a[2].length < b[2].length;
	}
}


function hashStateObjects(state) {
  return JSON.stringify(state).split('').reduce((hash, char) => {
    return (hash * 31 + char.charCodeAt(0)) % 1_000_000_003; // Simple hash
  }, 0);
}


async function solveLevelBFS(level) {
  function hashState(levelMap) {
    return JSON.stringify(levelMap).split('').reduce((hash, char) => {
      return (hash * 31 + char.charCodeAt(0)) % 1_000_003; // Simple hash
    }, 0);
  }

  // Load the level
  compile(['loadLevel', level], editor.getValue());
  init_level = backupLevel();
  init_level_map = init_level['dat'];

  // frontier = [init_level];
  // action_seqs = [[]];
  // frontier = new Queue();
  // action_seqs = new Queue();

  frontier = new Queue();

  frontier.enqueue([init_level, []]);
  // action_seqs.enqueue([]);

  sol = [];
  console.log(sol.length);
  visited = new Set([hashState(init_level_map)]);
  i = 0;
  start_time = Date.now();
  console.log(frontier.size())
  while (frontier.size() > 0) {
    backups = [];

    // const level = frontier.shift();
    // const action_seq = action_seqs.shift();
    const [level, action_seq] = frontier.dequeue();
    // const action_seq = action_seqs.dequeue();

    if (!action_seq) {
      console.log(`Action sequence is empty. Length of frontier: ${frontier.size()}`);
    }
    for (const move of Array(5).keys()) {
      if (i > 1_000_000) {
        console.log('Exceeded 1M iterations. Exiting.');
        return [-1, i];
      }
      restoreLevel(level);

      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      try {
        changed = processInputSearch(move);
      } catch (e) {
        console.log('Error while processing input:', e);
        return [-2, i];
      }
      if (winning) {
        console.log(`Winning! Solution:, ${new_action_seq}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [new_action_seq, i];
      }
      else if (changed) {
        new_level = backupLevel();
        new_level_map = new_level['dat'];
        const newHash = hashState(new_level_map);
        if (!visited.has(newHash)) {
          
          // UNCOMMENT THESE LINES FOR VISUAL DEBUGGING
          // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
          // redraw();

          frontier.enqueue([new_level, new_action_seq]);
          // frontier.enqueue(new_level);
          if (!new_action_seq) {
            console.log(`New action sequence is undefined when pushing.`);
          }
          // action_seqs.enqueue(new_action_seq);
          visited.add(newHash);
        } 
      }
    }
    if (i % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Size of frontier: ${frontier.size}`);
      console.log(`Visited states: ${visited.size}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    i++;
  }
  return [sol, i];
}

class MCTSNode{
  constructor(action, parent, max_children) {
    this.parent = parent;
    this.action = action;
    this.children = [];
    for(let i=0; i<max_children; i++){
      this.children.push(null);
    }
    this.visits = 0;
    this.score = 0;
  }

  ucb_score(c) {
    if(this.parent == null){
      return this.score / this.visits;
    }
    return this.score / this.visits + c * Math.sqrt(Math.log(this.parent.visits) / this.visits);
  }

  select(c){
    if(!this.is_fully_expanded()){
      return null;
    }
    let index = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].ucb_score(c) > this.children[index].ucb_score(c)){
        index = i;
      }
    }
    return this.children[index];
  }

  is_fully_expanded(){
    for(let child of this.children){
      if(child == null){
        return false;
      }
    }
    return true;
  }

  expand(){
    if(this.is_fully_expanded()){
      return null;
    }
    for(let i=0; i<this.children.length; i++){
      if(this.children[i] == null){
        let changed = processInputSearch(i);
        let level = this.level;
        if(changed){
          level = backupLevel();
        }
        this.children[i] = new MCTSNode(i, this, this.children.length)
        return this.children[i];
      }
    }
    return null;
  }

  backup(score){
    this.score += score;
    this.visits += 1;
    if(this.parent != null){
      this.parent.backup(score);
    }
  }

  simulate(max_length, score_fn, win_bonus){
    let changes = 0;
    for(let i=0; i<max_length; i++){
      let changed = processInputSearch(Math.min(5, Math.floor(Math.random() * 6)));
      if(changed){
        changes += 1;
      }
      if(winning){
        return win_bonus;
      }
    }
    if(score_fn){
      return (score_fn() + 0.01 * changes / max_length) / win_bonus;
    }
    return (changes / max_length) / win_bonus;
  }

  get_actions(){
    let sol = [];
    let current = this;
    while(current.parent != null){
      sol.push(current.action);
      current = current.parent;
    }
    return sol.reverse();
  }

  get_most_visited_action(){
    let max_action = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].visits > this.children[max_action].visits){
        max_action = i;
      }
    }
    return max_action;
  }

  get_best_action(){
    let max_action = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].score / this.children[i].visited > this.children[max_action].score / this.children[max_action].visited){
        max_action = i;
      }
    }
    return max_action;
  }
}

// level: is the starting level
// max_sim_length: maximum number of random simulation before stopping and backpropagate
// score_fn: if you want to use heuristic function which is advisable and make sure the values are always between 0 and 1
// explore_deadends: if you want to explore deadends by default, the search don't continue in deadends
// deadend_bonus: bonus when you find a deadend node (usually negative number to avoid)
// win_bonus: bonus when you find a winning node
// c: is the MCTS constant that balance between exploitation and exploration
// max_iterations: max number of iterations before you consider the solution is not available
async function solveLevelMCTS(level, options = {}) {
  // Load the level
  if(options == null){
    options = {};
  }
  let defaultOptions = {
    "max_sim_length": 1000,
    "score_fn": null, 
    "explore_deadends": false, 
    "deadend_bonus": -1, 
    "win_bonus": 100, 
    "c": Math.sqrt(2), 
    "max_iterations": -1
  };
  for(let key in defaultOptions){
    if(!options.hasOwnProperty(key)){
      options[key] = defaultOptions[key];
    }
  }
  compile(['loadLevel', level], editor.getValue());
  init_level = backupLevel();
  init_level_map = init_level['dat'];
  let rootNode = new MCTSNode(-1, null, 5);
  let i = 0;
  let deadend_nodes = 1;
  let start_time = Date.now();
  while(options.max_iterations <= 0 || (options.max_iterations > 0 && i < options.max_iterations)){
    // start from th root
    currentNode = rootNode;
    restoreLevel(init_level);
    let changed = true;
    // selecting next node
    while(currentNode.is_fully_expanded()){
      currentNode = currentNode.select(options.c);
      changed = processInputSearch(currentNode.action);
      if(winning){
        let sol = current.get_actions();
        console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [sol, i];
      }
      if(!options.explore_deadends && !changed){
        break;
      }
    }

    // if node is deadend, punish it
    if(!options.explore_deadends && !changed){
      currentNode.backup(options.deadend_bonus);
      deadend_nodes += 1;
    }
    //otherwise expand
    else{
      currentNode = currentNode.expand();
      changed = processInputSearch(currentNode.action);
      if(winning){
        let sol = current.get_actions();
        console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [sol, i];
      }
      // if node is deadend, punish it
      if(!options.explore_deadends && !changed){
        currentNode.backup(options.deadend_bonus);
        deadend_nodes += 1;
        
      }
      //otherwise simulate then backup
      else{
        let value = currentNode.simulate(options.max_sim_length, options.score_fn, options.win_bonus);
        currentNode.backup(value);
      }
    }
    // print progress
    if (i % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Visited Deadends: ${deadend_nodes}`);
      // console.log(`Visited states: ${visited.size}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    i+= 1;
  }
  let actions = [];
  currentNode = rootNode;
  while(currentNode.is_fully_expanded()){
    let action = currentNode.get_most_visited_action();
    actions.push(action);
    currentNode = currentNode.children[action];
  }
  return [actions, options.max_iterations];
}

async function testMCTS() {
  console.log('Testing MCTS...');
  const n_level = 0;
  compile(['loadLevel', n_level], editor.getValue());
  console.log('Solving level:', n_level, ' with MCTS');
  let heuristic = getScore;
  if(heuristic != null){
    precalcDistances();
  }
  var [sol_a, n_search_iters_a] = await solveLevelMCTS(level_i=n_level, {"score_fn": heuristic, "max_iterations": 1000000});
  console.log('Solution:', sol_a);
}


async function solveLevelAStar(captureStates=false, gameHash=0, levelI=0, maxIters=1_000_000) {
	// if (levelEditorOpened) return;
	// if (showingSolution) return;
	// if (solving) return;
	// if (textMode || state.levels.length === 0) return;

	precalcDistances();
	abortSolver = false;
	muted = true;
	solving = true;
	// restartTarget = backupLevel();
	DoRestartSearch();
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
  var totalIters = 0
	var iters = 0;
	var size = 1;

	var startTime = performance.now();

  if (captureStates) {
    const canvas = document.getElementById('gameCanvas');
    const imageData = canvas.toDataURL('image/png');
    await fetch('/save_init_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_hash: gameHash,
        game_level: levelI,
        state_hash: hashStateObjects(level.objects),
        im_data: imageData,
      })
    });
  }

	while (!queue.isEmpty() && totalIters < maxIters) {
		if (abortSolver) {
			consolePrint("solver aborted");
			// cancelLink.hidden = true;
			break;
		}
    if (totalIters > maxIters) {
      console.log('Exceeded max iterations. Exiting.');
      break;
    }
		iters++;
		if (iters > 500) {
			iters = 0;
			// console.log(size);
			// solvingProgress.innerHTML = "searched: " + size;
			// redraw();
			// await timeout(1);
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
        if (captureStates) {
          await processStateTransition(gameHash, parentState, level.objects, actions[i]);
          console.log(winning);
        }

        // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
        // redraw();

				exploredStates[level.objects] = [parentState, actions[i]];
				if (winning || hasUsedCheckpoint) {
          console.log('Winning!');
					muted = false;
					solving = false;
					winning = false;
					hasUsedCheckpoint = false;
					var solution = MakeSolution(level.objects);
					var chunks = chunkString(solution, 5).join(" ");
					var totalTime = (performance.now() - startTime) / 1000;
					consolePrint("solution found: (" + solution.length + " steps, " + size + " positions explored in " + totalTime + " seconds)");
					console.log("solution found:\n" + chunks + "\nin " + totalIters + " steps");
					// solvingProgress.innerHTML = "";
					deltatime = oldDT;
					playSound(13219900);
					DoRestartSearch();
					redraw();
					// cancelLink.hidden = true;
					// consolePrint("<a href=\"javascript:ShowSolution('" + solution + "');\">" + chunks + "</a>");
					// consolePrint("<br>");
					// consolePrint("<a href=\"javascript:StopSolution();\"> stop showing solution </a>");
					// consolePrint("<br>");
					// ShowSolution(solution);
					return [solution, totalIters];
				}
				size++;
				queue.add([getScore(), level.objects.slice(0), numSteps + 1]);
			}
		}
    totalIters++;
	}
	muted = false;
	solving = false;
	DoRestartSearch();
	consolePrint("no solution found (" + size + " positions explored)");
	console.log("no solution found");
	// solvingProgress.innerHTML = "";
	deltatime = oldDT;
	playSound(52291704);
	redraw();
	// cancelLink.hidden = true;
  return ['', totalIters];
}


async function captureGameState() {
  // Capture current game state as PNG
  const canvas = document.getElementById('gameCanvas');
  const imageData = canvas.toDataURL('image/png');
  return imageData;
}

async function processStateTransition(gameHash, parentState, childState, action) {
  // const img1 = await captureGameState(); 
  const hash1 = hashStateObjects(parentState);

  for (var k = 0, len2 = parentState.length; k < len2; k++) {
    level.objects[k] = parentState[k];
  }
  redraw(); 
  
  // Apply action and capture result
  processInput(action);
  while (againing) {
    processInput(-1);
  }
  
  const img2 = await captureGameState();
  const hash2 = hashStateObjects(childState);
  // const hash2 = state2

  // Save transition
  await fetch('/save_transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_hash: gameHash,
      game_level: curlevel,
      state1_hash: hash1,
      state2_hash: hash2, 
      state2_img: img2,
      action: action
    })
  });
}


async function genGame(genMode, parents, saveDir, expSeed, fewshot, cot,
  /* This funciton will recursively call itself to iterate on broken (uncompilable or unsolvable (or too simply solvable)) games. */
  fromIdea=false, idea='', fromPlan=false, maxGenAttempts=10) {
  consoleText = '';
  larkError = '';
  nGenAttempts = 0;
  code = '';
  compilationSuccess = false;
  solvable = false;
  solverText = '';
  compiledIters = [];
  solvedIters = [];
  anySolvedIters = [];

  bestIndividual = new GameIndividual('', null, -Infinity, [], [], true);
  while (nGenAttempts < maxGenAttempts & (nGenAttempts == 0 | !compilationSuccess | !solvable)) {
    console.log(`Game ${saveDir}, attempt ${nGenAttempts}.`);

    var response;

    if (fromPlan & nGenAttempts == 0) {
      response = await fetch('/gen_game_from_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          seed: expSeed,
          save_dir: saveDir,
          game_idea: idea,
          n_iter: nGenAttempts,
        }),
      });
    } else {
      // Get our GPT completion from python
      response = await fetch('/gen_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          seed: expSeed,
          fewshot: fewshot,
          cot: cot,
          save_dir: saveDir,
          gen_mode: genMode,
          parents: parents,
          code: code,
          from_idea: fromIdea,
          game_idea: idea,
          lark_error: larkError,
          console_text: consoleText,
          solver_text: solverText,
          compilation_success: compilationSuccess,
          n_iter: nGenAttempts,
        }),
      });
    }
  
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
  
    const data = await response.json();
    // for (const line of data.text.split('\n')) {
    //   consolePrint(line);
    // }
    code = data.code;
    minCode = null;
    // if min_code is not None, then use this
    if (data.min_code) {
      minCode = data.min_code;
    }
    sols = data.sols;
    larkError = data.lark_error
    if (data.skip) {
      return new GameIndividual(code, minCode, -1, [], [], true);
    }
    errorLoadingLevel = false;
    try {
      codeToCompile = minCode ? minCode : code;
      editor.setValue(codeToCompile);
      editor.clearHistory();
      clearConsole();
      setEditorClean();
      unloadGame();
    } catch (e) {
      console.log('Error while loading code:', e);
      errorLoadingLevel = true;
      consoleText = `Error while loading code into editor: ${e}.`;
      errorCount = 10;
    }
    if (!errorLoadingLevel) {
      try {
        compile(['restart'], codeToCompile);
      } catch (e) {
        console.log('Error while compiling code:', e);
      }
      consoleText = getConsoleText();
    }

    if (errorCount > 0) {
      compilationSuccess = false;
      solvable = false;
      solverText = '';
      // console.log(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
      fitness = -errorCount;
      dataURLs = [];
    } else {
      compiledIters.push(nGenAttempts);
      compilationSuccess = true;
      solverText = '';
      solvable = true;
      dataURLs = [];
      var anySolvable = false;
      var sol;
      var n_search_iters;
      // console.log('No compilation errors. Performing playtest.');
      for (level_i in state.levels) {
        // console.log('Levels:', state.levels);
        // Check if type `Level` or dict
        if (!state.levels[level_i].hasOwnProperty('height')) {
          // console.log(`Skipping level ${level_i} as it does not appear to be a map (just a message?): ${state.levels[level_i]}.`);
          continue;
        }
        // try {
          // Check if level_i is in sols
        if (sols.hasOwnProperty(level_i)) {
          // console.log('Using cached solution.');
          [sol, n_search_iters] = sols[level_i];
        } else {
          clearConsole();
          console.log(`Solving level ${level_i}...`);
          [sol, n_search_iters] = await solveLevelBFS(level_i);
          if (sol.length > 0) {
            console.log(`Solution for level ${level_i}:`, sol);
            console.log(`Saving gif for level ${level_i}.`);
            curlevel = level_i;
            compile(['loadLevel', level_i], editor.getValue());
            inputHistory = sol;
            const [ data_url, filename ] = makeGIFDoctor();
            dataURLs.push([data_url, level_i]);
          }
        }
        // } catch (e) {
        //   console.log('Error while solving level:', e);
        //   sol = [];
        //   n_search_iters = -1;
        //   solverText += ` Level ${level_i} resulted in error: ${e}. Please repair it.`;
        // }
        if (!sol) {
          console.log(`sol undefined`);
        }
        sols[level_i] = [sol, n_search_iters];
        fitness = n_search_iters
        // console.log('Solution:', sol);
        // check if sol is undefined
        if (sol.length > 0) {
          // console.log('Level is solvable.');
          // solverText += `Found solution for level ${level_i} in ${n_search_iters} iterations: ${sol}.\n`
          solverText += `Found solution for level ${level_i} in ${n_search_iters} iterations. Solution is ${sol.length} moves long.\n`
          if (sol.length > 1) {
            anySolvable = true;
          }
          if (sol.length < 10) {
            solverText += `Solution is very short. Please make it a bit more complex.\n`
            solvable = false;
          }
        } else if (sol == -1) {
          solvable = false;
          solverText += `Hit maximum search depth of ${i} while attempting to solve ${level_i}. Are you sure it's solvable? If so, please make it a bit simpler.\n`
        } else if (sol == -2) {
          solvable = false;
          consoleText = getConsoleText();
          solverText += `Error while solving level ${level_i}. Please repair it.\nThe PuzzleScript console output was:\n${consoleText}\n`
        } else {
          // console.log(`Level ${level_i} is not solvable.`);
          solvable = false;
          solverText += ` Level ${level_i} is not solvable. Please repair it.\n`
        }
      }
      if (solvable) {
        // If all levels are solvable
        solvedIters.push(nGenAttempts)
      }
      if (anySolvable) {
        anySolvedIters.push(nGenAttempts)
      }
    }
    response = await fetch('/log_gen_results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        save_dir: saveDir,
        sols: sols,
        n_iter: nGenAttempts,
        gif_urls: dataURLs,
        console_text: consoleText,
        solver_text: solverText,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    nGenAttempts++;
    individual = new GameIndividual(code, minCode, fitness, compiledIters, solvedIters, anySolvedIters, false);
    bestIndividual = bestIndividual.fitness < individual.fitness ? individual : bestIndividual;

  }
  return bestIndividual;
}


const popSize = 3;
const nGens = 20;

async function evolve() {
  // Create an initial population of 10 games
  pop = [];
  gen = 0
  for (indIdx = 0; indIdx < (popSize*2); indIdx++) {
    saveDir = `evo-${expSeed}/gen${gen}/game${indIdx}`;
    game_i = await genGame('init', [], saveDir, expSeed, fewshot=true, cot=true, fromIdea=false, idea='');
    pop.push(game_i);
  }
  for (gen = 1; gen < nGens; gen++) {
    // Sort the population by fitness, in descending order
    pop = pop.sort((a, b) => b.fitness - a.fitness);
    // Print list of fitnesses
    popFits = pop.map(game => game.fitness);
    meanPopFit = popFits.reduce((acc, fit) => acc + fit, 0) / popFits.length;
    console.log(`Generation ${gen}. Fitnesses: ${popFits}`);
    console.log(`Generation ${gen}. Mean fitness: ${meanPopFit}`);
    // Select the top half of the population as parents
    ancestors = pop.slice(0, popSize);
    // Get mean fitness of elites
    eliteFits =  ancestors.map(game => game.fitness);
    meanEliteFit = eliteFits.reduce((acc, fit) => acc + fit, 0) / eliteFits.length;
    console.log(`Generation ${gen}. Elite fitnesses: ${eliteFits}`);
    console.log(`Generation ${gen}. Mean elite fitness: ${meanEliteFit}`);
    // Generate the next generation
    newPop = [];
    for (indIdx = 0; indIdx < popSize; indIdx++) {
      doCrossOver = Math.random() < 0.5;
      if (doCrossOver) {
        genMode = 'crossover';
        // Get two random games from list without replacement
        parent1 = ancestors[Math.floor(Math.random() * popSize)];
        // Create copy of array without parent1
        remainingAncestors = ancestors.filter(parent => parent != parent1);
        parent2 = remainingAncestors[Math.floor(Math.random() * (popSize - 1))];
        parents = [parent1, parent2];
      } else {
        genMode = 'mutate';
        parents = [ancestors[Math.floor(Math.random() * popSize)]];
      }
      // console.log(`Parents: ${parents}. genMode: ${genMode}`);
      saveDir = `evo-${expSeed}/gen${gen}/game${indIdx}`;
      newPop.push(await genGame('mutate', parents, saveDir, expSeed, fewshot=fewshot, cot=cot));
    }
    pop = pop.concat(newPop);
  }
}

async function saveStats(saveDir, results) {
  const response = await fetch('/save_sweep_stats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      save_dir: saveDir,
      results: results,
    }),
  });
}

async function sweepGeneral() {
  isDone = false;
  while (!isDone) {
    response = await fetch('/get_sweep_args', {
      method: 'GET',
    });
    args = await response.json();
    gameInd = await genGame('init', [], args.gameStr,
      args.gameIdx, args.fewshot, args.cot, args.fromIdea, args.gameIdea, args.fromPlan);
    isDone = args.done;
  }
}

async function sweep() {
  saveDir = `sweep-${expSeed}`
  results = {};
  for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
    for (var fewshot_i = 0; fewshot_i < 2; fewshot_i++) {
      for (var cot_i = 0; cot_i < 2; cot_i++) {
        expName = `fewshot-${fewshot_i}_cot-${cot_i}`;
        if (!results.hasOwnProperty(expName)) {
          results[expName] = [];
        }
          gameStr = `${saveDir}/${expName}/game-${gameIdx}`;
          cot = cot_i == 1
          fewshot = fewshot_i == 1
          console.log(`Generating game ${gameStr}`);
          gameInd = await genGame('init', [], gameStr,
            gameIdx, fewshot, cot, fromIdea=false, idea='');
          results[expName].push(gameInd);
        }
      }
  }
  saveStats(saveDir, results);
}

brainstormSeed = 0;

async function fromIdeaSweep() {
  // Open the ideas json
  const response = await fetch('/load_ideas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ brainstorm_seed: brainstormSeed }),
  });
  ideas = await response.json()
  results = {};
  fewshot_i = 1;
  fromIdea_i = 1;
  for (var cot_i = 0; cot_i < 2; cot_i++) {
    hypStr = `fromIdea-${fromIdea_i}_fewshot-${fewshot_i}_cot-${cot_i}`;
    results[hypStr] = [];
    for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
      saveDir = `sweep-${expSeed}`
      gameStr = `${saveDir}/${hypStr}/game-${gameIdx}`;
      fewshot = fewshot_i == 1
      cot = cot_i == 1
      fromIdea = fromIdea_i == 1
      console.log(`Generating game ${gameStr}`);
      ideaIdx = gameIdx % ideas.length;
      idea = ideas[ideaIdx];
      gameInd = await genGame('init', [], gameStr,
        gameIdx, fewshot, cot, fromIdea, idea);
      results[hypStr].push(gameInd);
    }
  }
  saveStats(saveDir + '/fromIdea', results);
}

async function fromPlanSweep() {
  // Open the ideas json
  const response = await fetch('/load_ideas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ brainstorm_seed: brainstormSeed }),
  });
  ideas = await response.json()
  results = {};
  cot_i = 1;
  fewshot_i = 1;
  fromIdea_i = 1;
  fromPlan_i = 1;
  hypStr = `fromPlan-${fromPlan_i}`;
  results[hypStr] = [];
  for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
    saveDir = `sweep-${expSeed}`
    gameStr = `${saveDir}/${hypStr}/game-${gameIdx}`;
    fewshot = fewshot_i == 1
    cot = cot_i == 1
    fromPlan = fromPlan_i == 1
    fromIdea = fromIdea_i == 1
    console.log(`Generating game ${gameStr}`);
    ideaIdx = gameIdx % ideas.length;
    idea = ideas[ideaIdx];
    gameInd = await genGame('init', [], gameStr,
      gameIdx, fewshot, cot, fromIdea, idea, fromPlan);
    results[hypStr].push(gameInd);
  }
  saveStats(saveDir + '/fromPlan', results);
}

async function collectGameData(gamePath) {
  // Load game
  const response = await fetch('/load_game_from_file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game: gamePath })
  });
  
  const code = await response.text();
  
  // Initialize game
  editor.setValue(code);
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], code);

  // Process each level
  for (let level = 0; level < state.levels.length; level++) {
    if (!state.levels[level].hasOwnProperty('height')) {
      continue;
    }
    
    console.log(`Processing level ${level}`);
    compile(['loadLevel', level], code);
    await solveLevelAStar(captureStates=true, gameHash=gamePath, level_i=level, maxIters=1_000);
    console.log(`Finished processing level ${level}`);
  }
}

async function processAllGames() {
  const response = await fetch('/list_scraped_games');
  const games = await response.json();

  // Shuffle the games
  games.sort(() => Math.random() - 0.5);
  
  for (const game of games) {
    console.log(`Processing game: ${game}`);
    await collectGameData(game);
  }
}
var experimentDropdown = document.getElementById("experimentDropdown");
experimentDropdown.addEventListener("change", experimentDropdownChange, false);

var generateClickLink = document.getElementById("generateClickLink");
generateClickLink.addEventListener("click", generateClick, false);

var MCTSClickLink = document.getElementById("MCTSClickLink");
MCTSClickLink.addEventListener("click", testMCTS, false);

var solveClickLink = document.getElementById("solveClickLink");
solveClickLink.addEventListener("click", playTest, false);

var expFn = evolve;

function experimentDropdownChange() {
  console.log('Experiment changed');
  var experiment = experimentDropdown.value;
  if (experiment == 'evolve') {
    expFn = evolve;
  } else if (experiment == 'fewshot_cot') {
    expFn = sweep;
  } else if (experiment == 'from_idea') {
    expFn = fromIdeaSweep;
  } else if (experiment == 'from_plan') {
    expFn = fromPlanSweep;
  }
  else {
    console.log('Unknown experiment:', experiment);
  }
}

function generateClick() {
  console.log('Generate clicked');
  expFn();
}

 const expSeed = 21;

// sweepGeneral();
// sweep();
// fromIdeaSweep();
// fromPlanSweep();
// playTest();
// evolve();
// processAllGames();

// genGame('init', [], 'test_99', 99, fewshot=true, cot=true, maxGenAttempts=20);