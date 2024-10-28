// const { Deque } = import('./collections'); // Use a deque for efficient pop/push

const expSeed = 12;

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
  constructor(code, fitness, compiledIters, solvedIters, skipped) {
    this.code = code;
    this.fitness = fitness;
    this.compiledIters = compiledIters;
    this.solvedIters = solvedIters;
    this.skipped = skipped;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function playTest() {
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], editor.getValue());
  console.log('Playtesting...');
  const [sol, n_search_iters] = await solveLevel(4);
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


async function solveLevel(level) {
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
  frontier = new Queue();
  action_seqs = new Queue();
  frontier.enqueue(init_level);
  action_seqs.enqueue([]);

  sol = [];
  console.log(sol.length);
  visited = new Set([hashState(init_level_map)]);
  i = 0;
  start_time = Date.now();
  while (frontier.size() > 0) {
    backups = [];

    // const level = frontier.shift();
    // const action_seq = action_seqs.shift();
    const level = frontier.dequeue();
    const action_seq = action_seqs.dequeue();

    if (!action_seq) {
      console.log(`Action sequence is empty. Length of frontier: ${frontier.length}`);
    }
    for (const move of Array(5).keys()) {
      if (i > 1_000_000) {
        console.log('Exceeded 1M iterations. Exiting.');
        return [-1, i];
      }
      restoreLevel(level);
      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      changed = processInputSearch(move);
      if (winning) {
        console.log(`Winning! Solution:, ${new_action_seq}`);
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

          frontier.enqueue(new_level);
          if (!new_action_seq) {
            console.log(`New action sequence is undefined when pushing.`);
          }
          action_seqs.enqueue(new_action_seq);
          visited.add(newHash);
        } 
      }
    }
    if (i % 10_000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Size of frontier: ${frontier.length}`);
      console.log(`Visited states: ${visited.size}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    i++;
  }
  return [sol, i];
}


async function genGame(genMode, parents, saveDir, expSeed, fewshot, cot,
    fromIdea=false, idea='', fromPlan=false, maxGenAttempts=10) {
  consoleText = '';
  nGenAttempts = 0;
  code = '';
  compilationSuccess = false;
  solvable = false;
  solverText = '';
  compiledIters = [];
  solvedIters = [];
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
    sols = data.sols;
    if (data.skip) {
      return new GameIndividual(code, -1, [], [], true);
    }
    errorLoadingLevel = false;
    try {
      editor.setValue(code);
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
        compile(['restart'], code);
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
        try {
          // Check if level_i is in sols
          if (sols.hasOwnProperty(level_i)) {
            // console.log('Using cached solution.');
            [sol, n_search_iters] = sols[level_i];
          } else {
            console.log(`Solving level ${level_i}...`);
            [sol, n_search_iters] = await solveLevel(level_i);
            console.log(`Solution for level ${level_i}:`, sol);
          }
        } catch (e) {
          console.log('Error while solving level:', e);
          sol = [];
          n_search_iters = -1;
          solverText += ` Level ${level_i} resulted in error: ${e}. Please repair it.`;
        }
        if (!sol) {
          console.log(`sol undefined`);
        }
        sols[level_i] = [sol, n_search_iters];
        fitness = n_search_iters
        // console.log('Solution:', sol);
        // check if sol is undefined
        if (sol.length > 0) {
          // console.log('Level is solvable.');
          solverText += `Found solution for level ${level_i} in ${n_search_iters} iterations: ${sol}. `
          anySolvable = true;
        } else if (sol == -1) {
          solvable = false;
          solverText += `Hit maximum search depth of ${i} while attempting to solve ${level_i}. Are you sure it's solvable? If so, please make it a bit simpler.`
        }
        else {
          // console.log(`Level ${level_i} is not solvable.`);
          solvable = false;
          solverText += ` Level ${level_i} is not solvable. Please repair it.`
        }
      }
      dataURLs = [];
      if (solvable) {
        solvedIters.push(nGenAttempts)
        // Make a gif of each solution
        for (let level_i in sols) {
          const [sol, n_search_iters] = sols[level_i];
          if (sol.length > 0) {
            console.log(`Saving gif for level ${level_i}.`);
            curlevel = level_i;
            compile(['loadLevel', level_i], editor.getValue());
            inputHistory = sol;
            const [ data_url, filename ] = makeGIFDoctor();
            dataURLs.push([data_url, level_i]);
          }
        }
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
  }
  return new GameIndividual(code, fitness, compiledIters, solvedIters, false);
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

async function sweep() {
  results = {};
    for (var fewshot_i = 0; fewshot_i < 2; fewshot_i++) {
      for (var cot_i = 0; cot_i < 2; cot_i++) {
        results[`fewshot-${fewshot_i}_cot-${cot_i}`] = [];
        for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
          saveDir = `sweep-${expSeed}`
          gameStr = `${saveDir}/fewshot-${fewshot_i}_cot-${cot_i}/game-${gameIdx}`;
          cot = cot_i == 1
          fewshot = fewshot_i == 1
          console.log(`Generating game ${gameStr}`);
          gameInd = await genGame('init', [], gameStr,
            gameIdx, fewshot, cot, fromIdea=false, idea='');
          results[`fewshot-${fewshot_i}_cot-${cot_i}`].push(gameInd);
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

var experimentDropdown = document.getElementById("experimentDropdown");
experimentDropdown.addEventListener("change", experimentDropdownChange, false);

var generateClickLink = document.getElementById("generateClickLink");
generateClickLink.addEventListener("click", generateClick, false);

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

// sweep();
// fromIdeaSweep();
fromPlanSweep();
// evolve();

// genGame('init', [], 'test_99', 99, fewshot=true, cot=true, maxGenAttempts=20);
// playTest();