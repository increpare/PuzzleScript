
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
  constructor(code, fitness) {
    this.code = code;
    this.fitness = fitness;
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
  // sol = solveLevel(0);

  // Load the the text file demo/sokoban_match3.txt
  // tryLoadFile('sokoban_match3');
  var client = new XMLHttpRequest();
  client.open('GET', '/demo/sokoban_match3.txt');
  client.onreadystatechange = function() {
    console.log('Ready state:', client.readyState);
    console.log('Response', client.responseText);
    editor.setValue(client.responseText);
    sol = solveLevel(0);
  }
  client.send();
  // console.log('Loaded level:', editor.getValue());
  console.log('Solution:', sol);
}


function serialize(val) {
  return JSON.stringify(val);
}


async function solveLevel(level) {
  // Load the level
  compile(['loadLevel', level], editor.getValue());
  console.log('Solving level', level);
  init_level = backupLevel();
  init_level_map = init_level['dat'];
  frontier = [init_level];
  action_seqs = [[]];
  visited = new Set([serialize(init_level_map)]);
  i = 0;
  start_time = Date.now();
  while (frontier.length > 0) {
    const level = frontier.pop(0);
    action_seq = action_seqs.pop(0);
    for (const move of Array(5).keys()) {
      restoreLevel(level);
      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      changed = processInput(move);
      if (winning) {
        console.log('Winning!');
        return new_action_seq, i;
      }
      else if (changed) {
        new_level = backupLevel();
        new_level_map = new_level['dat'];
        if (!visited.has(serialize(new_level_map))) {
          
          // UNCOMMENT THESE LINES FOR VISUAL DEBUGGING
          // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
          // redraw();

          frontier.push(new_level);
          action_seqs.push(new_action_seq);
          visited.add(serialize(new_level_map));
        } 
      }
    }
    if (i % 1000 == 0) {
      console.log('Iteration:', i);
      console.log('FPS:', i / (Date.now() - start_time) * 1000);
    }
    i++;
  }
  let sol = [];
  return sol, i;
}


async function genGame(genMode, parents, saveDir, seed, fewshot, cot) {
  consoleText = '';
  nGenAttempts = 0;
  code = '';
  compilationSuccess = false;
  solvable = false;
  solverText = '';
  while (nGenAttempts < maxGenAttempts & (nGenAttempts == 0 | !compilationSuccess | !solvable)) {

    const response = await fetch('/gen_game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        seed: seed,
        fewshot: fewshot,
        cot: cot,
        save_dir: saveDir,
        gen_mode: genMode,
        parents: parents,
        code: code,
        console_text: consoleText,
        solver_text: solverText,
        compilation_success: compilationSuccess,
        n_iter: nGenAttempts,
    }),
    });
  
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
  
    const data = await response.json();
    for (const line of data.text.split('\n')) {
      consolePrint(line);
    }
    code = data.code;
    editor.setValue(code);
    editor.clearHistory();
    clearConsole();
    setEditorClean();
    unloadGame();
    compile(['restart'], code);
    consoleText = getConsoleText();

    if (errorCount > 0) {
      compilationSuccess = false;
      solvable = false;
        solverText = '';
      console.error(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
      fitness = -errorCount;
    } else {
      compilationSuccess = true;
      solverText = '';
      solvable = true;
      var sol;
      var n_search_iters;
      console.log('No compilation errors. Performing playtest.');
      for (level_i in state.levels) {
        console.log('Levels:', state.levels);
        // Check if type `Level` or dict
        if (!state.levels[level_i].hasOwnProperty('height')) {
          console.log(`Skipping level ${level_i} as it does not appear to be a map (just a message?): ${state.levels[level_i]}.`);
          continue;
        }
        sol, n_search_iters = await solveLevel(level_i);
        fitness = Math.max(fitness, n_search_iters)
        console.log('Solution:', sol);
        // check if sol is undefined
        if (sol) {
          console.log('Level is solvable.');
          solverText += `Found solution for ${level_i} in ${n_search_iters} iterations.`
        } else {
          console.log(`Level ${level_i} is not solvable.`);
          solvable = false;
          solverText += `Level ${level_i} is not solvable. Please repair it.`
        }
      }
    }

    nGenAttempts++;
  }
  return new GameIndividual(code, fitness);
}


const maxGenAttempts = 10;
const popSize = 3;
const nGens = 10;

async function evolve() {
  // Create an initial population of 10 games
  pop = [];
  for (i = 0; i < popSize*2; i++) {
    saveDir = `gen0/game${i}`;
    game_i = await genGame('init', [], saveDir, seed=seed, fewshot=fewshot, cot=cot);
    pop.push(game_i);
  }
  for (gen = 0; gen < nGens; gen++) {
    // Sort the population by fitness
    pop.sort((a, b) => a.fitness - b.fitness);
    // Select the top half of the population as parents
    parents = pop.slice(0, popSize);
    // Generate the next generation
    newPop = [];
    for (i = 0; i < popSize * 2; i++) {
      doCrossOver = Math.random() < 0.5;
      if (doCrossOver) {
        genMode = 'crossover';
        // Get two random games from list without replacement
        parent1 = parents[Math.floor(Math.random() * popSize)];
        // Create copy of array without parent1
        remainingParents = parents.filter(parent => parent != parent1);
        parent2 = remainingParents[Math.floor(Math.random() * (popSize - 1))];
        parents = [parent1, parent2];
      } else {
        genMode = 'mutate';
        parents = [parents[Math.floor(Math.random() * popSize)]];
      }
      saveDir = `gen${gen}/game${i}`;
      newPop.push(genGame('mutate', parents, saveDir, seed=seed, fewshot=fewshot, cot=cot));
    }
  }
}

async function sweep() {
  for (seed = 0; seed < 10; seed++) {
    for (cot in [true, false]) {
      for (fewshot in [true, false]) {
        gameInd = await genGame('init', [], `sweep/fewshot-${fewshot}_cot-${cot}/game-${i}`,
          seed, fewshot, cot);
      }
    }
  }
}

sweep()
// evolve();
// playTest();