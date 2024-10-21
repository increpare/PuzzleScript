
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

async function main() {
  consoleText = '';
  nGenAttempts = 0;
  code = '';
  compilationSuccess = false;
  solvable = false;
  solverText = '';
  while (nGenAttempts == 0 | !compilationSuccess | !solvable) {
    if (errorCount > 0) {
      compilationSuccess = false;
      solvable = false;
      console.error(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
    }

    const response = await fetch('/gen_game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
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

    if (errorCount == 0) {
      compilationSuccess = true;
      solverText = '';
      solvable = true;
      console.log('No compilation errors. Performing playtest.');
      for (level_i in state.levels) {
        console.log('Levels:', state.levels);
        // Check if type `Level` or dict
        if (!state.levels[level_i].hasOwnProperty('height')) {
          console.log(`Skipping level ${level_i} as it does not appear to be a map (just a message?): ${state.levels[level_i]}.`);
          continue;
        }
        sol = await solveLevel(level_i);
        console.log('Solution:', sol);
        if (sol.length > 0) {
          console.log('Level is solvable.');
        } else {
          console.log(`Level ${level_i} is not solvable.`);
          solvable = false;
          solverText = `Level ${level_i} is not solvable. Please repair it.`
          break
        }
      }
    }

    nGenAttempts++;
  }
  console.log('No errors!');
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
        return new_action_seq;
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
  return [];
}

main();
// playTest();