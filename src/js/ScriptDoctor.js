
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
  while (nGenAttempts == 0 | errorCount > 0) {
    if (nGenAttempts > 0) {
      console.error(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
    }

    const response = await fetch('/gen_game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        code: code,
        console_text: consoleText,
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
    nGenAttempts++;
  }
  console.log('No errors!');
  // Playtest
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function playTest() {
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], editor.getValue());
  console.log('Playtesting...');
  // Load the first level
  sol = solveLevel(0);
  console.log('Solution:', sol);
}


function solveLevel(level) {
  // Load the level
  compile(['loadLevel', level]);
  console.log('Solving level', level);
  frontier = [backupLevel()];
  action_seqs = [[]];
  frontier_set = new Set(frontier);
  while (frontier.length > 0) {
    const level = frontier.pop(0);
    console.log('Level:', level);
    action_seq = action_seqs.pop(0);
    frontier_set.delete(level);
    for (const move of Array(5).keys()) {
      restoreLevel(level);
      // Make a copy of the action sequence
      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      changed = processInput(move);
      console.log('Frontier size:', frontier.length);
      if (winning) {
        console.log('Winning!');
        return action_seq;
      }
      else if (changed) {
        if (!frontier_set.has(backupLevel())) {
          frontier.push(backupLevel());
          action_seqs.push(new_action_seq);
          frontier_set.add(backupLevel());
        }
      }
    }
  }
}

// main();
playTest();