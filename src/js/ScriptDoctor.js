
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

function playTest() {
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], editor.getValue());
  console.log('Playtesting...');
  // Load the first level
  compile(['loadLevel', 0]);
  // Move right x3 (to solve first level)
  processInput(3);
  processInput(3);
  processInput(3);
}

// main();
playTest();