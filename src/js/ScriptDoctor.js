(async () => {
    try {
      async function getCompletion(prompt) {
        const response = await fetch('/gen_game', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            // prompt: prompt
        }),
        });
  
        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }
  
        const data = await response.json();
        console.log(data.text);
        for (const line of data.text.split('\n')) {
          consolePrint(line);
        }
        // Clear the editor of the current code and replace it with the new code
        editor.setValue(data.code);
      }
  
      getCompletion('Explain JavaScript modules in simple terms.');
    } catch (error) {
      console.error('Error with API call:', error);
    }
})();