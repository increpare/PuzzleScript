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

// async function playTest() {
//   editor.clearHistory();
//   clearConsole();
//   setEditorClean();
//   unloadGame();
//   compile(['restart'], editor.getValue());
//   console.log('Playtesting...');
//   // sol = await solveLevel(0);

//   // Load the the text file demo/sokoban_match3.txt
//   // tryLoadFile('sokoban_match3');
//   var client = new XMLHttpRequest();
//   client.open('GET', '/demo/sokoban_match3.txt');
//   client.onreadystatechange = function() {
//     console.log('Ready state:', client.readyState);
//     console.log('Response', client.responseText);
//     editor.setValue(client.responseText);
//     sol = await solveLevel(0);
//   }
//   client.send();
//   // console.log('Loaded level:', editor.getValue());
//   console.log('Solution:', sol);
// }


function serialize(val) {
  return JSON.stringify(val);
}


/* returns a bool indicating if anything changed */
function processInputSearch(dir,dontDoWin,dontModify) {
	againing = false;
  
	var bak = backupLevel();
	var inputindex=dir;

	var playerPositions=[];
    if (dir<=4) {//when is dir>4???

		if (verbose_logging) { 
			debugger_turnIndex++;
			addToDebugTimeline(level,-2);//pre-movement-applied debug state
		}

    	if (dir>=0) {
	        switch(dir){
	            case 0://up
	            {
	                dir=parseInt('00001', 2);;
	                break;
	            }
	            case 1://left
	            {
	                dir=parseInt('00100', 2);;
	                break;
	            }
	            case 2://down
	            {
	                dir=parseInt('00010', 2);;
	                break;
	            }
	            case 3://right
	            {
	                dir=parseInt('01000', 2);;
	                break;
	            }
	            case 4://action
	            {
	                dir=parseInt('10000', 2);;
	                break;
	            }
	        }
	        playerPositions = startMovement(dir);
		}
			
		
		if (verbose_logging) { 
			consolePrint('Applying rules');

			var inspect_ID = addToDebugTimeline(level,-1);
				
			 if (dir===-1) {
				 consolePrint(`Turn starts with no input.`,false,null,inspect_ID)
			 } else {
				//  consolePrint('=======================');
				consolePrint(`Turn starts with input of ${['up','left','down','right','action'][inputindex]}.`,false,null,inspect_ID);
			 }
		}

		
        var bannedGroup = [];

        level.commandQueue=[];
        level.commandQueueSourceRules=[];
        var startRuleGroupIndex=0;
        var rigidloop=false;
		const startState = {
			objects: new Int32Array(level.objects),
			movements: new Int32Array(level.movements),
			rigidGroupIndexMask: level.rigidGroupIndexMask.concat([]),
			rigidMovementAppliedMask: level.rigidMovementAppliedMask.concat([]),
			commandQueue: [],
			commandQueueSourceRules: []
		}
	    sfxCreateMask.setZero();
	    sfxDestroyMask.setZero();

		seedsToPlay_CanMove=[];
		seedsToPlay_CantMove=[];

		calculateRowColMasks();

		var alreadyResolved=[];

        var i=0;
        do {
        //not particularly elegant, but it'll do for now - should copy the world state and check
        //after each iteration
        	rigidloop=false;
        	i++;
        	
        	applyRules(state.rules, state.loopPoint, startRuleGroupIndex, bannedGroup);
        	var shouldUndo = resolveMovements(level, bannedGroup);

        	if (shouldUndo) {
        		rigidloop=true;

				{
					// trackback
					if (IDE){
						// newBannedGroups is the list of keys of bannedGroup that aren't already in alreadyResolved
						var newBannedGroups = [];
						for (var key in bannedGroup) {
							if (!alreadyResolved.includes(key)) {
								newBannedGroups.push(key);
								alreadyResolved.push(key);
							}
						}
						var bannedLineNumbers = newBannedGroups.map( rgi => state.rules[rgi][0].lineNumber);
						var ts = bannedLineNumbers.length>1 ? "lines " : "line ";
						ts += bannedLineNumbers.map(ln => `<a onclick="jumpToLine(${ln});" href="javascript:void(0);">${ln}</a>`).join(", ");
						consolePrint(`Rigid movement application failed in rule-Group starting from ${ts}, and will be disabled in resimulation. Rolling back...`)
					}
					//don't need to concat or anythign here, once something is restored it won't be used again.
					level.objects = new Int32Array(startState.objects)
					level.movements = new Int32Array(startState.movements)
					level.rigidGroupIndexMask = startState.rigidGroupIndexMask.concat([])
					level.rigidMovementAppliedMask = startState.rigidMovementAppliedMask.concat([])
					// TODO: shouldn't we also save/restore the level data computed by level.calculateRowColMasks() ?
					level.commandQueue = startState.commandQueue.concat([])
					level.commandQueueSourceRules = startState.commandQueueSourceRules.concat([])
					sfxCreateMask.setZero()
					sfxDestroyMask.setZero()
					// TODO: should

				}

				if (verbose_logging && rigidloop && i>0){				
					consolePrint('Relooping through rules because of rigid.');
						
					debugger_turnIndex++;
					addToDebugTimeline(level,-2);//pre-movement-applied debug state
				}

        		startRuleGroupIndex=0;//rigidGroupUndoDat.ruleGroupIndex+1;
        	} else {
        		if (verbose_logging){

					var eof_idx = debug_visualisation_array[debugger_turnIndex].length+1;//just need some number greater than any rule group
					var inspect_ID = addToDebugTimeline(level,eof_idx);

					consolePrint(`Processed movements.`,false,null,inspect_ID);
					
					if (state.lateRules.length>0){
											
						debugger_turnIndex++;
						addToDebugTimeline(level,-2);//pre-movement-applied debug state
					
						consolePrint('Applying late rules');
					}
				}
        		applyRules(state.lateRules, state.lateLoopPoint, 0);
        		startRuleGroupIndex=0;
        	}
        } while (i < 50 && rigidloop);

        if (i>=50) {
            consolePrint("Looped through 50 times, gave up.  too many loops!");
        }


        if (playerPositions.length>0 && state.metadata.require_player_movement!==undefined) {
        	var somemoved=false;
        	for (var i=0;i<playerPositions.length;i++) {
        		var pos = playerPositions[i];
        		var val = level.getCell(pos);
        		if (state.playerMask.bitsClearInArray(val.data)) {
        			somemoved=true;
        			break;
        		}
        	}
        	if (somemoved===false) {
        		if (verbose_logging){
	    			consolePrint('require_player_movement set, but no player movement detected, so cancelling turn.');
	    			consoleCacheDump();
				}
        		addUndoState(bak);
        		DoUndo(true,false);
        		return false;
        	}
        	//play player cantmove sounds here
        }



	    if (level.commandQueue.indexOf('cancel')>=0) {
	    	if (verbose_logging) { 
	    		consoleCacheDump();
	    		var r = level.commandQueueSourceRules[level.commandQueue.indexOf('cancel')];
	    		consolePrintFromRule('CANCEL command executed, cancelling turn.',r,true);
			}

			if (!dontModify){
				processOutputCommands(level.commandQueue);
			}

			var commandsleft = level.commandQueue.length>1;

    		addUndoState(bak);
    		DoUndo(true,false);
    		tryPlayCancelSound();
    		return commandsleft;
	    } 

	    if (level.commandQueue.indexOf('restart')>=0) {
			
    		if (verbose_logging && runrulesonlevelstart_phase){
				var r = level.commandQueueSourceRules[level.commandQueue.indexOf('restart')];
    			logWarning('A "restart" command is being triggered in the "run_rules_on_level_start" section of level creation, which would cause an infinite loop if it was actually triggered, but it\'s being ignored, so it\'s not.',r.lineNumber,true);
    		}

	    	if (verbose_logging) { 
	    		var r = level.commandQueueSourceRules[level.commandQueue.indexOf('restart')];
	    		consolePrintFromRule('RESTART command executed, reverting to restart state.',r.lineNumber);
	    		consoleCacheDump();
			}
			if (!dontModify){
				processOutputCommands(level.commandQueue);
			}
    		// addUndoState(bak);

			if (!dontModify){
	    		DoRestart(true);
			}
    		return true;
		} 
		
		
        var modified=false;
	    for (var i=0;i<level.objects.length;i++) {
	    	if (level.objects[i]!==bak.dat[i]) {
				// if (dontModify) {
	      //   		if (verbose_logging) {
	      //   			consoleCacheDump();
	      //   		}
	      //   		addUndoState(bak);
	      //   		DoUndo(true,false);
				// 	return true;
				// } else {
					if (dir!==-1) {
						// addUndoState(bak);
					} else if (backups.length > 0) {
						// This is for the case that diffs break the undo buffer for real-time games 
						// ( c f https://github.com/increpare/PuzzleScript/pull/796 ),
						// because realtime ticks are ignored when the user presses undo and the backup
						// array reflects this structure.  
						backups[backups.length - 1] = unconsolidateDiff(backups[backups.length - 1], bak);					
	    			}
	    			modified=true;
	    		// }
	    		break;
	    	}
	    }

		if (dontModify && level.commandQueue.indexOf('win')>=0) {	
	    	return true;	
		}
		
		if (dontModify) {		
    		if (verbose_logging) {
    			consoleCacheDump();
    		}
			return false;
		}

        // for (var i=0;i<seedsToPlay_CantMove.length;i++) {			
	      //   	playSound(seedsToPlay_CantMove[i]);
        // }

        // for (var i=0;i<seedsToPlay_CanMove.length;i++) {
	      //   	playSound(seedsToPlay_CanMove[i]);
        // }

        // for (var i=0;i<state.sfx_CreationMasks.length;i++) {
        // 	var entry = state.sfx_CreationMasks[i];
        // 	if (sfxCreateMask.anyBitsInCommon(entry.objectMask)) {
	      //   	playSound(entry.seed);
        // 	}
        // }

        // for (var i=0;i<state.sfx_DestructionMasks.length;i++) {
        // 	var entry = state.sfx_DestructionMasks[i];
        // 	if (sfxDestroyMask.anyBitsInCommon(entry.objectMask)) {
	      //   	playSound(entry.seed);
        // 	}
        // }

		if (!dontModify){
	    	processOutputCommands(level.commandQueue);
		}

	    if (textMode===false) {
	    	if (verbose_logging) { 
	    		consolePrint('Checking win conditions.');
			}
			if (dontDoWin===undefined){
				dontDoWin = false;
			}
	    	checkWin( dontDoWin );
	    }

	    if (!winning) {
			if (level.commandQueue.indexOf('checkpoint')>=0) {
		    	if (verbose_logging) { 
	    			var r = level.commandQueueSourceRules[level.commandQueue.indexOf('checkpoint')];
		    		consolePrintFromRule('CHECKPOINT command executed, saving current state to the restart state.',r);
				}
				restartTarget=level4Serialization();
				hasUsedCheckpoint=true;
				var backupStr = JSON.stringify(restartTarget);
				storage_set(document.URL+'_checkpoint',backupStr);
				storage_set(document.URL,curlevel);				
			}	 

		    if (level.commandQueue.indexOf('again')>=0 && modified) {

	    		var r = level.commandQueueSourceRules[level.commandQueue.indexOf('again')];

		    	//first have to verify that something's changed
		    	var old_verbose_logging=verbose_logging;
		    	var oldmessagetext = messagetext;
		    	verbose_logging=false;
		    	if (processInput(-1,true,true)) {
			    	verbose_logging=old_verbose_logging;

			    	if (verbose_logging) { 
			    		consolePrintFromRule('AGAIN command executed, with changes detected - will execute another turn.',r);
					}

			    	againing=true;
			    	timer=0;
			    } else {		    	
			    	verbose_logging=old_verbose_logging;
					if (verbose_logging) { 
						consolePrintFromRule('AGAIN command not executed, it wouldn\'t make any changes.',r);
					}
			    }
			    verbose_logging=old_verbose_logging;
			    messagetext = oldmessagetext;
		    }   
		}
		
		if (verbose_logging) { 
			consolePrint(`Turn complete`);    
		}
		
	    level.commandQueue=[];
	    level.commandQueueSourceRules=[];

    }

	if (verbose_logging) {
		consoleCacheDump();
	}

	if (winning) {
		againing=false;
	}

	return modified;
}


async function solveLevel(level) {
  // Load the level
  compile(['loadLevel', level], editor.getValue());
  // console.log('Solving level', level);
  init_level = backupLevel();
  init_level_map = init_level['dat'];
  frontier = [init_level];
  action_seqs = [[]];
  // frontier = new Deque([init_level]);
  // action_seqs = new Deque([[]]);
  sol = [];
  console.log(sol.length);
  visited = new Set([serialize(init_level_map)]);
  i = 0;
  start_time = Date.now();
  while (frontier.length > 0) {
    const level = frontier.pop(0);
    const action_seq = action_seqs.pop(0);
    if (!action_seq) {
      console.log(`Action sequence is empty. Length of frontier: ${frontier.length}`);
    }
    // const level = frontier.shift();
    // const action_seq = action_seqs.shift();
    for (const move of Array(5).keys()) {
      if (i > 1_000_000) {
        console.log('Exceeded 1M iterations. Exiting.');
        return [-1, i];
      }
      restoreLevel(level);
      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      // console.time(`processInput-${i}-${move}`); // Start profiling
      changed = processInput(move);
      // console.timeEnd(`processInput-${i}-${move}`); // End profiling
      if (winning) {
        console.log(`Winning! Solution:, ${new_action_seq}`);
        return [new_action_seq, i];
      }
      else if (changed) {
        new_level = backupLevel();
        new_level_map = new_level['dat'];
        if (!visited.has(serialize(new_level_map))) {
          
          // UNCOMMENT THESE LINES FOR VISUAL DEBUGGING
          // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
          // redraw();

          frontier.push(new_level);
          if (!new_action_seq) {
            console.log(`New action sequence is undefined when pushing.`);
          }
          action_seqs.push(new_action_seq);
          visited.add(serialize(new_level_map));
        } 
      }
    }
    if (i % 1000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', i / (now - start_time) * 1000);
    }
    i++;
  }
  return [sol, i];
}


async function genGame(genMode, parents, saveDir, seed, fewshot, cot,
    fromIdea, idea='', maxGenAttempts=10) {
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

    // Get our GPT completion from python
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
        from_idea: fromIdea,
        game_idea: idea,
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
            console.log('Using cached solution.');
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
        fitness = Math.max(fitness, n_search_iters)
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
      if (solvable) {
        solvedIters.push(nGenAttempts)
      }
      const newlySaved = await fetch('/save_sols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          save_dir: saveDir,
          sols: sols,
          n_iter: nGenAttempts,
        }),
      });

      // if (newlySaved)
      // Save a gif of each solution

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

const seed = 12;

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
          saveDir = `sweep-${seed}`
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
      saveDir = `sweep-${seed}`
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

// sweep()
fromIdeaSweep()

// genGame('init', [], 'test_99', 99, fewshot=true, cot=true, maxGenAttempts=20);
// evolve();
// playTest();