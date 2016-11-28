'use strict';


function isColor(str) {
	str = str.trim();
	if (str in colorPalettes.arnecolors)
		return true;
	if (/^#([0-9A-F]{3}){1,2}$/i.test(str))
		return true;
	if (str === "transparent")
		return true;
	return false;
}

function colorToHex(palette,str) {
	str = str.trim();
	if (str in palette) {
		return palette[str];
	}

	return str;
}


function generateSpriteMatrix(dat) {

	var result = [];
	for (var i = 0; i < dat.length; i++) {
		var row = [];
		for (var j = 0; j < dat.length; j++) {
			var ch = dat[i].charAt(j);
			if (ch == '.') {
				row.push(-1);
			} else {
				row.push(ch);
			}
		}
		result.push(row);
	}
	return result;
}

var debugMode;
var colorPalette;

function generateExtraMembers(state) {

	if (state.collisionLayers.length===0) {
		logError("No collision layers defined.  All objects need to be in collision layers.");
	}

	//annotate objects with layers
	//assign ids at the same time
	state.idDict = {};
	var idcount=0;
	for (var layerIndex = 0; layerIndex < state.collisionLayers.length; layerIndex++) {
		for (var j = 0; j < state.collisionLayers[layerIndex].length; j++)
		{
			var n = state.collisionLayers[layerIndex][j];
			if (n in  state.objects)  {
				var o = state.objects[n];
				o.layer = layerIndex;
				o.id=idcount;
				state.idDict[idcount]=n;
				idcount++;
			}
		}
	}

	//set object count
	state.objectCount = idcount;

	//calculate blank mask template
	var layerCount = state.collisionLayers.length;
	var blankMask = [];
	for (var i = 0; i < layerCount; i++) {
		blankMask.push(-1);
	}

	// how many words do our bitvecs need to hold?
	STRIDE_OBJ = Math.ceil(state.objectCount/32)|0;
	STRIDE_MOV = Math.ceil(layerCount/5)|0;
	state.STRIDE_OBJ=STRIDE_OBJ;
	state.STRIDE_MOV=STRIDE_MOV;
	
	//get colorpalette name
	debugMode=false;
	verbose_logging=false;
	throttle_movement=false;
	colorPalette=colorPalettes.arnecolors;
	for (var i=0;i<state.metadata.length;i+=2){
		var key = state.metadata[i];
		var val = state.metadata[i+1];
		if (key==='color_palette') {
			if (val in colorPalettesAliases) {
				val = colorPalettesAliases[val];
			}
			if (colorPalettes[val]===undefined) {
				logError('Palette "'+val+'" not found, defaulting to arnecolors.',0);
			}else {
				colorPalette=colorPalettes[val];
			}
		} else if (key==='debug') {
			debugMode=true;
			cache_console_messages=true;
		} else if (key ==='verbose_logging') {
			verbose_logging=true;
			cache_console_messages=true;
		} else if (key==='throttle_movement') {
			throttle_movement=true;
		}
	}

	//convert colors to hex
	for (var n in state.objects) {
	      if (state.objects.hasOwnProperty(n)) {
			//convert color to hex
	      	var o = state.objects[n];
	      	if (o.colors.length>10) {
	      		logError("a sprite cannot have more than 10 colors.  Why you would want more than 10 is beyond me.",o.lineNumber+1);
	      	}
	      	for (var i=0;i<o.colors.length;i++) {
	      		var c = o.colors[i];
				if (isColor(c)) {
					c = colorToHex(colorPalette,c);
					o.colors[i] = c;
				} else {
					logError('Invalid color specified for object "' + n + '", namely "' + o.colors[i] + '".', o.lineNumber + 1);
					o.colors[i] = '#ff00ff'; // magenta error color
				}
			}
		}
	}

	//generate sprite matrix
	for (var n in state.objects) {
	      if (state.objects.hasOwnProperty(n)) {
	      	var o = state.objects[n];
	      	if (o.colors.length==0) {
	      		logError('color not specified for object "' + n +'".',o.lineNumber);
	      		o.colors=["#ff00ff"];
	      	}
			if (o.spritematrix.length===0) {
				o.spritematrix = [
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0], 
					[0,0,0,0,0,0,0,0]
				]
			} else {
				if ( o.spritematrix.length!==8 || o.spritematrix[0].length!==8 || o.spritematrix[1].length!==8 || o.spritematrix[2].length!==8 || o.spritematrix[3].length!==8 || o.spritematrix[4].length!==8 || o.spritematrix[5].length!==8 || o.spritematrix[6].length!==8 || o.spritematrix[7].length!==8 ){
					logWarning("Sprite graphics must be 8 wide and 8 high exactly.",o.lineNumber);
				}
				o.spritematrix = generateSpriteMatrix(o.spritematrix);
			}
		}
	}


	//calculate glyph dictionary
	var glyphDict = { };
	for (var n in state.objects) {
	      if (state.objects.hasOwnProperty(n)) {
	      	var o = state.objects[n];
			var mask = blankMask.concat([]);
			mask[o.layer] = o.id;
			glyphDict[n] = mask;
		}
	}
 	var added=true;
    while (added) {
        added=false;
        
        //then, synonyms
        for (var i = 0; i < state.legend_synonyms.length; i++) {
            var dat = state.legend_synonyms[i];
            var key = dat[0];
            var val = dat[1];
            if ((!(key in glyphDict)||(glyphDict[key]===undefined))&&(glyphDict[val]!==undefined)) {
                added=true;
                glyphDict[key] = glyphDict[val];
            }
        }
    
        //then, aggregates
        for (var i = 0; i < state.legend_aggregates.length; i++) {
            var dat = state.legend_aggregates[i];
            var key=dat[0];
            var vals=dat.slice(1);
            var allVallsFound=true;
            for (var j=0;j<vals.length;j++) {
            	var v = vals[j];
            	if (glyphDict[v]===undefined) {
            		allVallsFound=false;
            		break;
            	}
            }
            if ((!(key in glyphDict)||(glyphDict[key]===undefined))&&allVallsFound) {            
                var mask = blankMask.concat([]);
        
                for (var j = 1; j < dat.length; j++) {
                    var n = dat[j];
                    var o = state.objects[n];
                    if (o == undefined) {
                        logError('Object not found with name '+ n, state.lineNumber);
                    }
                    if (mask[o.layer] == -1) {
                        mask[o.layer] = o.id;
                    } else {
                    	if (o.layer===undefined) {
                    		logError('Object "' + n.toUpperCase() + '" has been defined, but not assigned to a layer.',dat.lineNumber);
                    	} else {                    		
	                        logError(
	                            'Trying to create an aggregate object (defined in the legend) with both "'
	                            + n.toUpperCase() + '" and "' + state.idDict[mask[o.layer]].toUpperCase() + '", which are on the same layer and therefore can\'t coexist.',
	                            dat.lineNumber
	                            );
	                    }
                    }
                }
                added=true;
                glyphDict[dat[0]] = mask;
            }
        }
    }
	state.glyphDict = glyphDict;

	var aggregatesDict = {};
	for (var i = 0; i < state.legend_aggregates.length; i++) {
		var entry = state.legend_aggregates[i];
		aggregatesDict[entry[0]] = entry.slice(1);
	}
	state.aggregatesDict = aggregatesDict;

	var propertiesDict = {};
	for (var i = 0; i < state.legend_properties.length; i++) {
		var entry = state.legend_properties[i];
		propertiesDict[entry[0]] = entry.slice(1);
	}
	state.propertiesDict = propertiesDict;

	//calculate lookup dictionaries
	var synonymsDict = {};
	for (var i = 0; i < state.legend_synonyms.length; i++) {
		var entry = state.legend_synonyms[i];
		var key = entry[0];
		var value=entry[1];
		if (value in aggregatesDict) {
			aggregatesDict[key]=aggregatesDict[value];
		}
		else if (value in propertiesDict) {
			propertiesDict[key]=propertiesDict[value];
		} else if (key!==value) {
			synonymsDict[key] = value;		
		}
	}
	state.synonymsDict = synonymsDict;

	var modified=true;
	while(modified){
		modified=false;
		for (var n in synonymsDict) {
			if (synonymsDict.hasOwnProperty(n)) {
				var value = synonymsDict[n];
				if (value in propertiesDict) {
					delete synonymsDict[n];
					propertiesDict[n]=propertiesDict[value];
					modified=true;
				}
				else if (value in aggregatesDict) {
					delete aggregatesDict[n];
					aggregatesDict[n]=aggregatesDict[value];
					modified=true;
				} else if (value in synonymsDict) {
					synonymsDict[n]=synonymsDict[value];
				}
			}
		}

		for (var n in propertiesDict) {
			if (propertiesDict.hasOwnProperty(n)) {
				var values = propertiesDict[n];
				for (var i=0;i<values.length;i++) {
					var value = values[i];
					if (value in synonymsDict) {
						values[i]=synonymsDict[value];
						modified=true;
					} else if (value in propertiesDict) {
						values.splice(i,1);
						var newvalues=propertiesDict[value];
						for (var j=0;j<newvalues.length;j++) {
							var newvalue=newvalues[j];
							if (values.indexOf(newvalue)===-1) {
								values.push(newvalue);
							}
						}
						modified=true;
					} if (value in aggregatesDict) {
						logError('Trying to define property "' + n.toUpperCase() +'" in terms of aggregate "'+value.toUpperCase()+'".');
					}
				}
			}
		}


		for (var n in aggregatesDict) {
			if (aggregatesDict.hasOwnProperty(n)) {
				var values = aggregatesDict[n];
				for (var i=0;i<values.length;i++) {
					var value = values[i];
					if (value in synonymsDict) {
						values[i]=synonymsDict[value];
						modified=true;
					} else if (value in aggregatesDict) {
						values.splice(i,1);
						var newvalues=aggregatesDict[value];
						for (var j=0;j<newvalues.length;j++) {
							var newvalue=newvalues[j];
							if (values.indexOf(newvalue)===-1) {
								values.push(newvalue);
							}
						}
						modified=true;
					} if (value in propertiesDict) {
						logError('Trying to define aggregate "' + n.toUpperCase() +'" in terms of property "'+value.toUpperCase()+'".');
					}
				}
			}
		}
	}

	/* determine which properties specify objects all on one layer */
	state.propertiesSingleLayer = {};
	for (var key in propertiesDict) {
		if (propertiesDict.hasOwnProperty(key)) {
			var values = propertiesDict[key];
			var sameLayer = true;
			for (var i = 1; i < values.length; i++) {
				if ((state.objects[values[i-1]].layer !== state.objects[values[i]].layer)) {
					sameLayer = false;
					break;
				}
			}
			if (sameLayer) {
				state.propertiesSingleLayer[key] = state.objects[values[0]].layer;
			}
		}
	}

	if (state.idDict[0]===undefined && state.collisionLayers.length>0) {
		logError('You need to have some objects defined');
	}
}

function levelFromString(state,level) {
	var o = new Level(level[0], level[1].length, level.length-1, state.collisionLayers.length, null);
	o.objects = new Int32Array(o.width * o.height * STRIDE_OBJ);

	for (var i = 0; i < o.width; i++) {
		for (var j = 0; j < o.height; j++) {
			var ch = level[j+1].charAt(i);
			if (ch.length==0) {
				ch=level[j+1].charAt(level[j+1].length-1);
			}
			var mask = state.glyphDict[ch];

			if (ch==="."){
				var blankMask = [];
				for (var i_ = 0; i_ < state.collisionLayers.length; i_++) {
					blankMask.push(-1);
				}
				mask = blankMask;
			}
			if (mask == undefined) {
				if (state.propertiesDict[ch]===undefined) {
					logError('Error, symbol "' + ch + '", used in map, not found.', level[0]+j);
				} else {
					logError('Error, symbol "' + ch + '" is defined using \'or\', and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of \'and\'?', level[0]+j);							
				}

			}

			var maskint = new BitVec(STRIDE_OBJ);
			mask = mask.concat([]);					
			for (var z = 0; z < o.layerCount; z++) {
				if (mask[z]>=0) {
					maskint.ibitset(mask[z]);
				}
			}
			for (var w = 0; w < STRIDE_OBJ; ++w) {
				o.objects[STRIDE_OBJ * (i * o.height + j) + w] = maskint.data[w];
			}
		}
	}
	return o;
}
//also assigns glyphDict
function levelsToArray(state) {
	var levels = state.levels;
	var processedLevels = [];

	for (var levelIndex = 0; levelIndex < levels.length; levelIndex++) {
		var level = levels[levelIndex];
		if (level.length == 0) {
			continue;
		}
		if (level[0] == '\n') {
			var o = {
				message: level[1]
			};
			processedLevels.push(o);
		} else {
			var o = levelFromString(state,level);
			processedLevels.push(o);
		}

	}

	state.levels = processedLevels;
}

var directionaggregates = {
	'horizontal' : ['left', 'right'],
	'vertical' : ['up', 'down'],
	'moving' : ['up', 'down', 'left', 'right', 'action'],
	'orthogonal' : ['up', 'down', 'left', 'right'],
	'perpendicular' : ['^','v'],
	'parallel' : ['<','>']
};

var relativeDirections = ['^', 'v', '<', '>','horizontal','vertical'];
var simpleAbsoluteDirections = ['up', 'down', 'left', 'right'];
var simpleRelativeDirections = ['^', 'v', '<', '>'];
var reg_directions_only = /^(\>|\<|\^|v|up|down|left|right|moving|stationary|no|randomdir|random|horizontal|vertical|orthogonal|perpendicular|parallel|action)$/;
//redeclaring here, i don't know why
var commandwords = ["sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10","cancel","checkpoint","restart","win","message","again"];



function directionalRule(rule) {
	for (var i=0;i<rule.lhs.length;i++) {
		var cellRow = rule.lhs[i];
		if (cellRow.length>1) {
			return true;
		}
		for (var j=0;j<cellRow.length;j++) {
			var cell = cellRow[j];
			for (var k=0;k<cell.length;k+=2) {
				if (relativeDirections.indexOf(cell[k])>=0) {
					return true;
				}
			}
		}
	}
	for (var i=0;i<rule.rhs.length;i++) {
		var cellRow = rule.rhs[i];
		if (cellRow.length>1) {
			return true;
		}
		for (var j=0;j<cellRow.length;j++) {
			var cell = cellRow[j];
			for (var k=0;k<cell.length;k+=2) {
				if (relativeDirections.indexOf(cell[k])>=0) {
					return true;
				}
			}
		}
	}
	return false;
}

function processRuleString(rule, state, curRules) 
{
/*

	intermediate structure
		dirs: Directions[]
		pre : CellMask[]
		post : CellMask[]

		//pre/post pairs must have same lengths
	final rule structure
		dir: Direction
		pre : CellMask[]
		post : CellMask[]
*/
	var line = rule[0];
	var lineNumber = rule[1];
	var origLine = rule[2];

// STEP ONE, TOKENIZE
	line = line.replace(/\[/g, ' [ ').replace(/\]/g, ' ] ').replace(/\|/g, ' | ').replace(/\-\>/g, ' -> ');
	var tokens = line.split(/\s/).filter(function(v) {return v !== ''});

	if (tokens.length == 0) {
		logError('Spooky error!  Empty line passed to rule function.', lineNumber);
	}


// STEP TWO, READ DIRECTIONS
/*
	STATE
	0 - scanning for initial directions
	LHS
	1 - reading cell contents LHS
	2 - reading cell contents RHS
*/
	var parsestate = 0;
	var directions = [];

	var curcell = null; // [up, cat, down mouse]
	var curcellrow = []; // [  [up, cat]  [ down, mouse ] ]

	var appendGroup=false;
	var rhs = false;
	var lhs_cells = [];
	var rhs_cells = [];
	var late = false;
	var rigid = false;
	var groupNumber=lineNumber;
	var commands=[];
	var randomRule=false;

	if (tokens.length===1) {
		if (tokens[0]==="startloop" ) {
			rule_line = {
				bracket: 1
			}
			return rule_line;
		} else if (tokens[0]==="endloop" ) {
			rule_line = {
				bracket: -1
			}
			return rule_line;
		}
	}

	if (tokens.indexOf('->') == -1) {
		logError("A rule has to have an arrow in it.  There's no arrow here! Consider reading up about rules - you're clearly doing something weird", lineNumber);
	}

	for (var i = 0; i < tokens.length; i++)
	{
		var token = tokens[i];
		switch (parsestate) {
			case 0: {
				//read initial directions
				if (token==='+') {					
					if (groupNumber===lineNumber) {
						if (curRules.length==0) {
							logError('The "+" symbol, for joining a rule with the group of the previous rule, needs a previous rule to be applied to.');							
						}
						if (i!==0) {
							logError('The "+" symbol, for joining a rule with the group of the previous rule, must be the first symbol on the line ');
						}						
						groupNumber = curRules[curRules.length-1].groupNumber;
					} else {
						logError('Two "+"s ("append to previous rule group" symbol)applied to the same rule.',lineNumber);
					}
				} else if (token in directionaggregates) {
					directions = directions.concat(directionaggregates[token]);						
				} else if (token==='late') {
						late=true;
				} else if (token==='rigid') {
					rigid=true;
				} else if (token==='random') {
					randomRule=true;
				} else if (simpleAbsoluteDirections.indexOf(token) >= 0) {
					directions.push(token);
				} else if (simpleRelativeDirections.indexOf(token) >= 0) {
					logError('You cannot use relative directions (\"^v<>\") to indicate in which direction(s) a rule applies.  Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions', lineNumber);
				} else if (token == '[') {
					if (directions.length == 0) {
						directions = directions.concat(directionaggregates['orthogonal']);
					}
					parsestate = 1;
					i--;
				} else {
					logError("The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions).  It seems you've just entered \"" + token.toUpperCase() + '\".', lineNumber);
				}
				break;
			}
			case 1: {
				if (token == '[') {
					if (curcellrow.length > 0) {
						logError('Error, malformed cell rule - encountered a "["" before previous bracket was closed', lineNumber);
					}
					curcell = [];
				} else if (reg_directions_only.exec(token)) {
					if (curcell.length % 2 == 1) {
						logError("Error, an item can't move in multiple directions.", lineNumber);
					} else {
						curcell.push(token);
					}
				} else if (token == '|') {
					if (curcell.length % 2 == 1) {
						logError('In a rule, if you specify a force, it has to act on an object.', lineNumber);
					} else {
						curcellrow.push(curcell);
						curcell = [];
					}
				} else if (token === ']') {
					if (curcell.length % 2 == 1) {
						if (curcell[0]==='...') {
							logError('Cannot end a rule with ellipses.', lineNumber);
						} else {
							logError('In a rule, if you specify a force, it has to act on an object.', lineNumber);
						}
					} else {
						curcellrow.push(curcell);
						curcell = [];
					}

					if (rhs) {
						rhs_cells.push(curcellrow);
					} else {
						lhs_cells.push(curcellrow);
					}
					curcellrow = [];
				} else if (token === '->') {
					if (rhs) {
						logError('Error, you can only use "->" once in a rule; it\'s used to separate before and after states.', lineNumber);
					} if (curcellrow.length > 0) {
						logError('Encountered an unexpected "->" inside square brackets.  It\'s used to separate states, it has no place inside them >:| .', lineNumber);
					} else {
						rhs = true;
					}
				} else if (state.names.indexOf(token) >= 0) {
					if (curcell.length % 2 == 0) {
						curcell.push('');
						curcell.push(token);
					} else if (curcell.length % 2 == 1) {
						curcell.push(token);
					}
				} else if (token==='...') {
					curcell.push(token);
					curcell.push(token);
				} else if (commandwords.indexOf(token)>=0) {
					if (rhs===false) {
						logError("Commands cannot appear on the left-hand side of the arrow.",lineNumber);
					}
					if (token==='message') {
						var message_match = origLine.match(/message (.*)/i);
						if (message_match === null) {
							logError("invalid message string", lineNumber);
						} else {
							commands.push([token, message_match[1].trim()]);
							i=tokens.length;
						}
					} else {
						commands.push([token]);
					}
				} else {
					logError('Error, malformed cell rule - was looking for cell contents, but found "' + token + '".  What am I supposed to do with this, eh, please tell me that.', lineNumber);
				}
			}

		}
	}

	if (lhs_cells.length != rhs_cells.length) {
		if (commands.length>0&&rhs_cells.length==0) {
			//ok
		} else {
			logError('Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right', lineNumber);
		}
	} else {
		for (var i = 0; i < lhs_cells.length; i++) {
			if (lhs_cells[i].length != rhs_cells[i].length) {
				logError('In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells).', lineNumber);
			}
			if (lhs_cells[i].length == 0) {
				logError("You have an totally empty pattern on the left-hand side.  This will match *everything*.  You certianly don't want this.");
			}
		}
	}

	if (lhs_cells.length == 0) {
		logError('This rule refers to nothing.  What the heck? :O', lineNumber);
	}

	var rule_line = {
		directions: directions,
		lhs: lhs_cells,
		rhs: rhs_cells,
		lineNumber: lineNumber,
		late: late,
		rigid: rigid,
		groupNumber: groupNumber,
		commands: commands,
		randomRule: randomRule
	};

	if (directionalRule(rule_line)===false) {
		rule_line.directions=['up'];
	}

	/* reset must appear by itself */

	for (var i=0;i<commands.length;i++) {
		var cmd = commands[i][0];
		if (cmd==='restart') {
			if (commands.length>1 || rhs_cells.length>0) {
				logError('The RESTART command can only appear by itself on the right hand side of the arrow.', lineNumber);
			}
		} else if (cmd==='cancel') {
			if (commands.length>1 || rhs_cells.length>0) {
				logError('The CANCEL command can only appear by itself on the right hand side of the arrow.', lineNumber);
			}
		}
	}

	//next up - replace relative directions with absolute direction

	return rule_line;
}

function deepCloneHS(HS) {
	var cloneHS = HS.map(function(arr) {return arr.map(function(deepArr) {return deepArr.slice();});});
	return cloneHS;
}

function deepCloneRule(rule) {
	var clonedRule = {
		direction: rule.direction,
		lhs: deepCloneHS(rule.lhs),
		rhs: deepCloneHS(rule.rhs),
		lineNumber: rule.lineNumber,
		late: rule.late,
		rigid: rule.rigid,
		groupNumber: rule.groupNumber,
		commands:rule.commands,
		randomRule:rule.randomRule
	};
	return clonedRule;
}

function rulesToArray(state) {
	var oldrules = state.rules;
	var rules = [];
	var loops=[];
	for (var i = 0; i < oldrules.length; i++) {
		var lineNumber = oldrules[i][1];
		var newrule = processRuleString(oldrules[i], state, rules);
		if (newrule.bracket!==undefined) {
			loops.push([lineNumber,newrule.bracket]);
			continue;
		}
		rules.push(newrule);
	}
	state.loops=loops;

	//now expand out rules with multiple directions
	var rules2 = [];
	for (var i = 0; i < rules.length; i++) {
		var rule = rules[i];
		var ruledirs = rule.directions;
		for (var j = 0; j < ruledirs.length; j++) {
			var dir = ruledirs[j];
			if (dir in directionaggregates && directionalRule(rule)) {
				var dirs = directionaggregates[dir];
				for (var k = 0; k < dirs.length; k++) {
					var modifiedrule = deepCloneRule(rule);
					modifiedrule.direction = dirs[k];
					rules2.push(modifiedrule);
				}
			} else {
				var modifiedrule = deepCloneRule(rule);
				modifiedrule.direction = dir;
				rules2.push(modifiedrule);
			}
		}
	}

	for (var i = 0; i < rules2.length; i++) {
		var rule = rules2[i];
		//remove relative directions
		convertRelativeDirsToAbsolute(rule);
		//optional: replace up/left rules with their down/right equivalents
		rewriteUpLeftRules(rule);
		//replace aggregates with what they mean
		atomizeAggregates(state, rule);
		//replace synonyms with what they mean
		rephraseSynonyms(state, rule);
	}

	var rules3 = [];
	//expand property rules
	for (var i = 0; i < rules2.length; i++) {
		var rule = rules2[i];
		rules3 = rules3.concat(concretizeMovingRule(state, rule,rule.lineNumber));
	}

	var rules4 = [];
	for (var i=0;i<rules3.length;i++) {
		var rule=rules3[i];
		rules4 = rules4.concat(concretizePropertyRule(state, rule,rule.lineNumber));

	}

	state.rules = rules4;
}

function containsEllipsis(rule) {
	for (var i=0;i<rule.lhs.length;i++) {
		for (var j=0;j<rule.lhs[i].length;j++) {
			if (rule.lhs[i][j][1]==='...')
				return true;
		}
	}
	return false;
}

function rewriteUpLeftRules(rule) {
	if (containsEllipsis(rule)) {
		return;
	}

	if (rule.direction == 'up') {
		rule.direction = 'down';
	} else if (rule.direction == 'left') {
		rule.direction = 'right';
	} else {
		return;
	}

	for (var i = 0; i < rule.lhs.length; i++) {
		rule.lhs[i].reverse();
		if (rule.rhs.length>0) {
			rule.rhs[i].reverse();
		}
	}
}

function getPropertiesFromCell(state,cell ) {
	var result = [];
	for (var j = 0; j < cell.length; j += 2) {
		var dir = cell[j];
		var name = cell[j+1];
		if (dir=="random") {
			continue;
		}
		if (name in state.propertiesDict) {
			result.push(name);
		}
	}
	return result;
}

//returns you a list of object names in that cell that're moving
function getMovings(state,cell ) {
	var result = [];
	for (var j = 0; j < cell.length; j += 2) {
		var dir = cell[j];
		var name = cell[j+1];
		if (dir in directionaggregates) {
			result.push([name,dir]);
		}
	}
	return result;
}

function concretizePropertyInCell(cell ,property, concreteType) {
	for (var j = 0; j < cell.length; j += 2) {
		if (cell[j+1] === property && cell[j]!=="random") {
			cell[j+1] = concreteType;
		}
	}
}

function concretizeMovingInCell(cell , ambiguousMovement, nameToMove, concreteDirection) {
	for (var j = 0; j < cell.length; j += 2) {
		if (cell[j]===ambiguousMovement && cell[j+1] === nameToMove) {
			cell[j] = concreteDirection;
		}
	}
}

function concretizeMovingInCellByAmbiguousMovementName(cell ,ambiguousMovement, concreteDirection) {
	for (var j = 0; j < cell.length; j += 2) {
		if (cell[j] === ambiguousMovement) {
			cell[j] = concreteDirection;
		}
	}
}

function expandNoPrefixedProperties(state, cell) {
	var expanded = [];
	for (var i=0;i<cell.length;i+=2)  {
		var dir = cell[i];
		var name = cell[i+1];

		if ( dir ==='no' && (name in state.propertiesDict)) {
			var aliases = state.propertiesDict[name];
			for (var j=0;j<aliases.length;j++) {
				var alias = aliases[j];
				expanded.push(dir);
				expanded.push(alias);
			}
		} else {
			expanded.push(dir);
			expanded.push(name);
		} 
	}
	return expanded;
}

function concretizePropertyRule(state, rule,lineNumber) {	

	//step 1, rephrase rule to change "no flying" to "no cat no bat"
	for (var i = 0; i < rule.lhs.length; i++) {
		var cur_cellrow_l = rule.lhs[i];
		for (var j=0;j<cur_cellrow_l.length;j++) {
			cur_cellrow_l[j] = expandNoPrefixedProperties(state,cur_cellrow_l[j]);
			if (rule.rhs.length > 0)
				rule.rhs[i][j] = expandNoPrefixedProperties(state,rule.rhs[i][j]);
		}
	}

	//are there any properties we could avoid processing?
	// e.g. [> player | movable] -> [> player | > movable],
	// 		doesn't need to be split up (assuming single-layer player/block aggregates)

	// we can't manage this if they're being used to disambiguate
	var ambiguousProperties = {};

	for (var j = 0; j < rule.rhs.length; j++) {
		var row_l = rule.lhs[j];
		var row_r = rule.rhs[j];
		for (var k = 0; k < row_r.length; k++) {
			var properties_l = getPropertiesFromCell(state, row_l[k]);
			var properties_r = getPropertiesFromCell(state, row_r[k]);
			for (var prop_n = 0; prop_n < properties_r.length; prop_n++) {
				var property = properties_r[prop_n];
				if (properties_l.indexOf(property) == -1) {
					ambiguousProperties[property] = true;
				}
			}
		}
	}

	var shouldremove;
	var result = [rule];
	var modified=true;
	while (modified) {
		modified = false;
		for (var i = 0; i < result.length; i++) {
			//only need to iterate through lhs
			var cur_rule = result[i];
			shouldremove = false;
			for (var j = 0; j < cur_rule.lhs.length&&!shouldremove; j++) {
				var cur_rulerow = cur_rule.lhs[j];
				for (var k = 0; k < cur_rulerow.length&&!shouldremove; k++) {
					var cur_cell = cur_rulerow[k];
					var properties = getPropertiesFromCell(state, cur_cell);
					for (var prop_n = 0; prop_n < properties.length; ++prop_n) {
						var property = properties[prop_n];

						if (state.propertiesSingleLayer.hasOwnProperty(property) &&
							ambiguousProperties[property] !== true) {
							// we don't need to explode this property
							continue;
						}

						var aliases = state.propertiesDict[property];

						shouldremove = true;
						modified = true;

						//just do the base property, let future iterations take care of the others

						for (var l = 0; l < aliases.length; l++) {
							var concreteType = aliases[l];
							var newrule = deepCloneRule(cur_rule);
							newrule.propertyReplacement={};
							for(var prop in cur_rule.propertyReplacement) {
								if (cur_rule.propertyReplacement.hasOwnProperty(prop)) {
									var propDat = cur_rule.propertyReplacement[prop];
									newrule.propertyReplacement[prop] = [propDat[0],propDat[1]];
								}
							}

							concretizePropertyInCell(newrule.lhs[j][k], property, concreteType);
							if (newrule.rhs.length>0) {
								concretizePropertyInCell(newrule.rhs[j][k], property, concreteType);//do for the corresponding rhs cell as well
							}
                            
                            if (newrule.propertyReplacement[property]===undefined) {
    							newrule.propertyReplacement[property]=[concreteType,1];
                            } else {
    							newrule.propertyReplacement[property][1]=newrule.propertyReplacement[property][1]+1;                                
                            }

							result.push(newrule);
						}

						break;
					}
				}
			}
			if (shouldremove)
			{
				result.splice(i, 1);
				i--;
			}
		}
	}

    
	for (var i = 0; i < result.length; i++) {
        //for each rule
		var cur_rule = result[i];
        if (cur_rule.propertyReplacement===undefined) {
            continue;
        }
        
        //for each property replacement in that rule
        for (var property in cur_rule.propertyReplacement) {
            if (cur_rule.propertyReplacement.hasOwnProperty(property)) {
            	var replacementInfo = cur_rule.propertyReplacement[property];
            	var concreteType = replacementInfo[0];
            	var occurrenceCount = replacementInfo[1];
            	if (occurrenceCount===1) {
            		//do the replacement
					for (var j=0;j<cur_rule.rhs.length;j++) {
						var cellRow_rhs = cur_rule.rhs[j];
						for (var k=0;k<cellRow_rhs.length;k++) {
							var cell=cellRow_rhs[k];
							concretizePropertyInCell(cell, property, concreteType);
						}
					}
            	}
            }
        }
	}

	//if any properties remain on the RHSes, bleep loudly
	var rhsPropertyRemains = '';
	for (var i = 0; i < result.length; i++) {
		var cur_rule = result[i];
		delete result.propertyReplacement;
		for (var j = 0; j < cur_rule.rhs.length; j++) {
			var cur_rulerow = cur_rule.rhs[j];
			for (var k = 0; k < cur_rulerow.length; k++) {
				var cur_cell = cur_rulerow[k];
				var properties = getPropertiesFromCell(state, cur_cell);
				for (var prop_n = 0; prop_n < properties.length; prop_n++) {
					if (ambiguousProperties.hasOwnProperty(properties[prop_n])) {
						rhsPropertyRemains = properties[prop_n];
					}
				}
			}
		}
	}


	if (rhsPropertyRemains.length > 0) {
		logError('This rule has a property on the right-hand side, \"'+ rhsPropertyRemains + "\", that can't be inferred from the left-hand side.  (either for every property on the right there has to be a corresponding one on the left in the same cell, OR, if there's a single occurrence of a particular property name on the left, all properties of the same name on the right are assumed to be the same).",lineNumber);
	}

	return result;
}


function concretizeMovingRule(state, rule,lineNumber) {	

	var shouldremove;
	var result = [rule];
	var modified=true;
	while (modified) {
		modified = false;
		for (var i = 0; i < result.length; i++) {
			//only need to iterate through lhs
			var cur_rule = result[i];
			shouldremove = false;
			for (var j = 0; j < cur_rule.lhs.length; j++) {
				var cur_rulerow = cur_rule.lhs[j];
				for (var k = 0; k < cur_rulerow.length; k++) {
					var cur_cell = cur_rulerow[k];
					var movings = getMovings(state, cur_cell);
					if (movings.length > 0) {
						shouldremove = true;
						modified = true;

						//just do the base property, let future iterations take care of the others
						var cand_name = movings[0][0];
						var ambiguous_dir = movings[0][1];
						var concreteDirs = directionaggregates[ambiguous_dir];
						for (var l = 0; l < concreteDirs.length; l++) {
							var concreteDirection = concreteDirs[l];
							var newrule = deepCloneRule(cur_rule);

							newrule.movingReplacement={};
							for(var moveTerm in cur_rule.movingReplacement) {
								if (cur_rule.movingReplacement.hasOwnProperty(moveTerm)) {
									var moveDat = cur_rule.movingReplacement[moveTerm];
									newrule.movingReplacement[moveTerm] = [moveDat[0],moveDat[1],moveDat[3]];
								}
							}

							concretizeMovingInCell(newrule.lhs[j][k], ambiguous_dir, cand_name, concreteDirection);
							if (newrule.rhs.length>0) {
								concretizeMovingInCell(newrule.rhs[j][k], ambiguous_dir, cand_name, concreteDirection);//do for the corresponding rhs cell as well
							}
                            
                            if (newrule.movingReplacement[cand_name]===undefined) {
    							newrule.movingReplacement[cand_name]=[concreteDirection,1,ambiguous_dir];
                            } else {
    							newrule.movingReplacement[cand_name][1]=newrule.movingReplacement[cand_name][1]+1;                                
                            }

							result.push(newrule);
						}
					}
				}
			}
			if (shouldremove)
			{
				result.splice(i, 1);
				i--;
			}
		}
	}

    
	for (var i = 0; i < result.length; i++) {
        //for each rule
		var cur_rule = result[i];
        if (cur_rule.movingReplacement===undefined) {
            continue;
        }
        var ambiguous_movement_dict={};
        //strict first - matches movement direction to objects
        //for each property replacement in that rule
        for (var cand_name in cur_rule.movingReplacement) {
            if (cur_rule.movingReplacement.hasOwnProperty(cand_name)) {
            	var replacementInfo = cur_rule.movingReplacement[cand_name];
            	var concreteMovement = replacementInfo[0];
            	var occurrenceCount = replacementInfo[1];
            	var ambiguousMovement = replacementInfo[2];
            	if ((ambiguousMovement in ambiguous_movement_dict) || (occurrenceCount!==1)) {
            		ambiguous_movement_dict[ambiguousMovement] = "INVALID";
            	} else {
            		ambiguous_movement_dict[ambiguousMovement] = concreteMovement
            	}

            	if (occurrenceCount===1) {
            		//do the replacement
					for (var j=0;j<cur_rule.rhs.length;j++) {
						var cellRow_rhs = cur_rule.rhs[j];
						for (var k=0;k<cellRow_rhs.length;k++) {
							var cell=cellRow_rhs[k];
							concretizeMovingInCell(cell, ambiguousMovement, cand_name, concreteMovement);
						}
					}
            	}
            }
        }

        //for each ambiguous word, if there's a single ambiguous movement specified in the whole lhs, then replace that wholesale
        for(var ambiguousMovement in ambiguous_movement_dict) {
        	if (ambiguous_movement_dict.hasOwnProperty(ambiguousMovement) && ambiguousMovement!=="INVALID") {
        		concreteMovement = ambiguous_movement_dict[ambiguousMovement];

				for (var j=0;j<cur_rule.rhs.length;j++) {
					var cellRow_rhs = cur_rule.rhs[j];
					for (var k=0;k<cellRow_rhs.length;k++) {
						var cell=cellRow_rhs[k];
						concretizeMovingInCellByAmbiguousMovementName(cell, ambiguousMovement, concreteMovement);
					}
				}
        	}
        }
	}

	//if any properties remain on the RHSes, bleep loudly
	var rhsAmbiguousMovementsRemain = '';
	for (var i = 0; i < result.length; i++) {
		var cur_rule = result[i];
		delete result.movingReplacement;
		for (var j = 0; j < cur_rule.rhs.length; j++) {
			var cur_rulerow = cur_rule.rhs[j];
			for (var k = 0; k < cur_rulerow.length; k++) {
				var cur_cell = cur_rulerow[k];
				var movings = getMovings(state, cur_cell);
				if (movings.length > 0) {
					rhsAmbiguousMovementsRemain = movings[0][1];					
				}
			}
		}
	}


	if (rhsAmbiguousMovementsRemain.length > 0) {
		logError('This rule has an ambiguous movement on the right-hand side, \"'+ rhsAmbiguousMovementsRemain + "\", that can't be inferred from the left-hand side.  (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that)).",lineNumber);
	}

	return result;
}

function rephraseSynonyms(state,rule) {
	for (var i = 0; i < rule.lhs.length; i++) {
		var cellrow_l = rule.lhs[i];
		var cellrow_r = rule.rhs[i];
		for (var j = 0; j < cellrow_l.length; j++) {
			var cell_l = cellrow_l[j];
			for (var k = 1; k < cell_l.length; k += 2) {
				var name = cell_l[k];
				if (name in state.synonymsDict) {
					cell_l[k] = state.synonymsDict[cell_l[k]];
				}
			}
			if (rule.rhs.length>0) {
				var cell_r = cellrow_r[j];
				for (var k = 1; k < cell_r.length; k += 2) {
					var name = cell_r[k];
					if (name in state.synonymsDict) {
						cell_r[k] = state.synonymsDict[cell_r[k]];
					}
				}
			}
		}
	}
}

function atomizeAggregates(state, rule) {
	for (var i = 0; i < rule.lhs.length; i++) {
		var cellrow = rule.lhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			atomizeCellAggregates(state, cell,rule.lineNumber);
		}
	}
	for (var i = 0; i < rule.rhs.length; i++) {
		var cellrow = rule.rhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			atomizeCellAggregates(state, cell,rule.lineNumber);
		}
	}
}

function atomizeCellAggregates(state, cell, lineNumber) {
	for (var i = 0; i < cell.length; i += 2) {
		var dir =cell[i];
		var c = cell[i+1];
		if (c in state.aggregatesDict) {
			if (dir==='no') {
				logError("You cannot use 'no' to exclude the aggregate object " +c.toUpperCase()+" (defined using 'AND'), only regular objects, or properties (objects defined using 'OR').  If you want to do this, you'll have to write it out yourself the long way.",lineNumber);
			}
			var equivs = state.aggregatesDict[c];
			cell[i+1] = equivs[0];
			for (var j= 1; j < equivs.length; j++) {
				cell.push(cell[i]);//push the direction
				cell.push(equivs[j]);
			}
		}
	}
}

function convertRelativeDirsToAbsolute(rule) {
	var forward = rule.direction;
	for (var i = 0; i < rule.lhs.length; i++) {
		var cellrow = rule.lhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			absolutifyRuleCell(forward, cell);
		}
	}
	for (var i = 0; i < rule.rhs.length; i++) {
		var cellrow = rule.rhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			absolutifyRuleCell(forward, cell);
		}
	}
}

var relativeDirs = ['^','v','<','>','parallel','perpendicular'];//used to index the following
var relativeDict = {
	'right': ['up', 'down', 'left', 'right','horizontal','vertical'],
	'up': ['left', 'right', 'down', 'up','vertical','horizontal'],
	'down': ['right', 'left', 'up', 'down','vertical','horizontal'],
	'left': ['down', 'up', 'right', 'left','horizontal','vertical']
};

function absolutifyRuleCell(forward, cell) {
	for (var i = 0; i < cell.length; i += 2) {
		var c = cell[i];
		var index = relativeDirs.indexOf(c);
		if (index >= 0) {
			cell[i] = relativeDict[forward][index];		
		}
	}
}
/*
	direction mask
	UP parseInt('%1', 2);
	DOWN parseInt('0', 2);
	LEFT parseInt('0', 2);
	RIGHT parseInt('0', 2);
	?  parseInt('', 2);

*/

var dirMasks = {
	'up'	: parseInt('00001', 2),
	'down'	: parseInt('00010', 2),
	'left'	: parseInt('00100', 2),
	'right'	: parseInt('01000', 2),
	'moving': parseInt('01111', 2),
	'no'	: parseInt('00011', 2),
	'randomdir': parseInt('00101', 2),
	'random' : parseInt('10010',2),
	'action' : parseInt('10000', 2),
	'' : parseInt('00000',2)
};

function rulesToMask(state) {
	/*

	*/
	var layerCount = state.collisionLayers.length;
	var layerTemplate = [];
	for (var i = 0; i < layerCount; i++) {
		layerTemplate.push(null);
	}

	for (var i = 0; i < state.rules.length; i++) {
		var rule = state.rules[i];
		for (var j = 0; j < rule.lhs.length; j++) {
			var cellrow_l = rule.lhs[j];
			var cellrow_r = rule.rhs[j];
			for (var k = 0; k < cellrow_l.length; k++) {
				var cell_l = cellrow_l[k];
				var layersUsed_l = layerTemplate.concat([]);
				var objectsPresent = new BitVec(STRIDE_OBJ);
				var objectsMissing = new BitVec(STRIDE_OBJ);
				var anyObjectsPresent = [];
				var movementsPresent = new BitVec(STRIDE_MOV);
				var movementsMissing = new BitVec(STRIDE_MOV);

				var objectlayers_l = new BitVec(STRIDE_MOV);
				for (var l = 0; l < cell_l.length; l += 2) {
					var object_dir = cell_l[l];
					if (object_dir==='...') {
						objectsPresent = ellipsisPattern;
						if (cell_l.length!==2) {
							logError("You can't have anything in with an ellipsis. Sorry.",rule.lineNumber);
						} else if ((k===0)||(k===cellrow_l.length-1)) {
							logError("There's no point in putting an ellipsis at the very start or the end of a rule",rule.lineNumber);
						} else if (rule.rhs.length>0) {
							var rhscell=cellrow_r[k];
							if (rhscell.length!==2 || rhscell[0]!=='...') {
								logError("An ellipsis on the left must be matched by one in the corresponding place on the right.",rule.lineNumber);								
							}
						} 
						break;
					}  else if (object_dir==='random') {
						logError("'random' cannot be matched on the left-hand side, it can only appear on the right",rule.lineNumber);
						continue;
					}

					var object_name = cell_l[l + 1];
					var object = state.objects[object_name];
					var objectMask = state.objectMasks[object_name];
					if (object) {
						var layerIndex = object.layer|0;
					} else {
						var layerIndex = state.propertiesSingleLayer[object_name];
					}

					if (typeof(layerIndex)==="undefined"){
						logError("Oops!  " +object_name.toUpperCase()+" not assigned to a layer.",rule.lineNumber);
					}

					if (object_dir==='no') {
						objectsMissing.ior(objectMask);
					} else {
						var existingname = layersUsed_l[layerIndex];
						if (existingname !== null) {
							logError('Rule matches object types that can\'t overlap: "' + object_name.toUpperCase() + '" and "' + existingname.toUpperCase() + '".', rule.lineNumber);
						}

						layersUsed_l[layerIndex] = object_name;

						if (object) {
							objectsPresent.ior(objectMask);
							objectlayers_l.ishiftor(0x1f, 5*layerIndex);
						} else {
							anyObjectsPresent.push(objectMask);
						}

						if (object_dir==='stationary') {
							movementsMissing.ishiftor(0x1f, 5*layerIndex);
						} else {
							movementsPresent.ishiftor(dirMasks[object_dir], 5 * layerIndex);
						}
					}
				}

				if (rule.rhs.length>0) {
					var rhscell = cellrow_r[k];
					var lhscell = cellrow_l[k];
					if (rhscell[0]==='...' && lhscell[0]!=='...' ) {
						logError("An ellipsis on the right must be matched by one in the corresponding place on the left.",rule.lineNumber);								
					}
					for (var l=0;l<rhscell.length;l+=2) {
						var content=rhscell[l];
						if (content==='...') {
							if (rhscell.length!==2) {
								logError("You can't have anything in with an ellipsis. Sorry.",rule.lineNumber);							
							}
						}
					}
				}

				if (objectsPresent === ellipsisPattern) {
					cellrow_l[k] = ellipsisPattern;
					continue;
				} else {
					cellrow_l[k] = new CellPattern([objectsPresent, objectsMissing, anyObjectsPresent, movementsPresent, movementsMissing, null]);
				}

				if (rule.rhs.length===0) {
					continue;
				}

				var cell_r = cellrow_r[k];
				var layersUsed_r = layerTemplate.concat([]);

				var objectsClear = new BitVec(STRIDE_OBJ);
				var objectsSet = new BitVec(STRIDE_OBJ);
				var movementsClear = new BitVec(STRIDE_MOV);
				var movementsSet = new BitVec(STRIDE_MOV);

				var objectlayers_r = new BitVec(STRIDE_MOV);
				var randomMask_r = new BitVec(STRIDE_OBJ);
				var postMovementsLayerMask_r = new BitVec(STRIDE_MOV);
				var randomDirMask_r = new BitVec(STRIDE_MOV);
				for (var l = 0; l < cell_r.length; l += 2) {
					var object_dir = cell_r[l];
					var object_name = cell_r[l + 1];

					if (object_dir==='...') {
						logError("spooky ellipsis found! (should never hit this)");
						break;
					} else if (object_dir==='random') {
						if (object_name in state.objectMasks) {
							var mask = state.objectMasks[object_name];    
							randomMask_r.ior(mask);                      
						} else {
							logError('You want to spawn a random "'+object_name.toUpperCase()+'", but I don\'t know how to do that',rule.lineNumber);
						}
						continue;
					}

					var object = state.objects[object_name];
					var objectMask = state.objectMasks[object_name];
					if (object) {
						var layerIndex = object.layer|0;
					} else {
						var layerIndex = state.propertiesSingleLayer[object_name];
					}

					
					if (object_dir=='no') {
						objectsClear.ior(objectMask);
					} else {
						var existingname = layersUsed_r[layerIndex];
						if (existingname !== null) {
							logError('Rule matches object types that can\'t overlap: "' + object_name.toUpperCase() + '" and "' + existingname.toUpperCase() + '".', rule.lineNumber);
						}

						layersUsed_r[layerIndex] = object_name;

						if (object_dir.length>0) {
							postMovementsLayerMask_r.ishiftor(0x1f, 5*layerIndex);
						}

						var layerMask = state.layerMasks[layerIndex];

						if (object) {
							objectsSet.ibitset(object.id);
							objectsClear.ior(layerMask);
							objectlayers_r.ishiftor(0x1f, 5*layerIndex);
						} else {
							// shouldn't need to do anything here...
						}
						if (object_dir==='stationary') {
							movementsClear.ishiftor(0x1f, 5*layerIndex);
						} if (object_dir==='randomdir') {
							randomDirMask_r.ishiftor(dirMasks[object_dir], 5 * layerIndex);
						} else {						
							movementsSet.ishiftor(dirMasks[object_dir], 5 * layerIndex);
						};
					}
				}

				if (!(objectsPresent.bitsSetInArray(objectsSet.data))) {
					objectsClear.ior(objectsPresent); // clear out old objects
				}
				if (!(movementsPresent.bitsSetInArray(movementsSet.data))) {
					movementsClear.ior(movementsPresent); // ... and movements
				}

				for (var l = 0; l < layerCount; l++) {
					if (layersUsed_l[l] !== null && layersUsed_r[l] === null) {
						// a layer matched on the lhs, but not on the rhs
						objectsClear.ior(state.layerMasks[l]);
						postMovementsLayerMask_r.ishiftor(0x1f, 5*l);
					}
				}

				objectlayers_l.iclear(objectlayers_r);

				postMovementsLayerMask_r.ior(objectlayers_l);
				if (objectsClear || objectsSet || movementsClear || movementsSet || postMovementsLayerMask_r) {
					// only set a replacement if something would change
					cellrow_l[k].replacement = new CellReplacement([objectsClear, objectsSet, movementsClear, movementsSet, postMovementsLayerMask_r, randomMask_r, randomDirMask_r]);
				}
			}
		}
	}
}

function cellRowMasks(rule) {
	var ruleMasks=[];
	var lhs=rule[1];
	for (var i=0;i<lhs.length;i++) {
		var cellRow = lhs[i];
		var rowMask=new BitVec(STRIDE_OBJ);
		for (var j=0;j<cellRow.length;j++) {
			if (cellRow[j] === ellipsisPattern)
				continue;
			rowMask.ior(cellRow[j].objectsPresent);
		}
		ruleMasks.push(rowMask);
	}
	return ruleMasks;
}

function collapseRules(groups) {
	for (var gn = 0; gn < groups.length; gn++) {
		var rules = groups[gn];
		for (var i = 0; i < rules.length; i++) {
			var oldrule = rules[i];
			var newrule = [0,[],oldrule.rhs.length>0,oldrule.lineNumber/*ellipses,group number,rigid,commands,randomrule,[cellrowmasks]*/];
			var ellipses = [];
			for (var j=0;j<oldrule.lhs.length;j++) {
				ellipses.push(false);
			}

			newrule[0]=dirMasks[oldrule.direction];
			for (var j = 0; j < oldrule.lhs.length; j++) {
				var cellrow_l = oldrule.lhs[j];
				for (var k = 0; k < cellrow_l.length; k++) {
					if (cellrow_l[k] === ellipsisPattern) {
						if (ellipses[j]) {
							logError("You can't use two ellipses in a single cell match pattern.  If you really want to, please implement it yourself and send me a patch :) ", oldrule.lineNumber);
						} 
						ellipses[j]=true;
					}
				}
				newrule[1][j] = cellrow_l;
			}
			newrule.push(ellipses);
			newrule.push(oldrule.groupNumber);
			newrule.push(oldrule.rigid);
			newrule.push(oldrule.commands);
			newrule.push(oldrule.randomRule);
			newrule.push(cellRowMasks(newrule));
			rules[i] = new Rule(newrule);
		}
	}
	matchCache = {}; // clear match cache so we don't slowly leak memory
}

function ruleGroupRandomnessTest(ruleGroup) {
	if (ruleGroup.length === 0)
		return;
	var firstLineNumber = ruleGroup[0].lineNumber;
	for (var i=1;i<ruleGroup.length;i++) {
		var rule=ruleGroup[i];
		if (rule.lineNumber === firstLineNumber) // random [A | B] gets turned into 4 rules, skip
			continue;
		if (rule.randomRule) {
			logError("A rule-group can only be marked random by the first rule", rule.lineNumber);
		}
	}
}

function arrangeRulesByGroupNumber(state) {
	var aggregates = {};
	var aggregates_late = {};
	for (var i=0;i<state.rules.length;i++) {
		var rule = state.rules[i];
		var targetArray = aggregates;
		if (rule.late) {
			targetArray=aggregates_late;
		}

		if (targetArray[rule.groupNumber]==undefined) {
			targetArray[rule.groupNumber]=[];
		}
		targetArray[rule.groupNumber].push(rule);
	}

	var result=[];
	for (var groupNumber in aggregates) {
		if (aggregates.hasOwnProperty(groupNumber)) {
			var ruleGroup = aggregates[groupNumber];
			ruleGroupRandomnessTest(ruleGroup);
			result.push(ruleGroup);
		}
	}
	var result_late=[];
	for (var groupNumber in aggregates_late) {
		if (aggregates_late.hasOwnProperty(groupNumber)) {
			var ruleGroup = aggregates_late[groupNumber];
			ruleGroupRandomnessTest(ruleGroup);
			result_late.push(ruleGroup);
		}
	}
	state.rules=result;

	//check that there're no late movements with direction requirements on the lhs
	state.lateRules=result_late;
}


function checkNoLateRulesHaveMoves(state){
	for (var ruleGroupIndex=0;ruleGroupIndex<state.lateRules.length;ruleGroupIndex++) {
		var lateGroup = state.lateRules[ruleGroupIndex];
		for (var ruleIndex=0;ruleIndex<lateGroup.length;ruleIndex++) {
			var rule = lateGroup[ruleIndex];
			for (var cellRowIndex=0;cellRowIndex<rule.patterns.length;cellRowIndex++) {
				var cellRow_l = rule.patterns[cellRowIndex];
				for (var cellIndex=0;cellIndex<cellRow_l.length;cellIndex++) {
					var cellPattern = cellRow_l[cellIndex];
					if (cellPattern === ellipsisPattern) {
						continue;
					}
					var moveMissing = cellPattern.movementsMissing;
					var movePresent = cellPattern.movementsPresent;
					if (!moveMissing.iszero() || !movePresent.iszero()) {
						logError("Movements cannot appear in late rules.",rule.lineNumber);
						return;
					}

					if (cellPattern.replacement!=null) {
						var movementsClear = cellPattern.replacement.movementsClear;
						var movementsSet = cellPattern.replacement.movementsSet;

						if (!movementsClear.iszero() || !movementsSet.iszero()) {
							logError("Movements cannot appear in late rules.",rule.lineNumber);
							return;
						}
					}				
				}
			}
		}
	}
}

function generateRigidGroupList(state) {
	var rigidGroupIndex_to_GroupIndex=[];
	var groupIndex_to_RigidGroupIndex=[];
	var groupNumber_to_GroupIndex=[];
	var groupNumber_to_RigidGroupIndex=[];
	var rigidGroups=[];
	for (var i=0;i<state.rules.length;i++) {
		var ruleset=state.rules[i];
		var rigidFound=false;
		for (var j=0;j<ruleset.length;j++) {
			var rule=ruleset[j];
			if (rule.isRigid) {
				rigidFound=true;
			}
		}
		rigidGroups[i]=rigidFound;
		if (rigidFound) {
			var groupNumber=ruleset[0].groupNumber;
			groupNumber_to_GroupIndex[groupNumber]=i;
			var rigid_group_index = rigidGroupIndex_to_GroupIndex.length;
			groupIndex_to_RigidGroupIndex[i]=rigid_group_index;
			groupNumber_to_RigidGroupIndex[groupNumber]=rigid_group_index;
			rigidGroupIndex_to_GroupIndex.push(i);
		}
	}
	if (rigidGroupIndex_to_GroupIndex.length>30) {
		logError("There can't be more than 30 rigid groups (rule groups containing rigid members).",rules[0][0][3]);
	}

	state.rigidGroups=rigidGroups;
	state.rigidGroupIndex_to_GroupIndex=rigidGroupIndex_to_GroupIndex;
	state.groupNumber_to_RigidGroupIndex=groupNumber_to_RigidGroupIndex;
	state.groupIndex_to_RigidGroupIndex=groupIndex_to_RigidGroupIndex;
}

function getMaskFromName(state,name) {
	var objectMask=new BitVec(STRIDE_OBJ);
	if (name in state.objects) {
		var o=state.objects[name];
		objectMask.ibitset(o.id);
	}

	if (name in state.aggregatesDict) {
		var objectnames = state.aggregatesDict[name];
		for(var i=0;i<objectnames.length;i++) {
			var n=objectnames[i];
			var o = state.objects[n];
			objectMask.ibitset(o.id);
		}
	}

	if (name in state.propertiesDict) {
		var objectnames = state.propertiesDict[name];
		for(var i=0;i<objectnames.length;i++) {
			var n = objectnames[i];
			var o = state.objects[n];
			objectMask.ibitset(o.id);
		}
	}

	if (name in state.synonymsDict) {
		var n = state.synonymsDict[name];
		var o = state.objects[n];
		objectMask.ibitset(o.id);
	}

	if (objectMask.iszero()) {
		logErrorNoLine("error, didn't find any object called player, either in the objects section, or the legends section. there must be a player!");
	}
	return objectMask;
}

function generateMasks(state) {
	state.playerMask=getMaskFromName(state,'player');

	var layerMasks=[];
	var layerCount = state.collisionLayers.length;
	for (var layer=0;layer<layerCount;layer++){
		var layerMask=new BitVec(STRIDE_OBJ);
		for (var j=0;j<state.objectCount;j++) {
			var n=state.idDict[j];
			var o = state.objects[n];
			if (o.layer==layer) {
				layerMask.ibitset(o.id);
			}
		}
		layerMasks.push(layerMask);
	}
	state.layerMasks=layerMasks;

	var objectMask={};
	for(var n in state.objects) {
		if (state.objects.hasOwnProperty(n)) {
			var o = state.objects[n];
			objectMask[n] = new BitVec(STRIDE_OBJ);
			objectMask[n].ibitset(o.id);
		}
	}

	// Synonyms can depend on properties, and properties can depend on synonyms.
	// Process them in order by combining & sorting by linenumber.

	var synonyms_and_properties = state.legend_synonyms.concat(state.legend_properties);
	synonyms_and_properties.sort(function(a, b) {
		return a.lineNumber - b.lineNumber;
	});

	for (var i=0;i<synonyms_and_properties.length;i++) {
		var synprop = synonyms_and_properties[i];
		if (synprop.length == 2) {
			// synonym (a = b)
			objectMask[synprop[0]]=objectMask[synprop[1]];
		} else {
			// property (a = b or c)
			var val = new BitVec(STRIDE_OBJ);
			for (var j=1;j<synprop.length;j++) {
				var n = synprop[j];
				val.ior(objectMask[n]);
			}
			objectMask[synprop[0]]=val;
		}
	}

	state.objectMasks = objectMask;
}

function checkObjectsAreLayered(state) {
	for (var n in state.objects) {
		if (state.objects.hasOwnProperty(n)) {
			var found=false;
			for (var i=0;i<state.collisionLayers.length;i++) {
				var layer = state.collisionLayers[i];
				for (var j=0;j<layer.length;j++) {
					if (layer[j]===n) {
						found=true;
						break;
					}
				}
				if (found) {
					break;
				}
			}
			if (found===false) {
				var o = state.objects[n];
				logError('Object "' + n.toUpperCase() + '" has been defined, but not assigned to a layer.',o.lineNumber);
			}
		}
	}
}

function twiddleMetaData(state) {
	var newmetadata = {};
	for (var i=0;i<state.metadata.length;i+=2) {
		var key = state.metadata[i];
		var val = state.metadata[i+1];
		newmetadata[key]=val;
	}

	if (newmetadata.flickscreen!==undefined) {
		var val = newmetadata.flickscreen;
		var coords = val.split('x');
		var intcoords = [parseInt(coords[0]),parseInt(coords[1])];
		newmetadata.flickscreen=intcoords;
	}
	if (newmetadata.zoomscreen!==undefined) {
		var val = newmetadata.zoomscreen;
		var coords = val.split('x');
		var intcoords = [parseInt(coords[0]),parseInt(coords[1])];
		newmetadata.zoomscreen=intcoords;
	}

	state.metadata=newmetadata;	
}

function processWinConditions(state) {
//	[-1/0/1 (no,some,all),ob1,ob2] (ob2 is background by default)
	var newconditions=[]; 
	for (var i=0;i<state.winconditions.length;i++)  {
		var wincondition=state.winconditions[i];
		if (wincondition.length==0) {
			return;
		}
		var num=0;
		switch(wincondition[0]) {
			case 'no':{num=-1;break;}
			case 'all':{num=1;break;}
		}

		var lineNumber=wincondition[wincondition.length-1];

		var n1 = wincondition[1];
		var n2;
		if (wincondition.length==5) {
			n2 = wincondition[3];
		} else {
			n2 = 'background';
		}

		var mask1=0;
		var mask2=0;
		if (n1 in state.objectMasks) {
			mask1=state.objectMasks[n1];
		} else {
			logError('unwelcome term "' + n1 +'" found in win condition. Win conditions objects have to be objects or properties (defined using "or", in terms of other properties)', lineNumber);
		}
		if (n2 in state.objectMasks) {
			mask2=state.objectMasks[n2];
		} else {
			logError('unwelcome term "' + n2+ '" found in win condition. Win conditions objects have to be objects or properties (defined using "or", in terms of other properties)', lineNumber);
		}
		var newcondition = [num,mask1,mask2,lineNumber];
		newconditions.push(newcondition);
	}
	state.winconditions=newconditions;
}

function printCellRow(cellRow) {
	var result ="[ ";
	for (var i=0;i<cellRow.length;i++) {
		if (i>0) {
			result += "| ";
		}
		var cell = cellRow[i];
		for (var j=0;j<cell.length;j+=2) {
			var direction = cell[j];
			var object = cell[j+1]
			if (direction==="...") {
				result += direction+" ";
			} else {
				result += direction+" "+object+" ";
			}
		}		
	}
	result +="] ";
	return result;
}

function printRule(rule) {
	var result="(<a onclick=\"jumpToLine('"+ rule.lineNumber.toString() + "');\"  href=\"javascript:void(0);\">"+rule.groupNumber+"</a>) "+ rule.direction.toString().toUpperCase()+" ";
	if (rule.rigid) {
		result = "RIGID "+result+" ";
	}
	if (rule.randomRule) {
		result = "RANDOM "+result+" ";
	}
	if (rule.late) {
		result = "LATE "+result+" ";
	}
	for (var i=0;i<rule.lhs.length;i++) {
		var cellRow = rule.lhs[i];
		result = result + printCellRow(cellRow);
	}
	result = result + "-> ";
	for (var i=0;i<rule.rhs.length;i++) {
		var cellRow = rule.rhs[i];
		result = result + printCellRow(cellRow);
	}
	for (var i=0;i<rule.commands.length;i++) {
		var command = rule.commands[i];
		if (command.length===1) {
			result = result +command[0].toString();
		} else {
			result = result + '('+command[0].toString()+", "+command[1].toString()+') ';			
		}
	}
	//print commands next
	return result;
}
function printRules(state) {
	var output = "<br>Rule Assembly : ("+ state.rules.length +" rules )<br>===========<br>";
	var loopIndex = 0;
	var loopEnd = -1;
	for (var i=0;i<state.rules.length;i++) {
		var rule = state.rules[i];
		if (loopIndex < state.loops.length) {
			if (state.loops[loopIndex][0] < rule.lineNumber) {
				output += "STARTLOOP<br>";
				loopIndex++;
				if (loopIndex < state.loops.length) { // don't die with mismatched loops
					loopEnd = state.loops[loopIndex][0];
					loopIndex++;
				}
			}
		}
		if (loopEnd !== -1 && loopEnd < rule.lineNumber) {
			output += "ENDLOOP<br>";
			loopEnd = -1;
		}
		output += printRule(rule) +"<br>";
	}
	if (loopEnd !== -1) {	// no more rules after loop end
		output += "ENDLOOP<br>";
	}
	output+="===========<br>";
	consolePrint(output);
}

function removeDuplicateRules(state) {
	console.log("rule count before = " +state.rules.length);
	var record = {};
	var newrules=[];
	var lastgroupnumber=-1;
	for (var i=state.rules.length-1;i>=0;i--) {
		var r = state.rules[i];
		var groupnumber = r.groupNumber;
		if (groupnumber!==lastgroupnumber) {
			record={};
		}
		var r_string=printRule(r);
		if (record.hasOwnProperty(r_string)) {
			state.rules.splice(i,1);
		} else {
			record[r_string]=true;
		}
		lastgroupnumber=groupnumber;
	}
	console.log("rule count after = " +state.rules.length);
}
function generateLoopPoints(state) {
	var loopPoint={};
	var loopPointIndex=0;
	var outside=true;
	var source=0;
	var target=0;
	if (state.loops.length%2===1) {
		logErrorNoLine("have to have matching number of  'startLoop' and 'endLoop' loop points.");
	}

	for (var j=0;j<state.loops.length;j++) {
		var loop = state.loops[j];
		for (var i=0;i<state.rules.length;i++) {
			var ruleGroup = state.rules[i];

			var firstRule = ruleGroup[0];			
			var lastRule = ruleGroup[ruleGroup.length-1];

			var firstRuleLine = firstRule.lineNumber;
			var lastRuleLine = lastRule.lineNumber;

			if (outside) {
				if (firstRuleLine>=loop[0]) {
					target=i;
					outside=false;
					if (loop[1]===-1) {
						logErrorNoLine("Need have to have matching number of  'startLoop' and 'endLoop' loop points.");						
					}
					break;
				}
			} else {
				if (firstRuleLine>=loop[0]) {
					source = i-1;		
					loopPoint[source]=target;
					outside=true;
					if (loop[1]===1) {
						logErrorNoLine("Need have to have matching number of  'startLoop' and 'endLoop' loop points.");						
					}
					break;
				}
			}
		}
	}
	if (outside===false) {
		var source = state.rules.length;
		loopPoint[source]=target;
	} else {
	}
	state.loopPoint=loopPoint;

	loopPoint={};
	outside=true;
	for (var j=0;j<state.loops.length;j++) {
		var loop = state.loops[j];
		for (var i=0;i<state.lateRules.length;i++) {
			var ruleGroup = state.lateRules[i];

			var firstRule = ruleGroup[0];			
			var lastRule = ruleGroup[ruleGroup.length-1];

			var firstRuleLine = firstRule.lineNumber;
			var lastRuleLine = lastRule.lineNumber;

			if (outside) {
				if (firstRuleLine>=loop[0]) {
					target=i;
					outside=false;
					if (loop[1]===-1) {
						logErrorNoLine("Need have to have matching number of  'startLoop' and 'endLoop' loop points.");						
					}
					break;
				}
			} else {
				if (firstRuleLine>=loop[0]) {
					source = i-1;		
					loopPoint[source]=target;
					outside=true;
					if (loop[1]===1) {
						logErrorNoLine("Need have to have matching number of  'startLoop' and 'endLoop' loop points.");						
					}
					break;
				}
			}
		}
	}
	if (outside===false) {
		var source = state.lateRules.length;
		loopPoint[source]=target;
	} else {
	}
	state.lateLoopPoint=loopPoint;
}

var soundEvents = ["titlescreen", "startgame", "endgame", "startlevel","undo","restart","endlevel","showmessage","closemessage","sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10"];
var soundMaskedEvents =["create","destroy","move","cantmove","action"];
var soundVerbs = soundEvents.concat(soundMaskedEvents);


function validSeed (seed ) {
	return /^\s*\d+\s*$/.exec(seed)!==null;
}


var soundDirectionIndicatorMasks = {
	'up'			: parseInt('00001', 2),
	'down'			: parseInt('00010', 2),
	'left'			: parseInt('00100', 2),
	'right'			: parseInt('01000', 2),
	'horizontal'	: parseInt('01100', 2),
	'vertical'		: parseInt('00011', 2),
	'orthogonal'	: parseInt('01111', 2),
	'___action____'		: parseInt('10000', 2)
};

var soundDirectionIndicators = ["up","down","left","right","horizontal","vertical","orthogonal","___action____"];


function generateSoundData(state) {
	var sfx_Events={};
	var sfx_CreationMasks=[];
	var sfx_DestructionMasks=[];
	var sfx_MovementMasks=[];
	var sfx_MovementFailureMasks=[];

	for (var i=0;i<state.sounds.length;i++) {
		var sound=state.sounds[i];
		if (sound.length<=1) {
			continue;
		}
		var lineNumber=sound[sound.length-1];

		if (sound.length===2){			
			logError('incorrect sound declaration.',lineNumber);
			continue;
		}

		if (soundEvents.indexOf(sound[0])>=0) {
			if (sound.length>4) {
				logError("too much stuff to define a sound event",lineNumber);
			}
			var seed = sound[1];
			if (validSeed(seed)) {
				if (sfx_Events[sound[0]]!==undefined){
					logError(sound[0].toUpperCase()+" already declared.",lineNumber);				
				} 
				sfx_Events[sound[0]]=sound[1];
			} else {
				logError("Expecting sfx data, instead found \""+sound[1]+"\".",lineNumber);				
			}
		} else {
			var target = sound[0].trim();
			var verb = sound[1].trim();
			var directions = sound.slice(2,sound.length-2);
			if (directions.length>0&&(verb!=='move'&&verb!=='cantmove')) {
				logError('incorrect sound declaration.',lineNumber);
			}

			if (verb==='action') {
				verb='move';
				directions=['___action____'];
			}

			if (directions.length==0) {
				directions=["orthogonal"];
			}
			var seed = sound[sound.length-2];

			if (target in state.aggregatesDict) {
				logError('cannot assign sound fevents to aggregate objects (declared with "and"), only to regular objects, or properties, things defined in terms of "or" ("'+target+'").',lineNumber);
			}
			else if (target in state.objectMasks) {

			} else {
				logError('Object "'+ target+'" not found.',lineNumber);
			}

			var objectMask = state.objectMasks[target];

			var directionMask=0;
			for (var j=0;j<directions.length;j++) {
				directions[j]=directions[j].trim();
				var direction=directions[j];
				if (soundDirectionIndicators.indexOf(direction)===-1) {
					logError('Was expecting a direction, instead found "'+direction+'".',lineNumber);
				} else {
					var soundDirectionMask = soundDirectionIndicatorMasks[direction];
					directionMask |= soundDirectionMask;
				}
			}


			var targets=[target];
			var modified=true;
			while(modified) {
				modified=false;
				for (var k=0;k<targets.length;k++) {
					var t = targets[k];
					if (t in state.synonymsDict) {
						targets[k]=state.synonymsDict[t];
						modified=true;
					} else if (t in state.propertiesDict) {
						modified=true;
						var props = state.propertiesDict[t];
						targets.splice(k,1);
						k--;
						for (var l=0;l<props.length;l++) {
							targets.push(props[l]);
						}
					}
				}
			}

			if (verb==='move' || verb==='cantmove') {
				for (var j=0;j<targets.length;j++) {
					var targetName = targets[j];
					var targetDat = state.objects[targetName];
					var targetLayer = targetDat.layer;
					var shiftedDirectionMask = new BitVec(STRIDE_MOV);
					shiftedDirectionMask.ishiftor(directionMask, 5*targetLayer);

					var o = {
						objectMask: objectMask,
						directionMask: shiftedDirectionMask,
						seed: seed
					};

					if (verb==='move') {
						sfx_MovementMasks.push(o);						
					} else {
						sfx_MovementFailureMasks.push(o);
					}
				}
			}


			if (!validSeed(seed)) {
				logError("Expecting sfx data, instead found \""+seed+"\".",lineNumber);	
			}

			var targetArray;
			switch(verb) {
				case "create": {
					var o = {
						objectMask: objectMask,
						seed: seed
					}
					sfx_CreationMasks.push(o);
					break;
				}
				case "destroy": {
					var o = {
						objectMask: objectMask,
						seed: seed
					}
					sfx_DestructionMasks.push(o);
					break;
				}
			}
		}
	}

	state.sfx_Events = sfx_Events;
	state.sfx_CreationMasks = sfx_CreationMasks;
	state.sfx_DestructionMasks = sfx_DestructionMasks;
	state.sfx_MovementMasks = sfx_MovementMasks;
	state.sfx_MovementFailureMasks = sfx_MovementFailureMasks;
}


function formatHomePage(state){
	if ('background_color' in state.metadata) {
		state.bgcolor=colorToHex(colorPalette,state.metadata.background_color);
	} else {
		state.bgcolor="#000000";
	}
	if ('text_color' in state.metadata) {
		state.fgcolor=colorToHex(colorPalette,state.metadata.text_color);
	} else {
		state.fgcolor="#FFFFFF";
	}
	
	if (isColor(state.fgcolor)===false ){
		logError("text_color in incorrect format - found "+state.fgcolor+", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').")
	}
	if (isColor(state.bgcolor)===false ){
		logError("background_color in incorrect format - found "+state.bgcolor+", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').")
	}

	if (canSetHTMLColors) {
		
		if ('background_color' in state.metadata)  {
			document.body.style.backgroundColor=state.bgcolor;
		}
		
		if ('text_color' in state.metadata) {
			var separator = document.getElementById("separator");
			if (separator!=null) {
			   separator.style.color = state.fgcolor;
			}
			
			var h1Elements = document.getElementsByTagName("a");
			for(var i = 0; i < h1Elements.length; i++) {
			   h1Elements[i].style.color = state.fgcolor;
			}

			var h1Elements = document.getElementsByTagName("h1");
			for(var i = 0; i < h1Elements.length; i++) {
			   h1Elements[i].style.color = state.fgcolor;
			}
		}
	}

	if ('homepage' in state.metadata) {
		var url = state.metadata['homepage'];
		url=url.replace("http://","");
		url=url.replace("https://","");
		state.metadata['homepage']=url;
	}
}

var MAX_ERRORS=5;
function loadFile(str) {
	window.console.log('loadFile');

	var processor = new codeMirrorFn();
	var state = processor.startState();

	var lines = str.split('\n');
	for (var i = 0; i < lines.length; i++) {
		var line = lines[i];
		state.lineNumber = i + 1;
		var ss = new CodeMirror.StringStream(line, 4);
		do {
			processor.token(ss, state);

			if (errorCount>MAX_ERRORS) {
				consolePrint("too many errors, aborting compilation");
				return;
			}
		}		
		while (ss.eol() === false);
	}

	delete state.lineNumber;

	generateExtraMembers(state);
	generateMasks(state);
	levelsToArray(state);
	rulesToArray(state);

	removeDuplicateRules(state);

	if (debugMode) {
		printRules(state);
	}

	rulesToMask(state);
	arrangeRulesByGroupNumber(state);
	collapseRules(state.rules);
	collapseRules(state.lateRules);

	checkNoLateRulesHaveMoves(state);

	generateRigidGroupList(state);

	processWinConditions(state);
	checkObjectsAreLayered(state);

	twiddleMetaData(state);

	generateLoopPoints(state);

	generateSoundData(state);

	formatHomePage(state);

	delete state.commentLevel;
	delete state.names;
	delete state.abbrevNames;
	delete state.objects_candname;
	delete state.objects_section;
	delete state.objects_spritematrix;
	delete state.section;
	delete state.subsection;
	delete state.tokenIndex;
	delete state.visitedSections;
	delete state.loops;
	/*
	var lines = stripComments(str);
	window.console.log(lines);
	var sections = generateSections(lines);
	window.console.log(sections);
	var sss = generateSemiStructuredSections(sections);*/
	return state;
}

var ifrm;
function compile(command,text,randomseed) {
	matchCache={};
	forceRegenImages=true;
	if (command===undefined) {
		command = ["restart"];
	}
	if (randomseed===undefined) {
		randomseed = null;
	}
	lastDownTarget=canvas;	

	if (text===undefined){
		var code = window.form1.code;

		var editor = code.editorreference;

		text = editor.getValue()+"\n";
	}
	if (canDump===true) {
		compiledText=text;
	}

	errorCount = 0;
	compiling = true;
	errorStrings = [];
	consolePrint('=================================');
	try
	{
		var state = loadFile(text);
//		consolePrint(JSON.stringify(state));
	} finally {
		compiling = false;
	}
	if (errorCount>MAX_ERRORS) {
		return;
	}
	/*catch(err)
	{
		if (anyErrors===false) {
			logErrorNoLine(err.toString());
		}
	}*/

	if (errorCount>0) {
		consoleError('<span class="systemMessage">Errors detected during compilation, the game may not work correctly.</span>');
	}
	else {
		var ruleCount=0;
		for (var i=0;i<state.rules.length;i++) {
			ruleCount+=state.rules[i].length;
		}
		for (var i=0;i<state.lateRules.length;i++) {
			ruleCount+=state.lateRules[i].length;
		}
		if (command[0]=="restart") {
			consolePrint('<span class="systemMessage">Successful Compilation, generated ' + ruleCount + ' instructions.</span>');
		} else {
			consolePrint('<span class="systemMessage">Successful live recompilation, generated ' + ruleCount + ' instructions.</span>');

		}
	}
	setGameState(state,command,randomseed);

	clearInputHistory();

	consoleCacheDump();
}



function qualifyURL(url) {
	var a = document.createElement('a');
	a.href = url;
	return a.href;
}
