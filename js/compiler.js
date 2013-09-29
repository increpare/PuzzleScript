'use strict';


function isColor(str) {
	str = str.trim();
	if (str in colorPalettes.arnecolors)
		return true;
	if (/(^#[0-9A-F]{6}$)|(^#[0-9A-F]{3}$)/i.test(str))
		return true;

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

function generateExtraMembers(state) {

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
	if (state.objectCount > 32)
	{
		logError('Cannot have more than 31 objects defined, you have '+ state.objects.length, state.objects[0].lineNumber);
	}

	//calculate blank mask template
	var layerCount = state.collisionLayers.length;
	var blankMask = [];
	for (var i = 0; i < layerCount; i++) {
		blankMask.push(-1);
	}

	//get colorpalette name
	debugMode=false;
	var colorPalette=colorPalettes.arnecolors;
	for (var i=0;i<state.metadata.length;i+=2){
		var key = state.metadata[i];
		var val = state.metadata[i+1];
		if (key==='color_palette') {
			if (colorPalettes[val]===undefined) {
				logError('palette "'+val+'" not found, defaulting to arnecolors.',0);
			}else {
				colorPalette=colorPalettes[val];
			}
		} else if (key==='debug') {
			debugMode=true;
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
	      		c = colorToHex(colorPalette,c);
				o.colors[i] = c;
				if (isColor(c) === false) {
					logError('Invalid color specified for object "' + n + '", namely "' + o.colors[i] + '".', o.lineNumber + 1);
				}
			}
		}
	}

	//generate sprite matrix
	for (var n in state.objects) {
	      if (state.objects.hasOwnProperty(n)) {
	      	var o = state.objects[n];
	      	if (o.colors.length==0) {
	      		debug.logError('color not specified for object "' + n +'".',o.lineNumber);
	      		o.colors=["#ff00ff"];
	      	}
			if (o.spritematrix.length===0) {
				o.spritematrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
			} else {
				o.spritematrix = generateSpriteMatrix(o.spritematrix);
			}
		}
	}


	//calculate glyph dictionary
	var glyphDict = {};
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
                        logError(
                            'Trying to create an aggregate object (defined in the legend) with both) "'
                            + n.toUpperCase() + '" and "' + state.idDict[mask[o.layer]].toUpperCase() + '", which are on the same layer and therefore can\'t coexist.',
                            dat.lineNumber
                            );
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
		} else {
			synonymsDict[key] = value;		
		}
	}
	state.synonymsDict = synonymsDict;


	//set default background object
	var backgroundid;
	var backgroundlayer;
	if (state.objects.background===undefined) {
		if ('background' in state.synonymsDict) {
			var n = state.synonymsDict['background'];
			var o = state.objects[n];
			backgroundid = o.id;
			backgroundlayer = o.layer;
		} else if ('background' in state.propertiesDict) {
			var n = state.propertiesDict['background'][0];
			var o = state.objects[n];
			backgroundid = o.id;
			backgroundlayer = o.layer;
		}else if ('background' in state.aggregatesDict) {
			var o=state.idDict[0];
			backgroundid=o.id;
			backgroundlayer=o.layer;
			logError("background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or').");
		} else {
			var o=state.idDict[0];
			backgroundid=o.id;
			backgroundlayer=o.layer;
			logError("you have to define something to be the background");
		}
	} else {
		backgroundid = state.objects.background.id;
		backgroundlayer = state.objects.background.layer;
	}
	state.backgroundid=backgroundid;
	state.backgroundlayer=backgroundlayer;
}


function calcLevelBackgroundMask(state,dat) {
	var backgroundMask = state.layerMasks[state.backgroundlayer];
	for (var i=0;i<dat.length;i++) {
		var val=dat[i];
		var masked = val&backgroundMask;
		if (masked!==0) {
			return masked;
		}
	}
	return 1<<state.backgroundid;
}

//also assigns glyphDict
function levelsToArray(state) {
	var levels = state.levels;
	var processedLevels = [];
	var backgroundlayer=state.backgroundlayer;
	var backgroundid=state.backgroundid;
	var backgroundLayerMask = state.layerMasks[backgroundlayer];

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
			var dat = [];
			var o = {
				lineNumber:level[0],
				w: level[1].length,
				h: level.length-1,
				layerCount: state.collisionLayers.length
			};

			for (var i = 0; i < o.w; i++) {
				for (var j = 0; j < o.h; j++) {
					var ch = level[j+1].charAt(i);
					if (ch.length==0) {
						ch=level[j+1].charAt(level[j+1].length-1);
					}
					var maskint = 0;
					var mask = state.glyphDict[ch];

					if (mask == undefined) {
						if (state.propertiesDict[ch]===undefined) {
							logError('Error, symbol "' + ch + '", used in map, not found.', state.lineNumber);
						} else {
							logError('Error, symbol "' + ch + '" is defined using \'or\', and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of \'and\'?', state.lineNumber);							
						}

					}

					mask = mask.concat([]);					
					for (var z = 0; z < o.layerCount; z++) {
						if (mask[z]>=0) {
							maskint = maskint | (1 << mask[z]);
						}
//						dat.push(mask[z]);
					}
					dat.push(maskint);
				}
			}
			o.dat = dat;

			var levelBackgroundMask = calcLevelBackgroundMask(state,o.dat);
			for (var i=0;i<o.dat.length;i++)
			{
				var val = o.dat[i];
				if ((val&backgroundLayerMask)===0) {
					o.dat[i]=val|levelBackgroundMask;
				}
				
			}
			processedLevels.push(o);
		}

	}

	state.levels = processedLevels;
}

var directionaggregates = {
	'horizontal' : ['left', 'right'],
	'vertical' : ['up', 'down'],
	'moving' : ['up', 'down', 'left', 'right'],
	'perpendicular' : ['^','v'],
	'parallel' : ['<','>']
};

var simpleAbsoluteDirections = ['up', 'down', 'left', 'right'];
var simpleRelativeDirections = ['^', 'v', '<', '>'];
var reg_directions_only = /^(\>|\<|\^|v|up|down|left|right|moving|stationary|no|randomdir|random|horizontal|vertical|perpendicular|parallel)$/;
//redclareing here, i don't know why
var commandwords = ["sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10","cancel","checkpoint","restart","win","message"];

function processRuleString(line, state, lineNumber,curRules) 
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

	if (tokens.length===2) {
		if (tokens[0]==="["&&tokens[1]==="[" ) {
			rule_line = {
				bracket: 1
			}
			return rule_line;
		} else if (tokens[0]==="]"&&tokens[1]==="]" ) {
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
				} else if (simpleAbsoluteDirections.indexOf(token) >= 0) {
					directions.push(token);
				} else if (simpleRelativeDirections.indexOf(token) >= 0) {
					logError('You cannot use relative directions (\"^v<>\") to indicate in which direction(s) a rule applies.  Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions', lineNumber);
				} else if (token == '[') {
					if (directions.length == 0) {
						directions = directions.concat(directionaggregates['moving']);
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
						var messagetokens = tokens.slice(i+1);
						var messagestring = messagetokens.join(' ');
						commands.push([token,messagestring]);
						i=tokens.length;
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
		commands:commands
	};

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
		commands:rule.commands
	};
	return clonedRule;
}

function rulesToArray(state) {


	var oldrules = state.rules;
	var rules = [];
	var loops=[];
	for (var i = 0; i < oldrules.length; i++) {
		var lineNumber = oldrules[i][1];
		var newrule = processRuleString(oldrules[i][0], state, lineNumber,rules);
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
			if (dir in directionaggregates) {
				var dirs = directionaggregates[dir];
				for (var k = 0; k < dirs.length; k++) {
					var modifiedrule = {
						direction: dirs[k],
						lhs: deepCloneHS(rule.lhs),
						rhs: deepCloneHS(rule.rhs),
						lineNumber: rule.lineNumber,
						late: rule.late,
						rigid: rule.rigid,
						groupNumber: rule.groupNumber,
						commands:rule.commands
					};
					rules2.push(modifiedrule);
				}
			} else {
				var modifiedrule = {
					direction: dir,
					lhs: deepCloneHS(rule.lhs),
					rhs: deepCloneHS(rule.rhs),
					lineNumber: rule.lineNumber,
					late: rule.late,
					rigid: rule.rigid,
					groupNumber: rule.groupNumber,
					commands:rule.commands
				};
				rules2.push(modifiedrule);
			}
		}
	}

	//remove relative directions
	for (var i = 0; i < rules2.length; i++) {
		convertRelativeDirsToAbsolute(rules2[i]);
	}

/*
	//optional
	//replace up/left rules with their down/right equivalents
	for (var i = 0; i < rules2.length; i++) {
		rewriteUpLeftRules(rules2[i]);
	}
*/
	//replace aggregates with what they mean
	for (var i = 0; i < rules2.length; i++) {
		atomizeAggregates(state, rules2[i]);
	}

	//replace aggregates with what they mean
	for (var i = 0; i < rules2.length; i++) {
		rephraseSynonyms(state, rules2[i]);
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

function rewriteUpLeftRules(rule) {
	if (rule.direction == 'up') {
		rule.direction = 'down';
	} else if (rule.direction == 'left') {
		rule.direction = 'right';
	} else {
		return;
	}

	for (var i = 0; i < rule.lhs.length; i++) {
		var cellrow_l = rule.lhs[i].reverse();
		var cellrow_r = rule.rhs[i].reverse();
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


	var modified = true;
	while(modified) {
		modified=false;
		for (var i = 0; i < rule.lhs.length; i++) { 
			var cur_cellrow_l = rule.lhs[i];
			var cur_cellrow_r = rule.lhs[i];
			for (var j=0;j<cur_cellrow_l.length;j++) {
				cur_cellrow_l[j] = expandNoPrefixedProperties(state,cur_cellrow_l[j]);
				cur_cellrow_r[j] = expandNoPrefixedProperties(state,cur_cellrow_r[j]);
			}
		}
	}

	var shouldremove;
	var result = [rule];
	modified=true;
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
					var properties = getPropertiesFromCell(state, cur_cell);
					if (properties.length > 0) {
						shouldremove = true;
						modified = true;

						//just do the base property, let future iterations take care of the others
						var property = properties[0];
						var aliases = state.propertiesDict[property];

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
				if (properties.length > 0) {
					rhsPropertyRemains = properties[0];					
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
							concretizeMovingInCell(newrule.rhs[j][k], ambiguous_dir, cand_name, concreteDirection);//do for the corresponding rhs cell as well
                            
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

function atomizeAggregates(state,rule) {
	for (var i = 0; i < rule.lhs.length; i++) {
		var cellrow = rule.lhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			atomizeCellAggregates(state, cell);
		}
	}
	for (var i = 0; i < rule.rhs.length; i++) {
		var cellrow = rule.rhs[i];
		for (var j = 0; j < cellrow.length; j++) {
			var cell = cellrow[j];
			atomizeCellAggregates(state, cell);
		}
	}
}

function atomizeCellAggregates(state, cell) {
	for (var i = 1; i < cell.length; i += 2) {
		var c = cell[i];
		if (c in state.aggregatesDict) {
			var equivs = state.aggregatesDict[c];
			cell[i] = equivs[0];
			for (var i = 1; i < equivs.length; i++) {
				cell.push(cell[i - 1]);//push the direction
				cell.push(equivs[i]);
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
	'action' : parseInt('10000', 2)
};


function rulesToMask(state) {
	/*

	*/
	var layerCount = state.collisionLayers.length;
	var maskTemplate = [];
	for (var i = 0; i < layerCount; i++) {
		maskTemplate.push(-2);
		maskTemplate.push(-2);
	}

	for (var i = 0; i < state.rules.length; i++) {
		var rule = state.rules[i];
		for (var j = 0; j < rule.lhs.length; j++) {
			var cellrow_l = rule.lhs[j];
			var cellrow_r = rule.rhs[j];
			for (var k = 0; k < cellrow_l.length; k++) {
				var cell_l = cellrow_l[k];
				var mask_l = maskTemplate.concat([]);
				var forcemask_l = 0;
				var cellmask_l = 0;
				var nonExistenceMask_l = 0;
				var moveNonExistenceMask_l = 0;
				var stationaryMask_l = 0;
				for (var l = 0; l < cell_l.length; l += 2) {
					var object_dir = cell_l[l];
					if (object_dir==='...') {
//						mask_l[ 2 * layerIndex + 0 ] = ellipsisDirection;
//						mask_l[ 2 * layerIndex + 1 ] = 0;
						cellmask_l = ellipsisDirection;
						forcemask_l = ellipsisDirection;
						if (cell_l.length!==2) {
							logError("You can't have anything in with an ellipsis. Sorry.",rule.lineNumber);
						} else {
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
					var layerIndex = object.layer;
					var object_id = object.id;
					var layerMask = state.layerMasks[layerIndex];

					if (object_dir==='no') {
						nonExistenceMask_l = nonExistenceMask_l | (1<<object_id );
					} else {
						var targetobjectid = mask_l[2 * layerIndex + 1];
						if (targetobjectid > -2) {
							var existingname = state.idDict[targetobjectid];
							logError('Rule matches object types that can\'t overlap: "' + object_name.toUpperCase() + '" and "' + existingname.toUpperCase() + '".', rule.lineNumber);
						}

						mask_l[2 * layerIndex + 0] = object_dir;
						mask_l[2 * layerIndex + 1] = object_id;

						cellmask_l = cellmask_l | (1 << object_id);
						if (object_dir==='stationary') {
							stationaryMask_l = stationaryMask_l | ((1+2+4+8+16)<<(5*layerIndex));
						} else {
							var forcemask = (dirMasks[object_dir] << (5 * layerIndex));
							forcemask_l = forcemask_l | forcemask;	
							if (forcemask!==0) {	
								moveNonExistenceMask_l = moveNonExistenceMask_l | ((1+2+4+8+16)<<(5*layerIndex));							
							}
						}
					}
				}
				cellrow_l[k] = [forcemask_l, cellmask_l,nonExistenceMask_l,moveNonExistenceMask_l,stationaryMask_l];
				//cellrow_l[k]=mask_l;

				if (rule.rhs.length===0) {
					continue;
				}
				var cell_r = cellrow_r[k];
				var mask_r = maskTemplate.concat([]);
				var forcemask_r = 0;				
				var cellmask_r = 0;
				var randomMask_r = 0;
				var nonExistenceMask_r = 0;
				var postMovementsLayerMask_r = 0;
				var stationaryMask_r = 0;
				for (var l = 0; l < cell_r.length; l += 2) {
					var object_dir = cell_r[l];
					if (object_dir==='...') {
//						mask_r[ 2 * layerIndex + 0 ] = ellipsisDirection;
//						mask_r[ 2 * layerIndex + 1 ] = 0;
						cellmask_r = ellipsisDirection;
						forcemask_r = ellipsisDirection;
						break;
					}  else if (object_dir==='random') {
						var object_name = cell_r[l+1];
						if (object_name in state.objectMasks) {
							var mask = state.objectMasks[object_name];                            
                            randomMask_r = randomMask_r | mask;
                            //forcemask_r = randomEntityMask;
//                            nonExistenceMask_r = 0; //don't know why i had this                           
						} else {
							logError('You want to spawn a random "'+object_name.toUpperCase()+'", but I don\'t know how to do that',rule.lineNumber);
						}
						break;
					}

					var object_name = cell_r[l + 1];
					var object = state.objects[object_name];
					var layerIndex = object.layer;
					var object_id = object.id;

					
					if (object_dir=='no') {
						nonExistenceMask_r = nonExistenceMask_r | (1<<object_id );
					} else {
						var targetobjectid = mask_r[2 * layerIndex + 1];
						if (targetobjectid > -2) {
							var existingname = state.idDict[targetobjectid];
							logError('Rule matches object types that can\'t overlap: "' + object_name.toUpperCase() + '" and "' + existingname.toUpperCase() + '".', rule.lineNumber);
						}

						var layerMask = state.layerMasks[layerIndex];

						mask_r[2 * layerIndex + 0] = object_dir;
						mask_r[2 * layerIndex + 1] = object_id;

						postMovementsLayerMask_r = postMovementsLayerMask_r | ((1+2+4+8+16)<<(5*layerIndex));
						cellmask_r = cellmask_r | (1 << object_id);
						if (object_dir==='stationary') {
							stationaryMask_r = stationaryMask_r | ((1+2+4+8+16)<<(5*layerIndex));
						} else {
							forcemask_r = forcemask_r | (dirMasks[object_dir] << (5 * layerIndex));
						}
						nonExistenceMask_r = nonExistenceMask_r | layerMask;
					}
				}
				cellrow_r[k] = [forcemask_r, cellmask_r, nonExistenceMask_r,postMovementsLayerMask_r,stationaryMask_r,randomMask_r];
			}
		}
	}
}

function collapseRules(state) {
	for (var i = 0; i < state.rules.length; i++)
	{
		var oldrule = state.rules[i];
		var newrule = [0,[],[],oldrule.lineNumber,oldrule.late/*ellipses,group number,rigid,commands,commandsonly*/];
		var ellipses = [];
		for (var j=0;j<oldrule.lhs.length;j++) {
			ellipses.push(false);
		}

		newrule[0]=dirMasks[oldrule.direction];
		for (var j = 0; j < oldrule.lhs.length; j++) {
			var cellrow_l = oldrule.lhs[j];
			var cellrow_r = oldrule.rhs[j];
			var newcellrow_l = [];
			var newcellrow_r = [];
			for (var k = 0; k < cellrow_l.length; k++) {
				var oldcellmask_l = cellrow_l[k];
				if (oldcellmask_l[0]===ellipsisDirection) {
					if (ellipses[j]) {
						logError("You can't use two ellipses in a single cell match pattern.  If you really want to, please implement it yourself and send me a patch :) ", oldrule.lineNumber);
					} 
					ellipses[j]=true;
				}
				newcellrow_l[k * 6] = oldcellmask_l[0];
				newcellrow_l[k * 6 + 1] = oldcellmask_l[1];
				newcellrow_l[k * 6 + 2] = oldcellmask_l[2];//nonexistence
				newcellrow_l[k * 6 + 3] = oldcellmask_l[3];//movenonexistence
				newcellrow_l[k * 6 + 4] = oldcellmask_l[4];//stationarymask
				newcellrow_l[k * 6 + 5]  = 0;//unassigned

				if (oldrule.rhs.length>0) {
					var oldcellmask_r = cellrow_r[k];
					newcellrow_r[k * 6] = oldcellmask_r[0];
					newcellrow_r[k * 6 + 1] = oldcellmask_r[1];
					newcellrow_r[k * 6 + 2] = oldcellmask_r[2];//nonexistence
					newcellrow_r[k * 6 + 3] = oldcellmask_r[3];//postCell_MovementsLayerMask
					newcellrow_r[k * 6 + 4] = oldcellmask_r[4];//stationarymask
					newcellrow_r[k * 6 + 5] = oldcellmask_r[5];//randomentitymask
				}
			}
			newrule[1][j] = newcellrow_l;
			newrule[2][j] = newcellrow_r;
		}
		newrule.push(ellipses);
		newrule.push(oldrule.groupNumber);
		newrule.push(oldrule.rigid);
		newrule.push(oldrule.commands);
		newrule.push(oldrule.rhs.length===0);
		
		state.rules[i] = newrule;

	}
}

function arrangeRulesByGroupNumber(state) {
	var aggregates = {};
	var aggregates_late = {};
	for (var i=0;i<state.rules.length;i++) {
		var rule = state.rules[i];
		var groupNumber=rule[6];
		var targetArray = aggregates;
		if (rule[4]) { 
			targetArray=aggregates_late;
		}

		if (targetArray[groupNumber]==undefined) {
			targetArray[groupNumber]=[];
		}
		targetArray[groupNumber].push(rule);
	}

	var result=[];
	for (var groupNumber in aggregates) {
		if (aggregates.hasOwnProperty(groupNumber)) {
			var ruleGroup = aggregates[groupNumber];
			result.push(ruleGroup);
		}
	}
	var result_late=[];
	for (var groupNumber in aggregates_late) {
		if (aggregates_late.hasOwnProperty(groupNumber)) {
			var ruleGroup = aggregates_late[groupNumber];
			result_late.push(ruleGroup);
		}
	}
	state.rules=result;

	//check that there're no late movements with direction requirements on the lhs
	state.lateRules=result_late;
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
			if (rule[7]) {
				rigidFound=true;
			}
		}
		rigidGroups[i]=rigidFound;
		if (rigidFound) {
			var groupNumber=ruleset[0][6];
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
	var objectMask=0;
	if (name in state.objects) {
		var o=state.objects[name];
		objectMask = objectMask | (1<<o.id);		
	}

	if (name in state.aggregatesDict) {
		var objectnames = state.aggregatesDict[name];
		for(var i=0;i<objects.length;i++) {
			var n=objectnames[i];
			var o = state.objects[n];
			objectMask = objectMask | (1<<o.id);
		}
	}

	if (name in state.propertiesDict) {
		var objectnames = state.propertiesDict[name];
		for(var i=0;i<objectnames.length;i++) {
			var n = objectnames[i];
			var o = state.objects[n];
			objectMask = objectMask | (1<<o.id);
		}
	}

	if (name in state.synonymsDict) {
		var n = state.synonymsDict[name];
		var o = state.objects[n];
		objectMask = objectMask | (1<<o.id);
	}

	if (objectMask==0) {
		logErrorNoLine("error, didn't find any object called player, either in the objects section, or the legends section. there must be a player!");
	}
	return objectMask;
}

function generatePlayerMask(state) {

	state.playerMask=getMaskFromName(state,'player');

	var layerMasks=[];
	var layerCount = state.collisionLayers.length;
	for (var layer=0;layer<layerCount;layer++){
		var layerMask=0;
		for (var j=0;j<state.objectCount;j++) {
			var n=state.idDict[j];
			var o = state.objects[n];
			if (o.layer==layer) {
				layerMask = layerMask | (1<<o.id);
			}
		}
		layerMasks.push(layerMask);
	}
	state.layerMasks=layerMasks;

	var objectMask=[];
	for(var n in state.objects) {
		if (state.objects.hasOwnProperty(n)) {
			var o = state.objects[n];
			objectMask[n]=1<<o.id;
		}
	}

	for (var i=0;i<state.legend_synonyms.length;i++) {
		var syn = state.legend_synonyms[i];
		objectMask[syn[0]]=objectMask[syn[1]];
	}

	for (var i=0;i<state.legend_properties.length;i++) {
		var prop = state.legend_properties[i];
		var val = 0;
		for (var j=1;j<prop.length;j++) {
			var n = prop[j];
			val = val | objectMask[n];
		}
		objectMask[prop[0]]=val;
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
				logError('Object "' + n + '" has been defined, but not assigned to a layer.',o.lineNumber);
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

function processWinCondition(state) {
//	[-1/0/1 (no,some,all),ob1,ob2] (ob2 is background by default)
	var newconditions=[]; 
	var wincondition=state.wincondition;
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
	state.wincondition=newcondition;
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
	var result="("+rule.groupNumber+") "+ rule.direction.toString().toUpperCase()+" ";
	if (rule.rigid) {
		result = "RIGID "+result+" ";
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
	return result;
}
function printRules(state) {
	var output = "Rule Assembly : ("+ state.rules.length +" rules )<br>===========<br>";
	for (var i=0;i<state.rules.length;i++) {
		var rule = state.rules[i];
		output += printRule(rule) +"<br>";
	}
	output+="===========";
	consolePrint(output);
}

function generateLoopPoints(state) {
	var loopPoint={};
	var loopPointIndex=0;
	var outside=true;
	var source=0;
	var target=0;
	if (state.loops.length%2===1) {
		logErrorNoLine("have to have matching number of  '[[' and ']]' loop points.");
	}

	for (var j=0;j<state.loops.length;j++) {
		var loop = state.loops[j];
		for (var i=0;i<state.rules.length-1;i++) {
			var ruleGroup = state.rules[i];

			var firstRule = ruleGroup[0];			
			var lastRule = ruleGroup[ruleGroup.length-1];

			var firstRuleLine = firstRule[3];
			var lastRuleLine = lastRule[3];

			var nextRuleGroup =state.rules[i+1];

			if (outside) {
				if (firstRuleLine>=loop[0]) {
					target=i;
					outside=false;
					if (loop[1]===-1) {
						logErrorNoLine("have to have matching number of  '[[' and ']]' loop points.");						
					}
					break;
				}
			} else {
				if (firstRuleLine>=loop[0]) {
					source = i-1;		
					loopPoint[source]=target;
					outside=true;
					if (loop[1]===-1) {
						logErrorNoLine("have to have matching number of  '[[' and ']]' loop points.");						
					}
					break;
				}
			}
		}
	}
	if (outside===false) {
		var source = state.rules.length;
		loopPoint[source]=target;
	}
	state.loopPoint=loopPoint;
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

		if (soundEvents.indexOf(sound[0])>=0) {
			if (sound.length>4) {
				logError("too much stuff to define a sound event",lineNumber);
			}
			var seed = sound[1];
			if (validSeed(seed)) {
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
					directionMask = directionMask | soundDirectionMask;
				}
			}
			var targets=[target];
			if (target in state.propertiesDict) {
				targets = state.propertiesDict[target];
			}

			if (verb==='move' || verb==='cantmove') {
				for (var j=0;j<targets.length;j++) {
					var targetName = targets[j];
					var targetDat = state.objects[targetName];
					var targetLayer = targetDat.layer;
					var shiftedDirectionMask = directionMask<<(5*targetLayer);

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
	generatePlayerMask(state);
	levelsToArray(state);
	rulesToArray(state);

	if (debugMode) {
		printRules(state);
	}

	rulesToMask(state);
	collapseRules(state);
	arrangeRulesByGroupNumber(state);
	generateRigidGroupList(state);

	processWinCondition(state);
	checkObjectsAreLayered(state);

	twiddleMetaData(state);

	generateLoopPoints(state);

	generateSoundData(state);

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
