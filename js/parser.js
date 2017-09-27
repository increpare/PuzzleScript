/*
credits

brunt of the work by stephen lavelle (www.increpare.com)

all open source mit license blah blah

testers:
none, yet

code used

colors used
color values for named colours from arne, mostly (and a couple from a 32-colour palette attributed to him)
http://androidarts.com/palette/16pal.htm

the editor is a slight modification of codemirro (codemirror.net), which is crazy awesome.

for post-launch credits, check out activty on github.com/increpare/PuzzleScript

*/

var compiling = false;
var errorStrings = [];
var errorCount=0;

function logErrorCacheable(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString);
            errorStrings.push(errorString);
            errorCount++;
        }
    }
}

function logError(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
            errorCount++;
        }
    }
}

function logWarning(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="warningText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
        }
    }
}
function logErrorNoLine(str,urgent) {
    if (compiling||urgent) {
        var errorString = '<span class="errorText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
        }
        errorCount++;
    }
}



function logBetaMessage(str,urgent){
    if (compiling||urgent) {
        var errorString = '<span class="betaText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consoleError(errorString);
            errorStrings.push(errorString);
        }
    }  
}

function blankLineHandle(state) {
    if (state.section === 'levels') {
            if (state.levels[state.levels.length - 1].length > 0)
            {
                state.levels.push([]);
            }
    } else if (state.section === 'objects') {
        state.objects_section = 0;
    }
}

var codeMirrorFn = function() {
    'use strict';


    function searchStringInArray(str, strArray) {
        for (var j = 0; j < strArray.length; j++) {
            if (strArray[j] === str) { return j; }
        }
        return -1;
    }

    function isMatrixLine(str) {
        for (var j = 0; j < str.length; j++) {
            if (str.charAt(j) !== '.' && str.charAt(j) !== '0') {
                return false;
            }
        }
        return true;
    }

    function checkNameNew(state,candname) {
        if (state.objects[candname] !== undefined) {
            logError('Object "' + candname.toUpperCase() + '" defined multiple times.', state.lineNumber);
            return 'ERROR';
        }
        for (var i=0;i<state.legend_synonyms.length;i++) {
            var entry = state.legend_synonyms[i];
            if (entry[0]==candname) {
                logError('Name "' + candname.toUpperCase() + '" already in use.', state.lineNumber);                                        
            }
        }
        for (var i=0;i<state.legend_aggregates.length;i++) {
            var entry = state.legend_aggregates[i];
            if (entry[0]==candname) {
                logError('Name "' + candname.toUpperCase() + '" already in use.', state.lineNumber);                                        
            }
        }
        for (var i=0;i<state.legend_properties.length;i++) {
            var entry = state.legend_properties[i];
            if (entry[0]==candname) {
                logError('Name "' + candname.toUpperCase() + '" already in use.', state.lineNumber);                                        
            }
        }
    }
    var absolutedirs = ['up', 'down', 'right', 'left'];
    var relativedirs = ['^', 'v', '<', '>', 'moving','stationary','parallel','perpendicular', 'no'];
    var logicWords = ['all', 'no', 'on', 'some'];
    var sectionNames = ['objects', 'legend', 'sounds', 'collisionlayers', 'rules', 'winconditions', 'levels'];
	var commandwords = ["sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10","cancel","checkpoint","restart","win","message","again"];
    var reg_commands = /\s*(sfx0|sfx1|sfx2|sfx3|Sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10|cancel|checkpoint|restart|win|message|again)\s*/;
    var reg_name = /[\w]+\s*/;///\w*[a-uw-zA-UW-Z0-9_]/;
    var reg_number = /[\d]+/;
    var reg_soundseed = /\d+\b/;
    var reg_spriterow = /[\.0-9]{5}\s*/;
    var reg_sectionNames = /^\s*(objects|collisionlayers|legend|sounds|rules|winconditions|levels)\s*$/;
    var reg_equalsrow = /[\=]+/;
    var reg_notcommentstart = /[^\(]+/;
    var reg_csv_separators = /[ \,]*/;
    var reg_soundverbs = /(move|action|create|destroy|cantmove|undo|restart|titlescreen|startgame|cancel|endgame|startlevel|endlevel|showmessage|closemessage|sfx0|sfx1|sfx2|sfx3|sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10)\s+/;
    var reg_directions = /^(action|up|down|left|right|\^|v|\<|\>|forward|moving|stationary|parallel|perpendicular|horizontal|orthogonal|vertical|no|randomdir|random)$/;
    var reg_loopmarker = /^(startloop|endloop)$/;
    var reg_ruledirectionindicators = /^(up|down|left|right|horizontal|vertical|orthogonal|late|rigid)$/;
    var reg_sounddirectionindicators = /\s*(up|down|left|right|horizontal|vertical|orthogonal)\s*/;
    var reg_winconditionquantifiers = /^(all|any|no|some)$/;
    var reg_keywords = /(checkpoint|objects|collisionlayers|legend|sounds|rules|winconditions|\.\.\.|levels|up|down|left|right|^|\||\[|\]|v|\>|\<|no|horizontal|orthogonal|vertical|any|all|no|some|moving|stationary|parallel|perpendicular|action)/;
    var keyword_array = ['checkpoint','objects', 'collisionlayers', 'legend', 'sounds', 'rules', '...','winconditions', 'levels','|','[',']','up', 'down', 'left', 'right', 'late','rigid', '^','v','\>','\<','no','randomdir','random', 'horizontal', 'vertical','any', 'all', 'no', 'some', 'moving','stationary','parallel','perpendicular','action','message'];

    //  var keywordRegex = new RegExp("\\b(("+cons.join(")|(")+"))$", 'i');

    var fullSpriteMatrix = [
        '00000',
        '00000',
        '00000',
        '00000',
        '00000'
    ];

    return {
        copyState: function(state) {
            var objectsCopy = {};
            for (var i in state.objects) {
              if (state.objects.hasOwnProperty(i)) {
                var o = state.objects[i];
                objectsCopy[i] = {
                  colors: o.colors.concat([]),
                  lineNumber : o.lineNumber,
                  spritematrix: o.spritematrix.concat([])
                }
              }
            }

            var collisionLayersCopy = [];
            for (var i = 0; i < state.collisionLayers.length; i++) {
              collisionLayersCopy.push(state.collisionLayers[i].concat([]));
            }

            var legend_synonymsCopy = [];
            var legend_aggregatesCopy = [];
            var legend_propertiesCopy = [];
            var soundsCopy = [];
            var levelsCopy = [];
            var winConditionsCopy = [];

            for (var i = 0; i < state.legend_synonyms.length; i++) {
              legend_synonymsCopy.push(state.legend_synonyms[i].concat([]));
            }
            for (var i = 0; i < state.legend_aggregates.length; i++) {
              legend_aggregatesCopy.push(state.legend_aggregates[i].concat([]));
            }
            for (var i = 0; i < state.legend_properties.length; i++) {
              legend_propertiesCopy.push(state.legend_properties[i].concat([]));
            }
            for (var i = 0; i < state.sounds.length; i++) {
              soundsCopy.push(state.sounds[i].concat([]));
            }
            for (var i = 0; i < state.levels.length; i++) {
              levelsCopy.push(state.levels[i].concat([]));
            }
            for (var i = 0; i < state.winconditions.length; i++) {
              winConditionsCopy.push(state.winconditions[i].concat([]));
            }

            var nstate = {
              lineNumber: state.lineNumber,

              objects: objectsCopy,
              collisionLayers: collisionLayersCopy,

              commentLevel: state.commentLevel,
              section: state.section,
              visitedSections: state.visitedSections.concat([]),

              objects_candname: state.objects_candname,
              objects_section: state.objects_section,
              objects_spritematrix: state.objects_spritematrix.concat([]),

              tokenIndex: state.tokenIndex,
              legend_synonyms: legend_synonymsCopy,
              legend_aggregates: legend_aggregatesCopy,
              legend_properties: legend_propertiesCopy,

              sounds: soundsCopy,

              rules: state.rules.concat([]),

              names: state.names.concat([]),

              winconditions: winConditionsCopy,

              abbrevNames: state.abbrevNames.concat([]),

              metadata : state.metadata.concat([]),

              levels: levelsCopy,

              STRIDE_OBJ : state.STRIDE_OBJ,
              STRIDE_MOV : state.STRIDE_MOV
            };

            return nstate;        
        },
        blankLine: function(state) {
            if (state.section === 'levels') {
                    if (state.levels[state.levels.length - 1].length > 0)
                    {
                        state.levels.push([]);
                    }
            }
        },
        token: function(stream, state) {
           	var mixedCase = stream.string;
            var sol = stream.sol();
            if (sol) {
                stream.string = stream.string.toLowerCase();
                state.tokenIndex=0;
                /*   if (state.lineNumber==undefined) {
                        state.lineNumber=1;
                }
                else {
                    state.lineNumber++;
                }*/

            }

            stream.eatWhile(/[ \t]/);

            ////////////////////////////////
            // COMMENT PROCESSING BEGIN
            ////////////////////////////////

            //NESTED COMMENTS
            var ch = stream.peek();
            if (ch === '(' && state.tokenIndex !== -4) { // tokenIndex -4 indicates message command
                stream.next();
                state.commentLevel++;
            } else if (ch === ')') {
                stream.next();
                if (state.commentLevel > 0) {
                    state.commentLevel--;
                    if (state.commentLevel === 0) {
                        return 'comment';
                    }
                }
            }
            if (state.commentLevel > 0) {
                while (true) {
                    stream.eatWhile(/[^\(\)]+/);

                    if (stream.eol()) {
                        break;
                    }

                    ch = stream.peek();

                    if (ch === '(') {
                        state.commentLevel++;
                    } else if (ch === ')') {
                        state.commentLevel--;
                    }
                    stream.next();

                    if (state.commentLevel === 0) {
                        break;
                    }
                }
                return 'comment';
            }

            stream.eatWhile(/[ \t]/);

            if (sol && stream.eol()) {
                return blankLineHandle(state);
            }

            //  if (sol)
            {

                //MATCH '==="s AT START OF LINE
                if (sol && stream.match(reg_equalsrow, true)) {
                    return 'EQUALSBIT';
                }

                //MATCH SECTION NAME
                if (stream.match(reg_sectionNames, true)) {
                    state.section = stream.string.slice(0, stream.pos).trim();
                    if (state.visitedSections.indexOf(state.section) >= 0) {
                        logError('cannot duplicate sections (you tried to duplicate \"' + state.section.toUpperCase() + '").', state.lineNumber);
                    }
                    state.visitedSections.push(state.section);
                    var sectionIndex = sectionNames.indexOf(state.section);
                    if (sectionIndex == 0) {
                        state.objects_section = 0;
                        if (state.visitedSections.length > 1) {
                            logError('section "' + state.section.toUpperCase() + '" must be the first section', state.lineNumber);
                        }
                    } else if (state.visitedSections.indexOf(sectionNames[sectionIndex - 1]) == -1) {
                        if (sectionIndex===-1) {
                            logError('no such section as "' + state.section.toUpperCase() + '".', state.lineNumber);
                        } else {
                            logError('section "' + state.section.toUpperCase() + '" is out of order, must follow  "' + sectionNames[sectionIndex - 1].toUpperCase() + '".', state.lineNumber);                            
                        }
                    }

                    if (state.section === 'sounds') {
                        //populate names from rules
                        for (var n in state.objects) {
                            if (state.objects.hasOwnProperty(n)) {
/*                                if (state.names.indexOf(n)!==-1) {
                                    logError('Object "'+n+'" has been declared to be multiple different things',state.objects[n].lineNumber);
                                }*/
                                state.names.push(n);
                            }
                        }
                        //populate names from legends
                        for (var i = 0; i < state.legend_synonyms.length; i++) {
                            var n = state.legend_synonyms[i][0];
                            /*
                            if (state.names.indexOf(n)!==-1) {
                                logError('Object "'+n+'" has been declared to be multiple different things',state.legend_synonyms[i].lineNumber);
                            }
                            */
                            state.names.push(n);
                        }
                        for (var i = 0; i < state.legend_aggregates.length; i++) {
                            var n = state.legend_aggregates[i][0];
                            /*
                            if (state.names.indexOf(n)!==-1) {
                                logError('Object "'+n+'" has been declared to be multiple different things',state.legend_aggregates[i].lineNumber);
                            }
                            */
                            state.names.push(n);
                        }
                        for (var i = 0; i < state.legend_properties.length; i++) {
                            var n = state.legend_properties[i][0];
                            /*
                            if (state.names.indexOf(n)!==-1) {
                                logError('Object "'+n+'" has been declared to be multiple different things',state.legend_properties[i].lineNumber);
                            }                           
                            */ 
                            state.names.push(n);
                        }
                    }
                    else if (state.section === 'levels') {
                        //populate character abbreviations
                        for (var n in state.objects) {
                            if (state.objects.hasOwnProperty(n) && n.length == 1) {
                                state.abbrevNames.push(n);
                            }
                        }

                        for (var i = 0; i < state.legend_synonyms.length; i++) {
                            if (state.legend_synonyms[i][0].length == 1) {
                                state.abbrevNames.push(state.legend_synonyms[i][0]);
                            }
                        }
                        for (var i = 0; i < state.legend_aggregates.length; i++) {
                            if (state.legend_aggregates[i][0].length == 1) {
                                state.abbrevNames.push(state.legend_aggregates[i][0]);
                            }
                        }
                    }
                    return 'HEADER';
                } else {
                    if (state.section === undefined) {
                        logError('must start with section "OBJECTS"', state.lineNumber);
                    }
                }

                if (stream.eol()) {
                    return null;
                }

                //if color is set, try to set matrix
                //if can't set matrix, try to parse name
                //if color is not set, try to parse color
                switch (state.section) {
                case 'objects':
                    {
						var tryParseName = function() {
                            //LOOK FOR NAME
                            var match_name = sol ? stream.match(reg_name, true) : stream.match(/[^\s\()]+\s*/,true);
                            if (match_name == null) {
                                stream.match(reg_notcommentstart, true);
                                if (stream.pos>0){                                
                                    logWarning('Unknown junk in object section (possibly: sprites have to be 5 pixels wide and 5 pixels high exactly. Or maybe: the main names for objects have to be words containing only the letters a-z0.9 - if you want to call them something like ",", do it in the legend section).',state.lineNumber);
                                }
                                return 'ERROR';
                            } else {
                            	var candname = match_name[0].trim();
                                if (state.objects[candname] !== undefined) {
                                    logError('Object "' + candname.toUpperCase() + '" defined multiple times.', state.lineNumber);
                                    return 'ERROR';
                                }
                                for (var i=0;i<state.legend_synonyms.length;i++) {
                                	var entry = state.legend_synonyms[i];
                                	if (entry[0]==candname) {
                                    	logError('Name "' + candname.toUpperCase() + '" already in use.', state.lineNumber);                                		
                                	}
                                }
                                if (keyword_array.indexOf(candname)>=0) {
                                    logWarning('You named an object "' + candname.toUpperCase() + '", but this is a keyword. Don\'t do that!', state.lineNumber);
                                }

                                if (sol) {
                                	state.objects_candname = candname;
                                	state.objects[state.objects_candname] = {
										                                	lineNumber: state.lineNumber,
										                                	colors: [],
										                                	spritematrix: []
										                                };
								} else {
									//set up alias
									var synonym = [candname,state.objects_candname];
									synonym.lineNumber = state.lineNumber;
									state.legend_synonyms.push(synonym);
								}
                                state.objects_section = 1;
                                return 'NAME';
                            }
                        };

                        if (sol && state.objects_section == 2) {
                            state.objects_section = 3;
                        }

                        if (sol && state.objects_section == 1) {
                            state.objects_section = 2;
                        }

                        switch (state.objects_section) {
                        case 0:
                        case 1:
                            {
                                state.objects_spritematrix = [];
                                return tryParseName();
                                break;
                            }
                        case 2:
                            {
                                //LOOK FOR COLOR
                                state.tokenIndex = 0;
                                var match_color = stream.match(reg_color, true);
                                if (match_color == null) {
                                    var str = stream.match(reg_name, true) || stream.match(reg_notcommentstart, true);
                                    logError('Was looking for color for object ' + state.objects_candname.toUpperCase() + ', got "' + str + '" instead.', state.lineNumber);
                                    return null;
                                } else {
                                    if (state.objects[state.objects_candname].colors === undefined) {
                                        state.objects[state.objects_candname].colors = [match_color[0].trim()];
                                    } else {
                                        state.objects[state.objects_candname].colors.push(match_color[0].trim());
                                    }

                                    var candcol = match_color[0].trim().toLowerCase();
                                    if (candcol in colorPalettes.arnecolors) {
                                        return 'COLOR COLOR-' + candcol.toUpperCase();
                                    } else if (candcol==="transparent") {
                                        return 'COLOR FADECOLOR';
                                    } else {
                                        return 'COLOR';
                                    }
                                }
                                break;
                            }
                        case 3:
                            {
                                var ch = stream.eat(/[.\d]/);
                                var spritematrix = state.objects_spritematrix;
                                if (ch === undefined) {
                                    if (spritematrix.length === 0) {
                                        return tryParseName();
                                    }
                                    logError('Unknown junk in spritematrix for object ' + state.objects_candname.toUpperCase() + '.', state.lineNumber);
                                    stream.match(reg_notcommentstart, true);
                                    return null;
                                }

                                if (sol) {
                                    spritematrix.push('');
                                }

                                var o = state.objects[state.objects_candname];

                                spritematrix[spritematrix.length - 1] += ch;
                                if (spritematrix[spritematrix.length-1].length>5){
                                    logError('Sprites must be 5 wide and 5 high.', state.lineNumber);
                                    stream.match(reg_notcommentstart, true);
                                    return null;
                                }
                                o.spritematrix = state.objects_spritematrix;
                                if (spritematrix.length === 5 && spritematrix[spritematrix.length - 1].length == 5) {
                                    state.objects_section = 0;
                                }

                                if (ch!=='.') {
                                    var n = parseInt(ch);
                                    if (n>=o.colors.length) {
                                        logError("Trying to access color number "+n+" from the color palette of sprite " +state.objects_candname.toUpperCase()+", but there are only "+o.colors.length+" defined in it.",state.lineNumber);
                                        return 'ERROR';
                                    }
                                    if (isNaN(n)) {
                                        logError('Invalid character "' + ch + '" in sprite for ' + state.objects_candname.toUpperCase(), state.lineNumber);
                                        return 'ERROR';
                                    }
                                    return 'COLOR BOLDCOLOR COLOR-' + o.colors[n].toUpperCase();
                                }
                                return 'COLOR FADECOLOR';
                            }
                        default:
                        	{
                        	window.console.logError("EEK shouldn't get here.");
                        	}
                        }
                        break;
                    }
                case 'sounds':
                    {
                        if (sol) {
                            var ok = true;
                            var splits = reg_notcommentstart.exec(stream.string)[0].split(/\s/).filter(function(v) {return v !== ''});                          
                            splits.push(state.lineNumber);
                            state.sounds.push(splits);
                        }
                        candname = stream.match(reg_soundverbs, true);
                        if (candname!==null) {
                        	return 'SOUNDVERB';
                        }
                        candname = stream.match(reg_sounddirectionindicators,true);
                        if (candname!==null) {
                        	return 'DIRECTION';
                        }
                        candname = stream.match(reg_soundseed, true);
                        if (candname !== null)
                        {
                            state.tokenIndex++;
                            return 'SOUND';
                        } 
                       	candname = stream.match(/[^\[\|\]\s]*/, true);
                       	if (candname!== null ) {
                       		var m = candname[0].trim();
                       		if (state.names.indexOf(m)>=0) {
                       			return 'NAME';
                       		}
                       	}

                        candname = stream.match(reg_notcommentstart, true);
                        logError('unexpected sound token "'+candname+'".' , state.lineNumber);
                        stream.match(reg_notcommentstart, true);
                        return 'ERROR';
                        break;
                    }
                case 'collisionlayers':
                    {
                        if (sol) {
                            //create new collision layer
                            state.collisionLayers.push([]);
                            state.tokenIndex=0;
                        }

                        var match_name = stream.match(reg_name, true);
                        if (match_name === null) {
                            //then strip spaces and commas
                            var prepos=stream.pos;
                            stream.match(reg_csv_separators, true);
                            if (stream.pos==prepos) {
                                logError("error detected - unexpected character " + stream.peek(),state.lineNumber);
                                stream.next();
                            }
                            return null;
                        } else {
                            //have a name: let's see if it's valid
                            var candname = match_name[0].trim();

                            var substitutor = function(n) {
                            	n = n.toLowerCase();
                            	if (n in state.objects) {
                            		return [n];
                            	} 


                                for (var i=0;i<state.legend_synonyms.length;i++) {
                                    var a = state.legend_synonyms[i];
                                    if (a[0]===n) {           
                                        return substitutor(a[1]);
                                    }
                                }

                            	for (var i=0;i<state.legend_aggregates.length;i++) {
                            		var a = state.legend_aggregates[i];
                            		if (a[0]===n) {           
                            			logError('"'+n+'" is an aggregate (defined using "and"), and cannot be added to a single layer because its constituent objects must be able to coexist.', state.lineNumber);
                            			return [];         
                            		}
                            	}
                            	for (var i=0;i<state.legend_properties.length;i++) {
                            		var a = state.legend_properties[i];
                            		if (a[0]===n) {  
                                        var result = [].concat.apply([],a.slice(1).map(substitutor));
                            			return result;
                            		}
                            	}
                            	logError('Cannot add "' + candname.toUpperCase() + '" to a collision layer; it has not been declared.', state.lineNumber);                                
                            	return [];
                            };
                            if (candname==='background' ) {
                                if (state.collisionLayers.length>0&&state.collisionLayers[state.collisionLayers.length-1].length>0) {
                                    logError("Background must be in a layer by itself.",state.lineNumber);
                                }
                                state.tokenIndex=1;
                            } else if (state.tokenIndex!==0) {
                                logError("Background must be in a layer by itself.",state.lineNumber);
                            }

                            var ar = substitutor(candname);

                            if (state.collisionLayers.length===0) {
                                logError("no layers found.",state.lineNumber);
                                return 'ERROR';
                            }
                            
                            var foundOthers=[];
                            for (var i=0;i<ar.length;i++){
                                var candname = ar[i];
                                for (var j=0;j<=state.collisionLayers.length-1;j++){
                                    var clj = state.collisionLayers[j];
                                    if (clj.indexOf(candname)>=0){
                                        if (j!=state.collisionLayers.length-1){
                                            foundOthers.push(j);
                                        }
                                    }
                                }
                            }
                            if (foundOthers.length>0){
                                var warningStr = 'Object "'+candname.toUpperCase()+'" included in multiple collision layers ( layers ';
                                for (var i=0;i<foundOthers.length;i++){
                                    warningStr+=foundOthers[i]+", ";
                                }
                                warningStr+=state.collisionLayers.length-1;
                                logWarning(warningStr +'). You should fix this!',state.lineNumber);                                        
                            }

                            state.collisionLayers[state.collisionLayers.length - 1] = state.collisionLayers[state.collisionLayers.length - 1].concat(ar);
                            if (ar.length>0) {
                            	return 'NAME';                            
                            } else {
                            	return 'ERROR';
                            }
                        }
                        break;
                    }
                case 'legend':
                    {
                        if (sol) {


                            //step 1 : verify format
                            var longer = stream.string.replace('=', ' = ');
                            longer = reg_notcommentstart.exec(longer)[0];

                            var splits = longer.split(/\s/).filter(function(v) {
                                return v !== '';
                            });
                            var ok = true;

                        	if (splits.length>0) {
                        		var candname = splits[0].toLowerCase();
	                            if (keyword_array.indexOf(candname)>=0) {
	                                logWarning('You named an object "' + candname.toUpperCase() + '", but this is a keyword. Don\'t do that!', state.lineNumber);
	                            }
                                if (splits.indexOf(candname, 2)>=2) {
                                    logError("You can't define object " + candname.toUpperCase() + " in terms of itself!", state.lineNumber);
                                    ok = false;
                                }
                                checkNameNew(state,candname);
                        	}

                            if (!ok) {
                            } else if (splits.length < 3) {
                                ok = false;
                            } else if (splits[1] !== '=') {
                                ok = false;
                            } /*else if (splits[0].charAt(splits[0].length - 1) == 'v') {
                                logError('names cannot end with the letter "v", because it\'s is used as a direction.', state.lineNumber);
                                stream.match(reg_notcommentstart, true);
                                return 'ERROR';
                            } */ else if (splits.length === 3) {
                                var synonym = [splits[0], splits[2].toLowerCase()];
                                synonym.lineNumber = state.lineNumber;
                                state.legend_synonyms.push(synonym);
                            } else if (splits.length % 2 === 0) {
                                ok = false;
                            } else {
                                var lowertoken = splits[3].toLowerCase();
                                if (lowertoken === 'and') {

	                                var substitutor = function(n) {
	                                	n = n.toLowerCase();
	                                	if (n in state.objects) {
	                                		return [n];
	                                	} 
	                                	for (var i=0;i<state.legend_synonyms.length;i++) {
	                                		var a = state.legend_synonyms[i];
	                                		if (a[0]===n) {   
	                                			return substitutor(a[1]);
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_aggregates.length;i++) {
	                                		var a = state.legend_aggregates[i];
	                                		if (a[0]===n) {                                			
	                                			return [].concat.apply([],a.slice(1).map(substitutor));
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_properties.length;i++) {
	                                		var a = state.legend_properties[i];
	                                		if (a[0]===n) {         
	                                			logError("Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').", state.lineNumber);
	                                			ok=false;
	                                			return [n];
	                                		}
	                                	}
	                                	return [n];
	                                };

                                    for (var i = 5; i < splits.length; i += 2) {
                                        if (splits[i].toLowerCase() !== 'and') {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if (ok) {
                                        var newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                                        for (var i = 6; i < splits.length; i += 2) {
                                            newlegend = newlegend.concat(substitutor(splits[i]));
                                        }
                                        newlegend.lineNumber = state.lineNumber;
                                        state.legend_aggregates.push(newlegend);
                                    }
                                } else if (lowertoken === 'or') {

	                                var substitutor = function(n) {
	                                	n = n.toLowerCase();
	                                	if (n in state.objects) {
	                                		return [n];
	                                	} 

	                                	for (var i=0;i<state.legend_synonyms.length;i++) {
	                                		var a = state.legend_synonyms[i];
	                                		if (a[0]===n) {   
	                                			return substitutor(a[1]);
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_aggregates.length;i++) {
	                                		var a = state.legend_aggregates[i];
	                                		if (a[0]===n) {           
	                                			logError("Cannot define a property (using 'or') in terms of aggregates (something that uses 'and').", state.lineNumber);
	                                			ok=false;          
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_properties.length;i++) {
	                                		var a = state.legend_properties[i];
	                                		if (a[0]===n) {  
	                                			return [].concat.apply([],a.slice(1).map(substitutor));
	                                		}
	                                	}
	                                	return [n];
	                                };

                                    for (var i = 5; i < splits.length; i += 2) {
                                        if (splits[i].toLowerCase() !== 'or') {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if (ok) {
                                        var newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                                        for (var i = 6; i < splits.length; i += 2) {
                                            newlegend.push(splits[i].toLowerCase());
                                        }
                                        newlegend.lineNumber = state.lineNumber;
                                        state.legend_properties.push(newlegend);
                                    }
                                } else {
                                    ok = false;
                                }
                            }

                            if (ok === false) {
                                logError('incorrect format of legend - should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)', state.lineNumber);
                                stream.match(reg_notcommentstart, true);
                                return 'ERROR';
                            }

                            state.tokenIndex = 0;
                        }

                        if (state.tokenIndex === 0) {
                            stream.match(/[^=]*/, true);
                            state.tokenIndex++;
                            return 'NAME';
                        } else if (state.tokenIndex === 1) {
                            stream.next();
                            stream.match(/\s*/, true);
                            state.tokenIndex++;
                            return 'ASSSIGNMENT';
                        } else {
                            var match_name = stream.match(reg_name, true);
                            if (match_name === null) {
                                logError("Something bad's happening in the LEGEND", state.lineNumber);
                                stream.match(reg_notcommentstart, true);
                                return 'ERROR';
                            } else {
                                var candname = match_name[0].trim();

                                if (state.tokenIndex % 2 === 0) {

	                                var wordExists = function(n) {
	                                	n = n.toLowerCase();
	                                	if (n in state.objects) {
	                                		return true;
	                                	} 
	                                	for (var i=0;i<state.legend_aggregates.length;i++) {
	                                		var a = state.legend_aggregates[i];
	                                		if (a[0]===n) {                                			
	                                			return true;
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_properties.length;i++) {
	                                		var a = state.legend_properties[i];
	                                		if (a[0]===n) {  
	                                			return true;
	                                		}
	                                	}
	                                	for (var i=0;i<state.legend_synonyms.length;i++) {
	                                		var a = state.legend_synonyms[i];
	                                		if (a[0]===n) {  
	                                			return true;
	                                		}
	                                	}
	                                	return false;
	                                };


                                    if (wordExists(candname)===false) {
                                            logError('Cannot reference "' + candname.toUpperCase() + '" in the LEGEND section; it has not been defined yet.', state.lineNumber);
                                            state.tokenIndex++;
                                            return 'ERROR';
                                    } else {
                                            state.tokenIndex++;
                                            return 'NAME';
                                    }
                                } else {
                                        state.tokenIndex++;
                                        return 'LOGICWORD';
                                }
                            }
                        }
                        break;
                    }
                case 'rules':
                    {                    	
                        if (sol) {
                            var rule = reg_notcommentstart.exec(stream.string)[0];
                            state.rules.push([rule, state.lineNumber, mixedCase]);
                            state.tokenIndex = 0;//in rules, records whether bracket has been found or not
                        }

                        if (state.tokenIndex===-4) {
                        	stream.skipToEnd();
                        	return 'MESSAGE';
                        }
                        if (stream.match(/\s*\-\>\s*/, true)) {
                            return 'ARROW';
                        }
                        if (ch === '[' || ch === '|' || ch === ']' || ch==='+') {
                        	if (ch!=='+') {
                            	state.tokenIndex = 1;
                            }
                            stream.next();
                            stream.match(/\s*/, true);
                            return 'BRACKET';
                        } else {
                            var m = stream.match(/[^\[\|\]\s]*/, true)[0].trim();

                            if (state.tokenIndex===0&&reg_loopmarker.exec(m)) {
                            	return 'BRACKET';
                            } else if (state.tokenIndex === 0 && reg_ruledirectionindicators.exec(m)) {
                                stream.match(/\s*/, true);
                                return 'DIRECTION';
                            } else if (state.tokenIndex === 1 && reg_directions.exec(m)) {
                                stream.match(/\s*/, true);
                                return 'DIRECTION';
                            } else {
                                if (state.names.indexOf(m) >= 0) {
                                    if (sol) {
                                        logError('Identifiers cannot appear outside of square brackets in rules, only directions can.', state.lineNumber);
                                        return 'ERROR';
                                    } else {
                                        stream.match(/\s*/, true);
                                        return 'NAME';
                                    }
                                } else if (m==='...') {
                                    return 'DIRECTION';
                                } else if (m==='rigid') {
                                    return 'DIRECTION';
                                } else if (m==='random') {
                                    return 'DIRECTION';
                                } else if (commandwords.indexOf(m)>=0) {
									if (m==='message') {
										state.tokenIndex=-4;
									}                                	
                                	return 'COMMAND';
                                } else {
                                    logError('Name "' + m + '", referred to in a rule, does not exist.', state.lineNumber);
                                    return 'ERROR';
                                }
                            }
                        }

                        break;
                    }
                case 'winconditions':
                    {
                        if (sol) {
                        	var tokenized = reg_notcommentstart.exec(stream.string);
                        	var splitted = tokenized[0].split(/\s/);
                        	var filtered = splitted.filter(function(v) {return v !== ''});
                            filtered.push(state.lineNumber);
                            
                            state.winconditions.push(filtered);
                            state.tokenIndex = -1;
                        }
                        state.tokenIndex++;
                        var match = stream.match(/\s*\w+\s*/);
                        if (match === null) {
                                logError('incorrect format of win condition.', state.lineNumber);
                                stream.match(reg_notcommentstart, true);
                                return 'ERROR';

                        } else {
                            var candword = match[0].trim();
                            if (state.tokenIndex === 0) {
                                if (reg_winconditionquantifiers.exec(candword)) {
                                    return 'LOGICWORD';
                                }
                                else {
                                    return 'ERROR';
                                }
                            }
                            else if (state.tokenIndex === 2) {
                                if (candword != 'on') {
                                    return 'ERROR';
                                } else {
                                    return 'LOGICWORD';
                                }
                            }
                            else if (state.tokenIndex === 1 || state.tokenIndex === 3) {
                                if (state.names.indexOf(candword)===-1) {
                                    logError('Error in win condition: "' + candword.toUpperCase() + '" is not a valid object name.', state.lineNumber);
                                    return 'ERROR';
                                } else {
                                    return 'NAME';
                                }
                            }
                        }
                        break;
                    }
                case 'levels':
                    {
                        if (sol)
                        {
                            if (stream.match(/\s*message\s*/, true)) {
                                state.tokenIndex = 1;//1/2 = message/level
                                var newdat = ['\n', mixedCase.slice(stream.pos).trim(),state.lineNumber];
                                if (state.levels[state.levels.length - 1].length == 0) {
                                    state.levels.splice(state.levels.length - 1, 0, newdat);
                                } else {
                                    state.levels.push(newdat);
                                }
                                return 'MESSAGE_VERB';
                            } else {
                                var line = stream.match(reg_notcommentstart, false)[0].trim();
                                state.tokenIndex = 2;
                                var lastlevel = state.levels[state.levels.length - 1];
                                if (lastlevel[0] == '\n') {
                                    state.levels.push([state.lineNumber,line]);
                                } else {
                                    if (lastlevel.length==0)
                                    {
                                        lastlevel.push(state.lineNumber);
                                    }
                                    lastlevel.push(line);  

                                    if (lastlevel.length>1) 
                                    {
                                        if (line.length!=lastlevel[1].length) {
                                            logWarning("Maps must be rectangular, yo (In a level, the length of each row must be the same).",state.lineNumber);
                                        }
                                    }
                                }
                                
                            }
                        } else {
                            if (state.tokenIndex == 1) {
                                stream.skipToEnd();
                               	return 'MESSAGE';
                            }
                        }

                        if (state.tokenIndex === 2 && !stream.eol()) {
                            var ch = stream.peek();
                            stream.next();
                            if (state.abbrevNames.indexOf(ch) >= 0) {
                                return 'LEVEL';
                            } else {
                                logError('Key "' + ch.toUpperCase() + '" not found. Do you need to add it to the legend, or define a new object?', state.lineNumber);
                                return 'ERROR';
                            }
                        }
                        break;
                    }
	                
	                default://if you're in the preamble
	                {
	            		if (sol) {
	            			state.tokenIndex=0;
	            		}
	            		if (state.tokenIndex==0) {
		                    var match = stream.match(/\s*\w+\s*/);	                    
		                    if (match!==null) {
		                    	var token = match[0].trim();
		                    	if (sol) {
		                    		if (['title','author','homepage','background_color','text_color','key_repeat_interval','realtime_interval','again_interval','flickscreen','zoomscreen','color_palette','youtube'].indexOf(token)>=0) {
		                    			
                                        if (token==='youtube' || token==='author' || token==='title') {
                                            stream.string=mixedCase;
                                        }
                                        
                                        var m2 = stream.match(reg_notcommentstart, false);
                                        
		                    			if(m2!=null) {
                                            state.metadata.push(token);
		                    				state.metadata.push(m2[0].trim());                                            
		                    			} else {
		                    				logError('MetaData "'+token+'" needs a value.',state.lineNumber);
		                    			}
		                    			state.tokenIndex=1;
		                    			return 'METADATA';
		                    		} else if ( ['run_rules_on_level_start','norepeat_action','require_player_movement','debug','verbose_logging','throttle_movement','noundo','noaction','norestart','scanline'].indexOf(token)>=0) {
		                    			state.metadata.push(token);
		                    			state.metadata.push("true");
		                    			state.tokenIndex=-1;
		                    			return 'METADATA';
		                    		} else  {
		                    			logError('Unrecognised stuff in metadata section.', state.lineNumber);
		                    			return 'ERROR';
		                    		}
		                    	} else if (state.tokenIndex==-1) {
	                   				logError('MetaData "'+token+'" has no parameters.',state.lineNumber);
		                    		return 'ERROR';
		                    	}
		                    	return 'METADATA';
		                    }       
		               	} else {
		               		stream.match(reg_notcommentstart, true);
		               		return "METADATATEXT";
		               	}
	                	break;
	                }
	            }
            };

            if (stream.eol()) {
                return null;
            }
            if (!stream.eol()) {
                stream.next();
                return null;
            }
        },
        startState: function() {
            return {
                /*
                    permanently useful
                */
                objects: {},

                /*
                    for parsing
                */
                lineNumber: 0,

                commentLevel: 0,

                section: '',
                visitedSections: [],

                objects_candname: '',
                objects_section: 0, //whether reading name/color/spritematrix
                objects_spritematrix: [],

                collisionLayers: [],

                tokenIndex: 0,

                legend_synonyms: [],
                legend_aggregates: [],
                legend_properties: [],

                sounds: [],
                rules: [],

                names: [],

                winconditions: [],
                metadata: [],

                abbrevNames: [],

                levels: [[]],

                subsection: ''
            };
        }
    };
};

window.CodeMirror.defineMode('puzzle', codeMirrorFn);
