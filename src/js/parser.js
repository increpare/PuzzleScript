/*
credits

brunt of the work by increpare (www.increpare.com)

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


const MAX_ERRORS_FOR_REAL=100;

var compiling = false;
var errorStrings = [];//also stores warning strings
var errorCount=0;//only counts errors

function TooManyErrors(){
    consolePrint("Too many errors/warnings; aborting compilation.",true);
    throw new Error("Too many errors/warnings; aborting compilation.");
}

function logErrorCacheable(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str,urgent);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString);
            errorStrings.push(errorString);
            errorCount++;
			if (errorStrings.length>MAX_ERRORS_FOR_REAL){
                TooManyErrors();
			}
        }
    }
}

function logError(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str,urgent);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
            errorCount++;
			if (errorStrings.length>MAX_ERRORS_FOR_REAL){
                TooManyErrors();
			}
        }
    }
}

function logWarning(str, lineNumber,urgent) {
    if (compiling||urgent) {
        if (lineNumber === undefined) {
            return logWarningNoLine(str,urgent);
        }
        var errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="warningText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
			if (errorStrings.length>MAX_ERRORS_FOR_REAL){
                TooManyErrors();
			}
        }
    }
}

function logWarningNoLine(str,urgent) {
    if (compiling||urgent) {
        var errorString = '<span class="warningText">' + str + '</span>';
         if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
         } else {
            consolePrint(errorString,true);
            errorStrings.push(errorString);
            errorCount++;
			if (errorStrings.length>MAX_ERRORS_FOR_REAL){
                TooManyErrors();
			}
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
            errorCount++;
			if (errorStrings.length>MAX_ERRORS_FOR_REAL){
                TooManyErrors();
			}
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

//returns null if not delcared, otherwise declaration
//note to self: I don't think that aggregates or properties know that they're aggregates or properties in and of themselves.
function wordAlreadyDeclared(state,n) {
    n = n.toLowerCase();
    if (n in state.objects) {
        return state.objects[n];
    } 
    for (var i=0;i<state.legend_aggregates.length;i++) {
        var a = state.legend_aggregates[i];
        if (a[0]===n) {                                			
            return state.legend_aggregates[i];
        }
    }
    for (var i=0;i<state.legend_properties.length;i++) {
        var a = state.legend_properties[i];
        if (a[0]===n) {  
            return state.legend_properties[i];
        }
    }
    for (var i=0;i<state.legend_synonyms.length;i++) {
        var a = state.legend_synonyms[i];
        if (a[0]===n) {  
            return state.legend_synonyms[i];
        }
    }
    return null;
}


//for IE support
if (typeof Object.assign != 'function') {
  (function () {
    Object.assign = function (target) {
      'use strict';
      // We must check against these specific cases.
      if (target === undefined || target === null) {
        throw new TypeError('Cannot convert undefined or null to object');
      }
 
      var output = Object(target);
      for (var index = 1; index < arguments.length; index++) {
        var source = arguments[index];
        if (source !== undefined && source !== null) {
          for (var nextKey in source) {
            if (source.hasOwnProperty(nextKey)) {
              output[nextKey] = source[nextKey];
            }
          }
        }
      }
      return output;
    };
  })();
}


var codeMirrorFn = function() {
    'use strict';

    function checkNameDefined(state,candname) {
        if (state.objects[candname] !== undefined) {
            return;
        }
        for (var i=0;i<state.legend_synonyms.length;i++) {
            var entry = state.legend_synonyms[i];
            if (entry[0]==candname) {
                return;                                       
            }
        }
        for (var i=0;i<state.legend_aggregates.length;i++) {
            var entry = state.legend_aggregates[i];
            if (entry[0]==candname) {
                return;                                                                          
            }
        }
        for (var i=0;i<state.legend_properties.length;i++) {
            var entry = state.legend_properties[i];
            if (entry[0]==candname) {
                return;                                    
            }
        }
        
        logError(`You're talking about ${candname.toUpperCase()} but it's not defined anywhere.`, state.lineNumber);    
    }

    function registerOriginalCaseName(state,candname,mixedCase,lineNumber){

        function escapeRegExp(str) {
          return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
        }

        var nameFinder =  new RegExp("\\b"+escapeRegExp(candname)+"\\b","i")
        var match = mixedCase.match(nameFinder);
        if (match!=null){
            state.original_case_names[candname] = match[0];
            state.original_line_numbers[candname] = lineNumber;
        }
    }

    const absolutedirs = ['up', 'down', 'right', 'left'];
    const relativedirs = ['^', 'v', '<', '>', 'moving','stationary','parallel','perpendicular', 'no'];
    const logicWords = ['all', 'no', 'on', 'some'];
    const sectionNames = ['objects', 'legend', 'sounds', 'collisionlayers', 'rules', 'winconditions', 'levels'];
	const commandwords = ["sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10","cancel","checkpoint","restart","win","message","again"];
    const reg_commands = /[\p{Z}\s]*(sfx0|sfx1|sfx2|sfx3|Sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10|cancel|checkpoint|restart|win|message|again)[\p{Z}\s]*/u;
    const reg_name = /[\p{L}\p{N}_]+[\p{Z}\s]*/u;///\w*[a-uw-zA-UW-Z0-9_]/;
    const reg_number = /[\d]+/;
    const reg_soundseed = /\d+\b/u;
    const reg_spriterow = /[\.0-9]{5}[\p{Z}\s]*/u;
    const reg_sectionNames = /(objects|collisionlayers|legend|sounds|rules|winconditions|levels)(?![\p{L}\p{N}_])[\p{Z}\s]*/u;
    const reg_equalsrow = /[\=]+/;
    const reg_notcommentstart = /[^\(]+/;
    const reg_match_until_commentstart_or_whitespace = /[^\p{Z}\s\()]+[\p{Z}\s]*/u;
    const reg_csv_separators = /[ \,]*/;
    const reg_soundverbs = /(move|action|create|destroy|cantmove)\b[\p{Z}\s]*/u;
    const soundverbs_directional = ['move','cantmove'];
    const reg_soundverbs_directional = /(move|cantmove)\b[\p{Z}\s]*/u;
    const reg_soundverbs_nondirectional = /(action|create|destroy)\b[\p{Z}\s]*/u;
    const reg_soundevents = /(undo|restart|titlescreen|startgame|cancel|endgame|startlevel|endlevel|showmessage|closemessage|sfx0|sfx1|sfx2|sfx3|sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10)\b[\p{Z}\s]*/u;

    const reg_directions = /^(action|up|down|left|right|\^|v|\<|\>|moving|stationary|parallel|perpendicular|horizontal|orthogonal|vertical|no|randomdir|random)$/;
    const reg_loopmarker = /^(startloop|endloop)$/;
    const reg_ruledirectionindicators = /^(up|down|left|right|horizontal|vertical|orthogonal|late|rigid)$/;
    const reg_sounddirectionindicators = /[\p{Z}\s]*(up|down|left|right|horizontal|vertical|orthogonal)(?![\p{L}\p{N}_])[\p{Z}\s]*/u;
    const reg_winconditionquantifiers = /^(all|any|no|some)$/;
    const reg_keywords = /(checkpoint|objects|collisionlayers|legend|sounds|rules|winconditions|\.\.\.|levels|up|down|left|right|^|\||\[|\]|v|\>|\<|no|horizontal|orthogonal|vertical|any|all|no|some|moving|stationary|parallel|perpendicular|action|move|action|create|destroy|cantmove|sfx0|sfx1|sfx2|sfx3|Sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10|cancel|checkpoint|restart|win|message|again|undo|restart|titlescreen|startgame|cancel|endgame|startlevel|endlevel|showmessage|closemessage)/;
    const keyword_array = ['checkpoint','objects', 'collisionlayers', 'legend', 'sounds', 'rules', '...','winconditions', 'levels','|','[',']','up', 'down', 'left', 'right', 'late','rigid', '^','v','\>','\<','no','randomdir','random', 'horizontal', 'vertical','any', 'all', 'no', 'some', 'moving','stationary','parallel','perpendicular','action','message', "move", "action", "create", "destroy", "cantmove", "sfx0", "sfx1", "sfx2", "sfx3", "Sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10", "cancel", "checkpoint", "restart", "win", "message", "again", "undo", "restart", "titlescreen", "startgame", "cancel", "endgame", "startlevel", "endlevel", "showmessage", "closemessage"];

    function errorFallbackMatchToken(stream){
        var match=stream.match(reg_match_until_commentstart_or_whitespace, true);
        if (match===null){
            //just in case, I don't know for sure if it can happen but, just in case I don't 
            //understand unicode and the above doesn't match anything, force some match progress.
            match=stream.match(reg_notcommentstart, true);                                    
        }
        return match;
    }
    
    function processLegendLine(state, mixedCase){
        var ok=true;
        var splits = state.current_line_wip_array;
        if (splits.length===0){
            return;
        }

        if (splits.length === 1) {
            logError('Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".', state.lineNumber);
            ok=false;
        } else if (splits.length%2===0){
            logError(`Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]", but it looks like you have a dangling "${state.current_line_wip_array[state.current_line_wip_array.length-1].toUpperCase()}"?`, state.lineNumber);
            ok=false;
        } else {
            var candname = splits[0];
            
            var alreadyDefined = wordAlreadyDeclared(state,candname);
            if (alreadyDefined!==null){
                logError(`Name "${candname.toUpperCase()}" already in use (on line <a onclick="jumpToLine(${alreadyDefined.lineNumber});" href="javascript:void(0);"><span class="errorTextLineNumber">line ${alreadyDefined.lineNumber}</span></a>).`, state.lineNumber);
                ok=false;
            }

            if (keyword_array.indexOf(candname)>=0) {
                logWarning('You named an object "' + candname.toUpperCase() + '", but this is a keyword. Don\'t do that!', state.lineNumber);
            }
        
        
            for (var i=2; i<splits.length; i+=2){
                var nname = splits[i];
                if (nname===candname){
                    logError("You can't define object " + candname.toUpperCase() + " in terms of itself!", state.lineNumber);
                    ok=false;
                    var idx = splits.indexOf(candname, 2);
                    while (idx >=2){
                        if (idx>=4){
                            splits.splice(idx-1, 2);
                        } else {
                            splits.splice(idx, 2);
                        }
                        idx = splits.indexOf(candname, 2);
                    }          
                }   
                for (var j=2;j<i;j+=2){
                    var oname = splits[j];
                    if(oname===nname){
                        logWarning("You're repeating the object " + oname.toUpperCase() + " here multiple times on the RHS.  This makes no sense.  Don't do that.", state.lineNumber);                        
                    }
                }                       
            } 

            //for every other word, check if it's a valid name
            for (var i=2;i<splits.length;i+=2){
                var defname = splits[i];
                if (defname!==candname){//we already have an error message for that just above.
                    checkNameDefined(state,defname);
                }
            }
        
            if (splits.length === 3) {
                //SYNONYM
                var synonym = [splits[0], splits[2]];
                synonym.lineNumber = state.lineNumber;
                registerOriginalCaseName(state,splits[0],mixedCase,state.lineNumber);
                state.legend_synonyms.push(synonym);
            } else if (splits[3]==='and'){
                //AGGREGATE
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
                                                
                var newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                for (var i = 6; i < splits.length; i += 2) {
                    newlegend = newlegend.concat(substitutor(splits[i]));
                }
                newlegend.lineNumber = state.lineNumber;

                registerOriginalCaseName(state,newlegend[0],mixedCase,state.lineNumber);
                state.legend_aggregates.push(newlegend);
        
            } else if (splits[3]==='or'){
                var malformed=true;

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
                            logError("Cannot define a property (something defined in terms of 'or') in terms of aggregates (something that uses 'and').", state.lineNumber);
                            malformed=false;          
                        }
                    }
                    for (var i=0;i<state.legend_properties.length;i++) {
                        var a = state.legend_properties[i];
                        if (a[0]===n) {  
                            var result = [];
                            for (var j=1;j<a.length;j++){
                                if (a[j]===n){
                                    //error here superfluous, also detected elsewhere (cf 'You can't define object' / #789)
                                    //logError('Error, recursive definition found for '+n+'.', state.lineNumber);                                
                                } else {
                                    result = result.concat(substitutor(a[j]));
                                }
                            }
                            return result;
                        }
                    }
                    return [n];
                };

                for (var i = 5; i < splits.length; i += 2) {
                    if (splits[i].toLowerCase() !== 'or') {
                        malformed = false;
                        break;
                    }
                }
                if (malformed) {
                    var newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                    for (var i = 6; i < splits.length; i += 2) {
                        newlegend.push(splits[i].toLowerCase());
                    }
                    newlegend.lineNumber = state.lineNumber;

                    registerOriginalCaseName(state,newlegend[0],mixedCase,state.lineNumber);
                    state.legend_properties.push(newlegend);
                }
            } else {
                if (ok){
                    //no it's not ok but we don't know why
                    logError('This legend-entry is incorrectly-formatted - it should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)', state.lineNumber);
                    ok=false;
                } 
            }                    
        }
    }

    function processSoundsLine(state){
        if (state.current_line_wip_array.length===0){
            return;
        }
        //if last entry in array is 'ERROR', do nothing
        if (state.current_line_wip_array[state.current_line_wip_array.length-1]==='ERROR'){

        } else {
            //take the first component from each pair in the array
            var soundrow = state.current_line_wip_array;//.map(function(a){return a[0];});
            soundrow.push(state.lineNumber);
            state.sounds.push(soundrow);
        }

    }

    // because of all the early-outs in the token function, this is really just right now attached
    // too places where we can early out during the legend. To make it more versatile we'd have to change 
    // all the early-outs in the token function to flag-assignment for returning outside the case 
    // statement.
    function endOfLineProcessing(state, mixedCase){
        if (state.section==='legend'){
            processLegendLine(state,mixedCase);
        } else if (state.section ==='sounds'){
            processSoundsLine(state);
        }
    }

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
            var rulesCopy = [];

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
            for (var i = 0; i < state.rules.length; i++) {
              rulesCopy.push(state.rules[i].concat([]));
            }

            var original_case_namesCopy = Object.assign({},state.original_case_names);
            var original_line_numbersCopy = Object.assign({},state.original_line_numbers);
            
            var nstate = {
              lineNumber: state.lineNumber,

              objects: objectsCopy,
              collisionLayers: collisionLayersCopy,

              commentLevel: state.commentLevel,
              section: state.section,
              visitedSections: state.visitedSections.concat([]),

              line_should_end: state.line_should_end,
              line_should_end_because: state.line_should_end_because,
              sol_after_comment: state.sol_after_comment,

              objects_candname: state.objects_candname,
              objects_section: state.objects_section,
              objects_spritematrix: state.objects_spritematrix.concat([]),

              tokenIndex: state.tokenIndex,

              current_line_wip_array: state.current_line_wip_array.concat([]),

              legend_synonyms: legend_synonymsCopy,
              legend_aggregates: legend_aggregatesCopy,
              legend_properties: legend_propertiesCopy,

              sounds: soundsCopy,

              rules: rulesCopy,

              names: state.names.concat([]),

              winconditions: winConditionsCopy,

              original_case_names : original_case_namesCopy,
              original_line_numbers : original_line_numbersCopy,

              abbrevNames: state.abbrevNames.concat([]),

              metadata : state.metadata.concat([]),
              metadata_lines: Object.assign({}, state.metadata_lines),

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
                
                state.current_line_wip_array = [];
                stream.string = stream.string.toLowerCase();
                state.tokenIndex=0;
                state.line_should_end = false;
                /*   if (state.lineNumber==undefined) {
                        state.lineNumber=1;
                }
                else {
                    state.lineNumber++;
                }*/

            }
            if (state.sol_after_comment){
                sol = true;
                state.sol_after_comment = false;
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
                } else {
                    logWarning("You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]",state.lineNumber);
                    return 'ERROR';
                }
            }
            if (state.commentLevel > 0) {
                if (sol) {
                    state.sol_after_comment = true;
                }
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
                
                if (stream.eol()){
                    endOfLineProcessing(state,mixedCase);  
                }
                return 'comment';
            }

            stream.eatWhile(/[ \t]/);

            if (sol && stream.eol()) {                
                endOfLineProcessing(state,mixedCase);  
                return blankLineHandle(state);
            }

            if (state.line_should_end && !stream.eol()) {
                logError('Only comments should go after ' + state.line_should_end_because + ' on a line.', state.lineNumber);
                stream.skipToEnd();
                return 'ERROR';
            }            

            //MATCH '==="s AT START OF LINE
            if (sol && stream.match(reg_equalsrow, true)) {
                state.line_should_end = true;
                state.line_should_end_because = 'a bunch of equals signs (\'===\')';
                return 'EQUALSBIT';
            }

            //MATCH SECTION NAME
            var sectionNameMatches = stream.match(reg_sectionNames, true);
            if (sol && sectionNameMatches ) {

                state.section = sectionNameMatches[0].trim();
                if (state.visitedSections.indexOf(state.section) >= 0) {
                    logError('cannot duplicate sections (you tried to duplicate \"' + state.section.toUpperCase() + '").', state.lineNumber);
                }
                state.line_should_end = true;
                state.line_should_end_because = `a section name ("${state.section.toUpperCase()}")`;
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
                        logError('section "' + state.section.toUpperCase() + '" is out of order, must follow  "' + sectionNames[sectionIndex - 1].toUpperCase() + '" (or it could be that the section "'+sectionNames[sectionIndex - 1].toUpperCase()+`"is just missing totally.  You have to include all section headings, even if the section itself is empty).`, state.lineNumber);                            
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
                
                endOfLineProcessing(state,mixedCase);  
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
                        var match_name = sol ? stream.match(reg_name, true) : stream.match(/[^\p{Z}\s\()]+[\p{Z}\s]*/u,true);
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
                                registerOriginalCaseName(state,candname,mixedCase,state.lineNumber);
                                state.objects[state.objects_candname] = {
                                                                        lineNumber: state.lineNumber,
                                                                        colors: [],
                                                                        spritematrix: []
                                                                    };

                            } else {
                                //set up alias
                                registerOriginalCaseName(state,candname,mixedCase,state.lineNumber);
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
                                    return 'MULTICOLOR'+match_color[0];
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
                                logWarning('Sprites must be 5 wide and 5 high.', state.lineNumber);
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
            case 'legend':
                {
                    var resultToken="";
                    var match_name=null;
                    if (state.tokenIndex === 0) {
                        match_name=stream.match(/[^=\p{Z}\s\(]*(\p{Z}\s)*/u, true);
                        var new_name=match_name[0].trim();
                        
                        if (wordAlreadyDeclared(state,new_name))
                        {
                            resultToken =  'ERROR';
                        } else {
                            resultToken =  'NAME';    
                        }

                        //if name already declared, we have a problem                            
                        state.tokenIndex++;
                    } else if (state.tokenIndex === 1) {
                        match_name = stream.match(/=/u,true);                              
                        if (match_name===null||match_name[0].trim()!=="="){
                            logError(`In the legend, define new items using the equals symbol - declarations must look like "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".`, state.lineNumber);
                            stream.match(reg_notcommentstart, true);
                            resultToken = 'ERROR';
                            match_name=["ERROR"];//just to reduce the chance of crashes
                        }
                        stream.match(/[\p{Z}\s]*/u, true);
                        state.tokenIndex++;
                        resultToken = 'ASSSIGNMENT';
                    } else if (state.tokenIndex >= 3 && ((state.tokenIndex % 2) === 1)) {
                        //matches AND/OR
                        match_name = stream.match(reg_name, true);
                        if (match_name === null) {
                            logError("Something bad's happening in the LEGEND", state.lineNumber);
                            match=stream.match(reg_notcommentstart, true);
                            resultToken = 'ERROR';
                        } else {
                            var candname = match_name[0].trim();
                            if (candname === "and" || candname === "or"){                                             
                                resultToken =  'LOGICWORD';
                                if (state.tokenIndex>=5){
                                    if (candname !== state.current_line_wip_array[3]){
                                        logError("Hey! You can't go mixing ANDs and ORs in a single legend entry.", state.lineNumber);
                                        resultToken = 'ERROR';
                                    }
                                }
                            } else {
                                logError(`Expected and 'AND' or an 'OR' here, but got ${candname.toUpperCase()} instead. In the legend, define new items using the equals symbol - declarations must look like 'A = B' or 'A = B and C' or 'A = B or C'.`, state.lineNumber);
                                resultToken = 'ERROR';
                                // match_name=["and"];//just to reduce the chance of crashes
                            }
                        }
                        state.tokenIndex++;
                    }
                    else {
                        match_name = stream.match(reg_name, true);
                        if (match_name === null) {
                            logError("Something bad's happening in the LEGEND", state.lineNumber);
                            match=stream.match(reg_notcommentstart, true);
                            resultToken = 'ERROR';
                        } else {
                            var candname = match_name[0].trim();
                            if (wordAlreadyDeclared(state,candname))
                            {
                                resultToken =  'NAME';    
                            } else {
                                resultToken =  'ERROR';
                            }
                            state.tokenIndex++;

                        }
                    }

                    if (match_name!==null){
                        state.current_line_wip_array.push(match_name[0].trim());
                    }
                    
                    if (stream.eol()){
                        processLegendLine(state,mixedCase);
                    }               

                    return resultToken;
                    break;
                }
            case 'sounds':
                {
                    /*
                    SOUND DEFINITION:
                        SOUNDEVENT ~ INT (Sound events take precedence if there's name overlap)
                        OBJECT_NAME
                            NONDIRECTIONAL_VERB ~ INT
                            DIRECTIONAL_VERB
                                INT
                                DIR+ ~ INT
                    */
                    var tokentype="";

                    if (state.current_line_wip_array.length>0 && state.current_line_wip_array[state.current_line_wip_array.length-1]==='ERROR'){
                        // match=stream.match(reg_notcommentstart, true);
                        //if there was an error earlier on the line just try to do greedy matching here
                        var match = null;

                        //events
                        if (match === null) { 
                            match = stream.match(reg_soundevents, true);
                            if (match !== null) { 
                                tokentype = 'SOUNDEVENT';
                            }
                        }

                        //verbs
                        if (match === null) { 
                            match = stream.match(reg_soundverbs, true);
                            if (match !== null) {
                                tokentype = 'SOUNDVERB';
                            }
                        }
                        //directions
                        if (match === null) { 
                            match = stream.match(reg_sounddirectionindicators, true);
                            if (match !== null) {
                                tokentype = 'DIRECTION';
                            }
                        }

                        //sound seeds
                        if (match === null) {                                           
                            var match = stream.match(reg_soundseed, true);
                            if (match !== null)
                            {
                                tokentype = 'SOUND';
                            }
                        }

                        //objects
                        if (match === null) { 
                            match = stream.match(reg_name, true);
                            if (match !== null) {
                                if (wordAlreadyDeclared(state, match[0])){
                                    tokentype = 'NAME';
                                } else {
                                    tokentype = 'ERROR';                   
                                }
                            }                          
                        }

                        //error
                        if (match === null) { 
                            match = errorFallbackMatchToken(stream);
                            tokentype = 'ERROR';                            
                        }


                    } else if (state.current_line_wip_array.length===0){
                        //can be OBJECT_NAME or SOUNDEVENT
                        var match = stream.match(reg_soundevents, true);
                        if (match == null){
                            match = stream.match(reg_name, true);
                            if (match == null ){
                                tokentype = 'ERROR';
                                match=errorFallbackMatchToken(stream);
                                state.current_line_wip_array.push("ERROR");
                                logWarning("Was expecting a sound event (like SFX3, or ENDLEVEL) or an object name, but didn't find either.", state.lineNumber);                        
                            } else {
                                var matched_name = match[0].trim();
                                if (!wordAlreadyDeclared(state, matched_name)){                 
                                    tokentype = 'ERROR';
                                    state.current_line_wip_array.push("ERROR");
                                    logError(`unexpected sound token "${matched_name}".`, state.lineNumber);
                                } else {                                    
                                    tokentype = 'NAME';
                                    state.current_line_wip_array.push([matched_name,tokentype]);    
                                    state.tokenIndex++;
                                }
                            }
                        } else {
                            tokentype = 'SOUNDEVENT';
                            state.current_line_wip_array.push([match[0].trim(),tokentype]);  
                            state.tokenIndex++;  
                        }

                    } else if (state.current_line_wip_array.length===1) {
                        var is_soundevent = state.current_line_wip_array[0][1] === 'SOUNDEVENT';

                        if (is_soundevent){                            
                            var match = stream.match(reg_soundseed, true);
                            if (match !== null)
                            {
                                tokentype = 'SOUND';
                                state.current_line_wip_array.push([match[0].trim(),tokentype]);
                                state.tokenIndex++;
                            } else {
                                match=errorFallbackMatchToken(stream);
                                logError("Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else.", state.lineNumber);                                
                                tokentype = 'ERROR';
                                state.current_line_wip_array.push("ERROR");
                            }
                        } else {
                            //[0] is object name
                            //it's a sound verb
                            var match = stream.match(reg_soundverbs, true);
                            if (match !== null){
                                tokentype = 'SOUNDVERB';
                                state.current_line_wip_array.push([match[0].trim(),tokentype]);
                                state.tokenIndex++;
                            } else {
                                match=errorFallbackMatchToken(stream);
                                logError("Was expecting a soundverb here (MOVE, DESTROY, CANTMOVE, or the like), but found something else.", state.lineNumber);                                
                                tokentype = 'ERROR';
                                state.current_line_wip_array.push("ERROR");
                            }
                            
                        }
                    } else {
                        var is_soundevent = state.current_line_wip_array[0][1] === 'SOUNDEVENT';
                        if (is_soundevent){
                            match=errorFallbackMatchToken(stream);
                            logError(`I wasn't expecting anything after the sound declaration ${state.current_line_wip_array[state.current_line_wip_array.length-1][0].toUpperCase()} on this line, so I don't know what to do with "${match[0].trim().toUpperCase()}" here.`, state.lineNumber);
                            tokentype = 'ERROR';
                            state.current_line_wip_array.push("ERROR");
                        } else {                            
                            //if there's a seed on the right, any additional content is superfluous
                            var is_seedonright = state.current_line_wip_array[state.current_line_wip_array.length-1][1] === 'SOUND';
                            if (is_seedonright){
                                match=errorFallbackMatchToken(stream);
                                logError(`I wasn't expecting anything after the sound declaration ${state.current_line_wip_array[state.current_line_wip_array.length-1][0].toUpperCase()} on this line, so I don't know what to do with "${match[0].trim().toUpperCase()}" here.`, state.lineNumber);
                                tokentype = 'ERROR';
                                state.current_line_wip_array.push("ERROR");
                            } else {
                                var directional_verb = soundverbs_directional.indexOf(state.current_line_wip_array[1][0])>=0;    
                                if (directional_verb){  
                                    //match seed or direction                          
                                    var is_direction = stream.match(reg_sounddirectionindicators, true);
                                    if (is_direction !== null){
                                        tokentype = 'DIRECTION';
                                        state.current_line_wip_array.push([is_direction[0].trim(),tokentype]);
                                        state.tokenIndex++;
                                    } else {
                                        var is_seed = stream.match(reg_soundseed, true);
                                        if (is_seed !== null){
                                            tokentype = 'SOUND';
                                            state.current_line_wip_array.push([is_seed[0].trim(),tokentype]);
                                            state.tokenIndex++;
                                        } else {
                                            match=errorFallbackMatchToken(stream);
                                            //depending on whether the verb is directional or not, we log different errors
                                            logError(`Ah I was expecting direction or a sound seed here after ${state.current_line_wip_array[state.current_line_wip_array.length-1][0].toUpperCase()}, but I don't know what to make of "${match[0].trim().toUpperCase()}".`, state.lineNumber);
                                            tokentype = 'ERROR';
                                            state.current_line_wip_array.push("ERROR");
                                        }
                                    }
                                } else {
                                    //only match seed
                                    var is_seed = stream.match(reg_soundseed, true);
                                    if (is_seed !== null){
                                        tokentype = 'SOUND';
                                        state.current_line_wip_array.push([is_seed[0].trim(),tokentype]);
                                        state.tokenIndex++;
                                    } else {
                                        match=errorFallbackMatchToken(stream);
                                        //depending on whether the verb is directional or not, we log different errors
                                        logError(`Ah I was expecting a sound seed here after ${state.current_line_wip_array[state.current_line_wip_array.length-1][0].toUpperCase()}, but I don't know what to make of "${match[0].trim().toUpperCase()}".`, state.lineNumber);
                                        tokentype = 'ERROR';
                                        state.current_line_wip_array.push("ERROR");
                                    }
                                }
                            }
                        }
                    }

                    if (stream.eol()){
                        processSoundsLine(state);
                    }     

                    return tokentype;

                    break;
                }
            case 'collisionlayers':
                {
                    if (sol) {
                        //create new collision layer
                        state.collisionLayers.push([]);
                        //empty current_line_wip_array
                        state.current_line_wip_array = [];
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
                                    var result = [];
                                    for (var j=1;j<a.length;j++){
                                        if (a[j]===n){
                                            //error here superfluous, also detected elsewhere (cf 'You can't define object' / #789)
                                            //logError('Error, recursive definition found for '+n+'.', state.lineNumber);                                
                                        } else {
                                            result = result.concat(substitutor(a[j]));
                                        }
                                    }
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
                        var foundSelves=[];
                        for (var i=0;i<ar.length;i++){
                            var tcandname = ar[i];
                            for (var j=0;j<=state.collisionLayers.length-1;j++){
                                var clj = state.collisionLayers[j];
                                if (clj.indexOf(tcandname)>=0){
                                    if (j!==state.collisionLayers.length-1){
                                        foundOthers.push(j);
                                    } else {
                                        foundSelves.push(j);
                                    }
                                }
                            }
                        }
                        if (foundOthers.length>0){
                            var warningStr = 'Object "'+candname.toUpperCase()+'" included in multiple collision layers ( layers ';
                            for (var i=0;i<foundOthers.length;i++){
                                warningStr+="#"+(foundOthers[i]+1)+", ";
                            }
                            warningStr+="#"+state.collisionLayers.length;
                            logWarning(warningStr +' ). You should fix this!',state.lineNumber);                                        
                        }

                        if (state.current_line_wip_array.indexOf(candname)>=0){
                            var warningStr = 'Object "'+candname.toUpperCase()+'" included explicitly multiple times in the same layer. Don\'t do that innit.';
                            logWarning(warningStr,state.lineNumber);         
                        }
                        state.current_line_wip_array.push(candname);

                        state.collisionLayers[state.collisionLayers.length - 1] = state.collisionLayers[state.collisionLayers.length - 1].concat(ar);
                        if (ar.length>0) {
                            return 'NAME';                            
                        } else {
                            return 'ERROR';
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
                    if (stream.match(/[\p{Z}\s]*->[\p{Z}\s]*/u, true)) {
                        return 'ARROW';
                    }
                    if (ch === '[' || ch === '|' || ch === ']' || ch==='+') {
                        if (ch!=='+') {
                            state.tokenIndex = 1;
                        }
                        stream.next();
                        stream.match(/[\p{Z}\s]*/u, true);
                        return 'BRACKET';
                    } else {
                        var m = stream.match(/[^\[\|\]\p{Z}\s]*/u, true)[0].trim();

                        if (state.tokenIndex===0&&reg_loopmarker.exec(m)) {
                            return 'BRACKET';
                        } else if (state.tokenIndex === 0 && reg_ruledirectionindicators.exec(m)) {
                            stream.match(/[\p{Z}\s]*/u, true);
                            return 'DIRECTION';
                        } else if (state.tokenIndex === 1 && reg_directions.exec(m)) {
                            stream.match(/[\p{Z}\s]*/u, true);
                            return 'DIRECTION';
                        } else {
                            if (state.names.indexOf(m) >= 0) {
                                if (sol) {
                                    logError('Objects cannot appear outside of square brackets in rules, only directions can.', state.lineNumber);
                                    return 'ERROR';
                                } else {
                                    stream.match(/[\p{Z}\s]*/u, true);
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
                        var splitted = tokenized[0].split(/[\p{Z}\s]/u);
                        var filtered = splitted.filter(function(v) {return v !== ''});
                        filtered.push(state.lineNumber);
                        
                        state.winconditions.push(filtered);
                        state.tokenIndex = -1;
                    }
                    state.tokenIndex++;

                    var match = stream.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);
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
                                logError('Expecting the start of a win condition ("ALL","SOME","NO") but got "'+candword.toUpperCase()+"'.", state.lineNumber);
                                return 'ERROR';
                            }
                        }
                        else if (state.tokenIndex === 2) {
                            if (candword != 'on') {
                                logError('Expecting the word "ON" but got "'+candword.toUpperCase()+"\".", state.lineNumber);
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
                        } else {
                            logError("Error in win condition: I don't know what to do with "+candword.toUpperCase()+".", state.lineNumber);
                            return 'ERROR';
                        }
                    }
                    break;
                }
            case 'levels':
                {
                    if (sol)
                    {
                        if (stream.match(/[\p{Z}\s]*message\b[\p{Z}\s]*/u, true)) {
                            state.tokenIndex = -4;//-4/2 = message/level
                            var newdat = ['\n', mixedCase.slice(stream.pos).trim(),state.lineNumber];
                            if (state.levels[state.levels.length - 1].length == 0) {
                                state.levels.splice(state.levels.length - 1, 0, newdat);
                            } else {
                                state.levels.push(newdat);
                            }
                            return 'MESSAGE_VERB';//a duplicate of the previous section as a legacy thing for #589 
                        } else if (stream.match(/[\p{Z}\s]*message[\p{Z}\s]*/u, true)) {//duplicating previous section because of #589
                            logWarning("You probably meant to put a space after 'message' innit.  That's ok, I'll still interpret it as a message, but you probably want to put a space there.",state.lineNumber);
                            state.tokenIndex = -4;//-4/2 = message/level
                            var newdat = ['\n', mixedCase.slice(stream.pos).trim(),state.lineNumber];
                            if (state.levels[state.levels.length - 1].length == 0) {
                                state.levels.splice(state.levels.length - 1, 0, newdat);
                            } else {
                                state.levels.push(newdat);
                            }
                            return 'MESSAGE_VERB';
                        } else {
                            var matches = stream.match(reg_notcommentstart, false);
                            if (matches===null || matches.length===0){
                                logError("Detected a comment where I was expecting a level. Oh gosh; if this is to do with you using '(' as a character in the legend, please don't do that ^^",state.lineNumber);
                                state.commentLevel++;
                                stream.skipToEnd();
                                return 'comment';
                            } else {
                                var line = matches[0].trim();
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
                        }
                    } else {
                        if (state.tokenIndex == -4) {
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
                        var match = stream.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);	                    
                        if (match!==null) {
                            var token = match[0].trim();
                            if (sol) {
                                if (['title','author','homepage','background_color','text_color','key_repeat_interval','realtime_interval','again_interval','flickscreen','zoomscreen','color_palette','youtube'].indexOf(token)>=0) {
                                    
                                    if (token==='author' || token==='homepage' || token==='title') {
                                        stream.string=mixedCase;
                                    }

                                    if (token==="youtube") {
                                        logWarning("Unfortunately, YouTube support hasn't been working properly for a long time - it was always a hack and it hasn't gotten less hacky over time, so I can no longer pretend to support it.",state.lineNumber);
                                    }
                                    
                                    var m2 = stream.match(reg_notcommentstart, false);
                                    
                                    if(m2!==null) {
                                        state.metadata.push(token);
                                        state.metadata.push(m2[0].trim());  
                                        if (token in state.metadata_lines){
                                            var otherline = state.metadata_lines[token];
                                            logWarning(`You've already defined a ${token.toUpperCase()} in the prelude on line <a onclick="jumpToLine(${otherline})>${otherline}</a>.`,state.lineNumber);
                                        }
                                        state.metadata_lines[token]=state.lineNumber;                                                                                    
                                    } else {
                                        logError('MetaData "'+token+'" needs a value.',state.lineNumber);
                                    }
                                    state.tokenIndex=1;
                                    return 'METADATA';
                                } else if ( ['run_rules_on_level_start','norepeat_action','require_player_movement','debug','verbose_logging','throttle_movement','noundo','noaction','norestart','scanline'].indexOf(token)>=0) {
                                    state.metadata.push(token);
                                    state.metadata.push("true");
                                    state.tokenIndex=-1;


                                    var m2 = stream.match(reg_notcommentstart, false);
                                    
                                    if(m2!==null) {
                                        var extra = m2[0].trim();      
                                        logWarning('MetaData '+token.toUpperCase()+' doesn\'t take any parameters, but you went and gave it "'+extra+'".',state.lineNumber);                                      
                                    } 

                                    return 'METADATA';
                                } else  {
                                    logError('Unrecognised stuff in the prelude.', state.lineNumber);
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
                        state.tokenIndex++;

                        var key = state.metadata[state.metadata.length-2];
                        var val = state.metadata[state.metadata.length-1];

                        if( state.tokenIndex>2){
                            logWarning("Error: you can't embed comments in metadata values. Anything after the comment will be ignored.",state.lineNumber);
                            return 'ERROR';
                        }
						if (key === "background_color" || key === "text_color"){
							var candcol = val.trim().toLowerCase();
                            if (candcol in colorPalettes.arnecolors) {
                                return 'COLOR COLOR-' + candcol.toUpperCase();
                            } else if (candcol==="transparent") {
                                return 'COLOR FADECOLOR';
                            } else if ( (candcol.length===4) || (candcol.length===7)) {
                                var color = candcol.match(/#[0-9a-fA-F]+/);
                                if (color!==null){                
                                    return 'MULTICOLOR'+color[0];
                                }
                            }
                                                       
						}                        
                        return "METADATATEXT";
                    }
                    break;
                }
            }
        

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

                line_should_end: false,
                line_should_end_because: '',
                sol_after_comment: false,

                objects_candname: '',
                objects_section: 0, //whether reading name/color/spritematrix
                objects_spritematrix: [],

                collisionLayers: [],

                tokenIndex: 0,

                current_line_wip_array: [],

                legend_synonyms: [],
                legend_aggregates: [],
                legend_properties: [],

                sounds: [],
                rules: [],

                names: [],

                winconditions: [],
                metadata: [],
                metadata_lines: {},

                original_case_names: {},
                original_line_numbers: {},

                abbrevNames: [],

                levels: [[]],

                subsection: ''
            };
        }
    };
};

window.CodeMirror.defineMode('puzzle', codeMirrorFn);
