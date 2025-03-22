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


const MAX_ERRORS_FOR_REAL = 100;

let compiling = false;
let errorStrings = [];//also stores warning strings
let errorCount = 0;//only counts errors

function TooManyErrors() {
    const message = compiling ? "Too many errors/warnings; aborting compilation." : "Too many errors/warnings; noping out.";
    consolePrint(message, true);
    throw new Error(message);
}

function logErrorCacheable(str, lineNumber, urgent) {
    if (compiling || urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str, urgent);
        }
        let errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
        if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
        } else {
            consolePrint(errorString);
            errorStrings.push(errorString);
            errorCount++;
            if (errorStrings.length > MAX_ERRORS_FOR_REAL) {
                TooManyErrors();
            }
        }
    }
}

function logError(str, lineNumber, urgent) {
    if (compiling || urgent) {
        if (lineNumber === undefined) {
            return logErrorNoLine(str, urgent);
        }
        let errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="errorText">' + str + '</span>';
        if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
        } else {
            consolePrint(errorString, true);
            errorStrings.push(errorString);
            errorCount++;
            if (errorStrings.length > MAX_ERRORS_FOR_REAL) {
                TooManyErrors();
            }
        }
    }
}

function logWarning(str, lineNumber, urgent) {
    if (compiling || urgent) {
        if (lineNumber === undefined) {
            return logWarningNoLine(str, urgent);
        }
        let errorString = '<a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + lineNumber.toString() + '</span></a> : ' + '<span class="warningText">' + str + '</span>';
        if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
        } else {
            consolePrint(errorString, true);
            errorStrings.push(errorString);
            if (errorStrings.length > MAX_ERRORS_FOR_REAL) {
                TooManyErrors();
            }
        }
    }
}

function logWarningNoLine(str, urgent) {
    if (compiling || urgent) {
        let errorString = '<span class="warningText">' + str + '</span>';
        if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
        } else {
            consolePrint(errorString, true);
            errorStrings.push(errorString);
            errorCount++;
            if (errorStrings.length > MAX_ERRORS_FOR_REAL) {
                TooManyErrors();
            }
        }
    }
}


function logErrorNoLine(str, urgent) {
    if (compiling || urgent) {
        let errorString = '<span class="errorText">' + str + '</span>';
        if (errorStrings.indexOf(errorString) >= 0 && !urgent) {
            //do nothing, duplicate error
        } else {
            consolePrint(errorString, true);
            errorStrings.push(errorString);
            errorCount++;
            if (errorStrings.length > MAX_ERRORS_FOR_REAL) {
                TooManyErrors();
            }
        }
    }
}

function blankLineHandle(state) {
    if (state.section === 'levels') {
        if (state.levels[state.levels.length - 1].length > 0) {
            state.levels.push([]);
        }
    } else if (state.section === 'objects') {
        state.objects_section = 0;
    }
}

//returns null if not delcared, otherwise declaration
//note to self: I don't think that aggregates or properties know that they're aggregates or properties in and of themselves.
function wordAlreadyDeclared(state, n) {
    n = n.toLowerCase();
    if (n in state.objects) {
        return state.objects[n];
    }
    for (let i = 0; i < state.legend_aggregates.length; i++) {
        let a = state.legend_aggregates[i];
        if (a[0] === n) {
            return state.legend_aggregates[i];
        }
    }
    for (let i = 0; i < state.legend_properties.length; i++) {
        let a = state.legend_properties[i];
        if (a[0] === n) {
            return state.legend_properties[i];
        }
    }
    for (let i = 0; i < state.legend_synonyms.length; i++) {
        let a = state.legend_synonyms[i];
        if (a[0] === n) {
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

            let output = Object(target);
            for (let index = 1; index < arguments.length; index++) {
                let source = arguments[index];
                if (source !== undefined && source !== null) {
                    for (let nextKey in source) {
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


let codeMirrorFn = function () {
    'use strict';

    function checkNameDefined(state, candname) {
        if (state.objects[candname] !== undefined) {
            return;
        }
        for (let i = 0; i < state.legend_synonyms.length; i++) {
            let entry = state.legend_synonyms[i];
            if (entry[0] == candname) {
                return;
            }
        }
        for (let i = 0; i < state.legend_aggregates.length; i++) {
            let entry = state.legend_aggregates[i];
            if (entry[0] == candname) {
                return;
            }
        }
        for (let i = 0; i < state.legend_properties.length; i++) {
            let entry = state.legend_properties[i];
            if (entry[0] == candname) {
                return;
            }
        }

        logError(`You're talking about ${candname.toUpperCase()} but it's not defined anywhere.`, state.lineNumber);
    }

    function registerOriginalCaseName(state, candname, mixedCase, lineNumber) {

        function escapeRegExp(str) {
            return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
        }

        let nameFinder = new RegExp("\\b" + escapeRegExp(candname) + "\\b", "i")
        let match = mixedCase.match(nameFinder);
        if (match != null) {
            state.original_case_names[candname] = match[0];
            state.original_line_numbers[candname] = lineNumber;
        }
    }

    function errorFallbackMatchToken(stream) {
        let match = stream.match(reg_match_until_commentstart_or_whitespace, true);
        if (match === null) {
            //just in case, I don't know for sure if it can happen but, just in case I don't 
            //understand unicode and the above doesn't match anything, force some match progress.
            match = stream.match(reg_notcommentstart, true);
        }
        return match;
    }

    function processLegendLine(state, mixedCase) {
        let ok = true;
        let splits = state.current_line_wip_array;
        if (splits.length === 0) {
            return;
        }

        if (splits.length === 1) {
            logError('Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".', state.lineNumber);
            ok = false;
        } else if (splits.length % 2 === 0) {
            logError(`Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]", but it looks like you have a dangling "${state.current_line_wip_array[state.current_line_wip_array.length - 1].toUpperCase()}"?`, state.lineNumber);
            ok = false;
        } else {
            let candname = splits[0];

            let alreadyDefined = wordAlreadyDeclared(state, candname);
            if (alreadyDefined !== null) {
                logError(`Name "${candname.toUpperCase()}" already in use (on line <a onclick="jumpToLine(${alreadyDefined.lineNumber});" href="javascript:void(0);"><span class="errorTextLineNumber">line ${alreadyDefined.lineNumber}</span></a>).`, state.lineNumber);
                ok = false;
            }

            if (keyword_array.indexOf(candname) >= 0) {
                logWarning('You named an object "' + candname.toUpperCase() + '", but this is a keyword. Don\'t do that!', state.lineNumber);
            }


            for (let i = 2; i < splits.length; i += 2) {
                let nname = splits[i];
                if (nname === candname) {
                    logError("You can't define object " + candname.toUpperCase() + " in terms of itself!", state.lineNumber);
                    ok = false;
                    let idx = splits.indexOf(candname, 2);
                    while (idx >= 2) {
                        if (idx >= 4) {
                            splits.splice(idx - 1, 2);
                        } else {
                            splits.splice(idx, 2);
                        }
                        idx = splits.indexOf(candname, 2);
                    }
                }
                for (let j = 2; j < i; j += 2) {
                    let oname = splits[j];
                    if (oname === nname) {
                        logWarning("You're repeating the object " + oname.toUpperCase() + " here multiple times on the RHS.  This makes no sense.  Don't do that.", state.lineNumber);
                    }
                }
            }

            //for every other word, check if it's a valid name
            for (let i = 2; i < splits.length; i += 2) {
                let defname = splits[i];
                if (defname !== candname) {//we already have an error message for that just above.
                    checkNameDefined(state, defname);
                }
            }

            if (splits.length === 3) {
                //SYNONYM
                let synonym = [splits[0], splits[2]];
                synonym.lineNumber = state.lineNumber;
                registerOriginalCaseName(state, splits[0], mixedCase, state.lineNumber);
                state.legend_synonyms.push(synonym);
            } else if (splits[3] === 'and') {
                //AGGREGATE
                let substitutor = function (n) {
                    n = n.toLowerCase();
                    if (n in state.objects) {
                        return [n];
                    }
                    for (let i = 0; i < state.legend_synonyms.length; i++) {
                        let a = state.legend_synonyms[i];
                        if (a[0] === n) {
                            return substitutor(a[1]);
                        }
                    }
                    for (let i = 0; i < state.legend_aggregates.length; i++) {
                        let a = state.legend_aggregates[i];
                        if (a[0] === n) {
                            return [].concat.apply([], a.slice(1).map(substitutor));
                        }
                    }
                    for (let i = 0; i < state.legend_properties.length; i++) {
                        let a = state.legend_properties[i];
                        if (a[0] === n) {
                            logError("Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').", state.lineNumber);
                            ok = false;
                            return [n];
                        }
                    }
                    //seems like this shouldn't be reachable?
                    return [n];
                };

                let newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                for (let i = 6; i < splits.length; i += 2) {
                    newlegend = newlegend.concat(substitutor(splits[i]));
                }
                newlegend.lineNumber = state.lineNumber;

                registerOriginalCaseName(state, newlegend[0], mixedCase, state.lineNumber);
                state.legend_aggregates.push(newlegend);

            } else if (splits[3] === 'or') {
                let malformed = false;

                let substitutor = function (n) {

                    n = n.toLowerCase();
                    if (n in state.objects) {
                        return [n];
                    }

                    for (let i = 0; i < state.legend_synonyms.length; i++) {
                        let a = state.legend_synonyms[i];
                        if (a[0] === n) {
                            return substitutor(a[1]);
                        }
                    }
                    for (let i = 0; i < state.legend_aggregates.length; i++) {
                        let a = state.legend_aggregates[i];
                        if (a[0] === n) {
                            logError(`Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and').  In this case, you can't define "${splits[0]}" in terms of "${n}".`, state.lineNumber);
                            malformed = true;
                            return [];
                        }
                    }
                    for (let i = 0; i < state.legend_properties.length; i++) {
                        let a = state.legend_properties[i];
                        if (a[0] === n) {
                            let result = [];
                            for (let j = 1; j < a.length; j++) {
                                if (a[j] === n) {
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

                for (let i = 5; i < splits.length; i += 2) {
                    if (splits[i].toLowerCase() !== 'or') {
                        malformed = true;
                        break;
                    }
                }
                if (!malformed) {
                    let newlegend = [splits[0]].concat(substitutor(splits[2])).concat(substitutor(splits[4]));
                    for (let i = 6; i < splits.length; i += 2) {
                        newlegend.push(splits[i].toLowerCase());
                    }
                    newlegend.lineNumber = state.lineNumber;

                    registerOriginalCaseName(state, newlegend[0], mixedCase, state.lineNumber);
                    state.legend_properties.push(newlegend);
                }
            } else {
                if (ok) {
                    //no it's not ok but we don't know why
                    logError('This legend-entry is incorrectly-formatted - it should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)', state.lineNumber);
                    ok = false;
                }
            }
        }
    }

    function processSoundsLine(state) {
        if (state.current_line_wip_array.length === 0) {
            return;
        }
        //if last entry in array is 'ERROR', do nothing
        if (state.current_line_wip_array[state.current_line_wip_array.length - 1] === 'ERROR') {

        } else {
            //take the first component from each pair in the array
            let soundrow = state.current_line_wip_array;//.map(function(a){return a[0];});
            soundrow.push(state.lineNumber);
            state.sounds.push(soundrow);
        }

    }

    // because of all the early-outs in the token function, this is really just right now attached
    // too places where we can early out during the legend. To make it more versatile we'd have to change 
    // all the early-outs in the token function to flag-assignment for returning outside the case 
    // statement.
    function endOfLineProcessing(state, mixedCase) {
        if (state.section === 'legend') {
            processLegendLine(state, mixedCase);
        } else if (state.section === 'sounds') {
            processSoundsLine(state);
        }
    }

    //  let keywordRegex = new RegExp("\\b(("+cons.join(")|(")+"))$", 'i');

    let fullSpriteMatrix = [
        '00000',
        '00000',
        '00000',
        '00000',
        '00000'
    ];

    return {
        copyState: function (state) {
            let objectsCopy = {};
            for (let i in state.objects) {
                if (state.objects.hasOwnProperty(i)) {
                    let o = state.objects[i];
                    objectsCopy[i] = {
                        colors: o.colors.concat([]),
                        lineNumber: o.lineNumber,
                        spritematrix: o.spritematrix.concat([])
                    }
                }
            }

            let collisionLayersCopy = [];
            for (let i = 0; i < state.collisionLayers.length; i++) {
                collisionLayersCopy.push(state.collisionLayers[i].concat([]));
            }

            let legend_synonymsCopy = [];
            let legend_aggregatesCopy = [];
            let legend_propertiesCopy = [];
            let soundsCopy = [];
            let levelsCopy = [];
            let winConditionsCopy = [];
            let rulesCopy = [];

            for (let i = 0; i < state.legend_synonyms.length; i++) {
                legend_synonymsCopy.push(state.legend_synonyms[i].concat([]));
            }
            for (let i = 0; i < state.legend_aggregates.length; i++) {
                legend_aggregatesCopy.push(state.legend_aggregates[i].concat([]));
            }
            for (let i = 0; i < state.legend_properties.length; i++) {
                legend_propertiesCopy.push(state.legend_properties[i].concat([]));
            }
            for (let i = 0; i < state.sounds.length; i++) {
                soundsCopy.push(state.sounds[i].concat([]));
            }
            for (let i = 0; i < state.levels.length; i++) {
                levelsCopy.push(state.levels[i].concat([]));
            }
            for (let i = 0; i < state.winconditions.length; i++) {
                winConditionsCopy.push(state.winconditions[i].concat([]));
            }
            for (let i = 0; i < state.rules.length; i++) {
                rulesCopy.push(state.rules[i].concat([]));
            }

            let original_case_namesCopy = Object.assign({}, state.original_case_names);
            let original_line_numbersCopy = Object.assign({}, state.original_line_numbers);

            let nstate = {
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

                original_case_names: original_case_namesCopy,
                original_line_numbers: original_line_numbersCopy,

                abbrevNames: state.abbrevNames.concat([]),

                metadata: state.metadata.concat([]),
                metadata_lines: Object.assign({}, state.metadata_lines),

                levels: levelsCopy,

                STRIDE_OBJ: state.STRIDE_OBJ,
                STRIDE_MOV: state.STRIDE_MOV
            };

            return nstate;
        },
        blankLine: function (state) {
            if (state.section === 'levels') {
                if (state.levels[state.levels.length - 1].length > 0) {
                    state.levels.push([]);
                }
            }
        },
        token: function (stream, state) {
            let mixedCase = stream.string;
            let sol = stream.sol();
            if (sol) {
                state.lineNumber++;
                state.current_line_wip_array = [];
                stream.string = stream.string.toLowerCase();
                state.tokenIndex = 0;
                state.line_should_end = false;
            }
            if (state.sol_after_comment) {
                sol = true;
                state.sol_after_comment = false;
            }



            stream.eatWhile(/[ \t]/);

            ////////////////////////////////
            // COMMENT PROCESSING BEGIN
            ////////////////////////////////

            //NESTED COMMENTS
            let ch = stream.peek();
            if (ch === '(' && state.tokenIndex !== -4) { // tokenIndex -4 indicates message command
                stream.next();
                state.commentLevel++;
            } else if (ch === ')') {
                stream.next();
                if (state.commentLevel > 0) {
                    state.commentLevel--;
                    if (state.commentLevel === 0) {
                        state.sol_after_comment = true;
                        return 'comment';
                    }
                } else {
                    logWarning("You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]", state.lineNumber);
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

                if (stream.eol()) {
                    endOfLineProcessing(state, mixedCase);
                }
                return 'comment';
            }

            stream.eatWhile(/[ \t]/);

            if (sol && stream.eol()) {
                endOfLineProcessing(state, mixedCase);
                return blankLineHandle(state);
            }

            if (state.line_should_end && !stream.eol()) {
                logError('Only comments should go after ' + state.line_should_end_because + ' on a line.', state.lineNumber);
                stream.skipToEnd();
                return 'ERROR';
            }

            //MATCH '==="s AT START OF LINE
            //for #977 we need to be careful about matching an equals row in the levels section
            //check if the line contains something other than an equals characte or space
            let shouldmatchequals = true;
            if (sol && state.section === "levels") {
                let linestring = stream.string.substring(stream.pos);
                const reg_matchall_whitespace_equals = /^[\p{Z}\s=]*$/u;
                if (!reg_matchall_whitespace_equals.test(linestring)) {
                    shouldmatchequals = false;
                }
            }

            if (sol && (shouldmatchequals && stream.match(reg_equalsrow, true))) {
                state.line_should_end = true;
                state.line_should_end_because = 'a bunch of equals signs (\'===\')';
                return 'EQUALSBIT';
            }

            //MATCH SECTION NAME
            if (state.section !== "levels" /*cf #976 lol*/) {
                let sectionNameMatches = stream.match(reg_sectionNames, true);
                if (sol && sectionNameMatches) {

                    state.section = sectionNameMatches[0].trim();
                    if (state.visitedSections.indexOf(state.section) >= 0) {
                        logError('cannot duplicate sections (you tried to duplicate \"' + state.section.toUpperCase() + '").', state.lineNumber);
                    }
                    state.line_should_end = true;
                    state.line_should_end_because = `a section name ("${state.section.toUpperCase()}")`;
                    state.visitedSections.push(state.section);
                    let sectionIndex = sectionNames.indexOf(state.section);
                    if (sectionIndex == 0) {
                        state.objects_section = 0;
                        if (state.visitedSections.length > 1) {
                            logError('section "' + state.section.toUpperCase() + '" must be the first section', state.lineNumber);
                        }
                    } else if (state.visitedSections.indexOf(sectionNames[sectionIndex - 1]) == -1) {
                        if (sectionIndex === -1) {
                            //honestly not sure how I could get here.
                            logError('no such section as "' + state.section.toUpperCase() + '".', state.lineNumber);
                        } else {
                            logError('section "' + state.section.toUpperCase() + '" is out of order, must follow  "' + sectionNames[sectionIndex - 1].toUpperCase() + '" (or it could be that the section "' + sectionNames[sectionIndex - 1].toUpperCase() + `"is just missing totally.  You have to include all section headings, even if the section itself is empty).`, state.lineNumber);
                        }
                    }

                    if (state.section === 'sounds') {
                        //populate names from rules
                        for (let n in state.objects) {
                            if (state.objects.hasOwnProperty(n)) {
                                /*                                if (state.names.indexOf(n)!==-1) {
                                                                logError('Object "'+n+'" has been declared to be multiple different things',state.objects[n].lineNumber);
                                                            }*/
                                state.names.push(n);
                            }
                        }
                        //populate names from legends
                        for (let i = 0; i < state.legend_synonyms.length; i++) {
                            let n = state.legend_synonyms[i][0];
                            /*
                            if (state.names.indexOf(n)!==-1) {
                                logError('Object "'+n+'" has been declared to be multiple different things',state.legend_synonyms[i].lineNumber);
                            }
                            */
                            state.names.push(n);
                        }
                        for (let i = 0; i < state.legend_aggregates.length; i++) {
                            let n = state.legend_aggregates[i][0];
                            /*
                            if (state.names.indexOf(n)!==-1) {
                                logError('Object "'+n+'" has been declared to be multiple different things',state.legend_aggregates[i].lineNumber);
                            }
                            */
                            state.names.push(n);
                        }
                        for (let i = 0; i < state.legend_properties.length; i++) {
                            let n = state.legend_properties[i][0];
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
                        for (let n in state.objects) {
                            if (state.objects.hasOwnProperty(n) && n.length === 1) {
                                state.abbrevNames.push(n);
                            }
                        }

                        for (let i = 0; i < state.legend_synonyms.length; i++) {
                            if (state.legend_synonyms[i][0].length === 1) {
                                state.abbrevNames.push(state.legend_synonyms[i][0]);
                            }
                        }
                        for (let i = 0; i < state.legend_aggregates.length; i++) {
                            if (state.legend_aggregates[i][0].length === 1) {
                                state.abbrevNames.push(state.legend_aggregates[i][0]);
                            }
                        }
                    }
                    return 'HEADER';
                } else {
                    if (state.section === undefined) {
                        //unreachable I think, pre-empted caught above
                        logError('must start with section "OBJECTS"', state.lineNumber);
                    }
                }
            }

            if (stream.eol()) {

                endOfLineProcessing(state, mixedCase);
                return null;
            }

            //if color is set, try to set matrix
            //if can't set matrix, try to parse name
            //if color is not set, try to parse color
            switch (state.section) {
                case 'objects':
                    {
                        let tryParseName = function () {
                            //LOOK FOR NAME
                            let match_name = sol ? stream.match(reg_name, true) : stream.match(/[^\p{Z}\s\()]+[\p{Z}\s]*/u, true);
                            if (match_name == null) {
                                stream.match(reg_notcommentstart, true);
                                if (stream.pos > 0) {
                                    logWarning('Unknown junk in object section (possibly: sprites have to be 5 pixels wide and 5 pixels high exactly. Or maybe: the main names for objects have to be words containing only the letters a-z0.9 - if you want to call them something like ",", do it in the legend section).', state.lineNumber);
                                }
                                return 'ERROR';
                            } else {
                                let candname = match_name[0].trim();
                                if (state.objects[candname] !== undefined) {
                                    logError('Object "' + candname.toUpperCase() + '" defined multiple times.', state.lineNumber);
                                    return 'ERROR';
                                }
                                for (let i = 0; i < state.legend_synonyms.length; i++) {
                                    let entry = state.legend_synonyms[i];
                                    if (entry[0] == candname) {
                                        logError('Name "' + candname.toUpperCase() + '" already in use.', state.lineNumber);
                                    }
                                }
                                if (keyword_array.indexOf(candname) >= 0) {
                                    logWarning('You named an object "' + candname.toUpperCase() + '", but this is a keyword. Don\'t do that!', state.lineNumber);
                                }

                                if (sol) {
                                    state.objects_candname = candname;
                                    registerOriginalCaseName(state, candname, mixedCase, state.lineNumber);
                                    state.objects[state.objects_candname] = {
                                        lineNumber: state.lineNumber,
                                        colors: [],
                                        spritematrix: []
                                    };

                                } else {
                                    //set up alias
                                    registerOriginalCaseName(state, candname, mixedCase, state.lineNumber);
                                    let synonym = [candname, state.objects_candname];
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
                                }
                            case 2:
                                {
                                    //LOOK FOR COLOR
                                    state.tokenIndex = 0;

                                    let match_color = stream.match(reg_color, true);
                                    if (match_color == null) {
                                        let str = stream.match(reg_name, true) || stream.match(reg_notcommentstart, true);
                                        logError('Was looking for color for object ' + state.objects_candname.toUpperCase() + ', got "' + str + '" instead.', state.lineNumber);
                                        return null;
                                    } else {
                                        if (state.objects[state.objects_candname].colors === undefined) {
                                            state.objects[state.objects_candname].colors = [match_color[0].trim()];
                                        } else {
                                            state.objects[state.objects_candname].colors.push(match_color[0].trim());
                                        }

                                        let candcol = match_color[0].trim().toLowerCase();
                                        if (candcol in colorPalettes.arnecolors) {
                                            return 'COLOR COLOR-' + candcol.toUpperCase();
                                        } else if (candcol === "transparent") {
                                            return 'COLOR FADECOLOR';
                                        } else {
                                            return 'MULTICOLOR' + match_color[0];
                                        }
                                    }
                                }
                            case 3:
                                {
                                    let ch = stream.eat(/[.\d]/);
                                    let spritematrix = state.objects_spritematrix;
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

                                    let o = state.objects[state.objects_candname];

                                    spritematrix[spritematrix.length - 1] += ch;
                                    if (spritematrix[spritematrix.length - 1].length > 5) {
                                        logWarning('Sprites must be 5 wide and 5 high.', state.lineNumber);
                                        stream.match(reg_notcommentstart, true);
                                        return null;
                                    }
                                    o.spritematrix = state.objects_spritematrix;
                                    if (spritematrix.length === 5 && spritematrix[spritematrix.length - 1].length === 5) {
                                        state.objects_section = 0;
                                    }

                                    if (ch !== '.') {
                                        let n = parseInt(ch);
                                        if (n >= o.colors.length) {
                                            logError("Trying to access color number " + n + " from the color palette of sprite " + state.objects_candname.toUpperCase() + ", but there are only " + o.colors.length + " defined in it.", state.lineNumber);
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
                        let resultToken = "";
                        let match_name = null;
                        if (state.tokenIndex === 0) {
                            match_name = stream.match(/[^=\p{Z}\s\(]*(\p{Z}\s)*/u, true);
                            let new_name = match_name[0].trim();

                            if (wordAlreadyDeclared(state, new_name)) {
                                resultToken = 'ERROR';
                            } else {
                                resultToken = 'NAME';
                            }

                            //if name already declared, we have a problem                            
                            state.tokenIndex++;
                        } else if (state.tokenIndex === 1) {
                            match_name = stream.match(/=/u, true);
                            if (match_name === null || match_name[0].trim() !== "=") {
                                logError(`In the legend, define new items using the equals symbol - declarations must look like "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".`, state.lineNumber);
                                stream.match(reg_notcommentstart, true);
                                resultToken = 'ERROR';
                                match_name = ["ERROR"];//just to reduce the chance of crashes
                            }
                            stream.match(/[\p{Z}\s]*/u, true);
                            state.tokenIndex++;
                            resultToken = 'ASSIGNMENT';
                        } else if (state.tokenIndex >= 3 && ((state.tokenIndex % 2) === 1)) {
                            //matches AND/OR
                            match_name = stream.match(reg_name, true);
                            if (match_name === null) {
                                logError("Something bad's happening in the LEGEND", state.lineNumber);
                                let match = stream.match(reg_notcommentstart, true);
                                resultToken = 'ERROR';
                            } else {
                                let candname = match_name[0].trim();
                                if (candname === "and" || candname === "or") {
                                    resultToken = 'LOGICWORD';
                                    if (state.tokenIndex >= 5) {
                                        if (candname !== state.current_line_wip_array[3]) {
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
                                let match = stream.match(reg_notcommentstart, true);
                                resultToken = 'ERROR';
                            } else {
                                let candname = match_name[0].trim();
                                if (wordAlreadyDeclared(state, candname)) {
                                    resultToken = 'NAME';
                                } else {
                                    resultToken = 'ERROR';
                                }
                                state.tokenIndex++;

                            }
                        }

                        if (match_name !== null) {
                            state.current_line_wip_array.push(match_name[0].trim());
                        }

                        if (stream.eol()) {
                            processLegendLine(state, mixedCase);
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
                        let tokentype = "";

                        if (state.current_line_wip_array.length > 0 && state.current_line_wip_array[state.current_line_wip_array.length - 1] === 'ERROR') {
                            // match=stream.match(reg_notcommentstart, true);
                            //if there was an error earlier on the line just try to do greedy matching here
                            let match = null;

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
                                let match = stream.match(reg_soundseed, true);
                                if (match !== null) {
                                    tokentype = 'SOUND';
                                }
                            }

                            //objects
                            if (match === null) {
                                match = stream.match(reg_name, true);
                                if (match !== null) {
                                    if (wordAlreadyDeclared(state, match[0].trim())) {
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


                        } else if (state.current_line_wip_array.length === 0) {
                            //can be OBJECT_NAME or SOUNDEVENT
                            let match = stream.match(reg_soundevents, true);
                            if (match == null) {
                                match = stream.match(reg_name, true);
                                if (match == null) {
                                    tokentype = 'ERROR';
                                    match = errorFallbackMatchToken(stream);
                                    state.current_line_wip_array.push("ERROR");
                                    logWarning("Was expecting a sound event (like SFX3, or ENDLEVEL) or an object name, but didn't find either.", state.lineNumber);
                                } else {
                                    let matched_name = match[0].trim();
                                    if (!wordAlreadyDeclared(state, matched_name)) {
                                        tokentype = 'ERROR';
                                        state.current_line_wip_array.push("ERROR");
                                        logError(`unexpected sound token "${matched_name}".`, state.lineNumber);
                                    } else {
                                        tokentype = 'NAME';
                                        state.current_line_wip_array.push([matched_name, tokentype]);
                                        state.tokenIndex++;
                                    }
                                }
                            } else {
                                tokentype = 'SOUNDEVENT';
                                state.current_line_wip_array.push([match[0].trim(), tokentype]);
                                state.tokenIndex++;
                            }

                        } else if (state.current_line_wip_array.length === 1) {
                            let is_soundevent = state.current_line_wip_array[0][1] === 'SOUNDEVENT';

                            if (is_soundevent) {
                                let match = stream.match(reg_soundseed, true);
                                if (match !== null) {
                                    tokentype = 'SOUND';
                                    state.current_line_wip_array.push([match[0].trim(), tokentype]);
                                    state.tokenIndex++;
                                } else {
                                    match = errorFallbackMatchToken(stream);
                                    logError("Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else.", state.lineNumber);
                                    tokentype = 'ERROR';
                                    state.current_line_wip_array.push("ERROR");
                                }
                            } else {
                                //[0] is object name
                                //it's a sound verb
                                let match = stream.match(reg_soundverbs, true);
                                if (match !== null) {
                                    tokentype = 'SOUNDVERB';
                                    state.current_line_wip_array.push([match[0].trim(), tokentype]);
                                    state.tokenIndex++;
                                } else {
                                    match = errorFallbackMatchToken(stream);
                                    logError("Was expecting a soundverb here (MOVE, DESTROY, CANTMOVE, or the like), but found something else.", state.lineNumber);
                                    tokentype = 'ERROR';
                                    state.current_line_wip_array.push("ERROR");
                                }

                            }
                        } else {
                            let is_soundevent = state.current_line_wip_array[0][1] === 'SOUNDEVENT';
                            if (is_soundevent) {
                                let match = errorFallbackMatchToken(stream);
                                logError(`I wasn't expecting anything after the sound declaration ${state.current_line_wip_array[state.current_line_wip_array.length - 1][0].toUpperCase()} on this line, so I don't know what to do with "${match[0].trim().toUpperCase()}" here.`, state.lineNumber);
                                tokentype = 'ERROR';
                                state.current_line_wip_array.push("ERROR");
                            } else {
                                //if there's a seed on the right, any additional content is superfluous
                                let is_seedonright = state.current_line_wip_array[state.current_line_wip_array.length - 1][1] === 'SOUND';
                                if (is_seedonright) {
                                    let match = errorFallbackMatchToken(stream);
                                    logError(`I wasn't expecting anything after the sound declaration ${state.current_line_wip_array[state.current_line_wip_array.length - 1][0].toUpperCase()} on this line, so I don't know what to do with "${match[0].trim().toUpperCase()}" here.`, state.lineNumber);
                                    tokentype = 'ERROR';
                                    state.current_line_wip_array.push("ERROR");
                                } else {
                                    let directional_verb = soundverbs_directional.indexOf(state.current_line_wip_array[1][0]) >= 0;
                                    if (directional_verb) {
                                        //match seed or direction                          
                                        let is_direction = stream.match(reg_sounddirectionindicators, true);
                                        if (is_direction !== null) {
                                            tokentype = 'DIRECTION';
                                            state.current_line_wip_array.push([is_direction[0].trim(), tokentype]);
                                            state.tokenIndex++;
                                        } else {
                                            let is_seed = stream.match(reg_soundseed, true);
                                            if (is_seed !== null) {
                                                tokentype = 'SOUND';
                                                state.current_line_wip_array.push([is_seed[0].trim(), tokentype]);
                                                state.tokenIndex++;
                                            } else {
                                                let match = errorFallbackMatchToken(stream);
                                                //depending on whether the verb is directional or not, we log different errors
                                                logError(`Ah I was expecting direction or a sound seed here after ${state.current_line_wip_array[state.current_line_wip_array.length - 1][0].toUpperCase()}, but I don't know what to make of "${match[0].trim().toUpperCase()}".`, state.lineNumber);
                                                tokentype = 'ERROR';
                                                state.current_line_wip_array.push("ERROR");
                                            }
                                        }
                                    } else {
                                        //only match seed
                                        let is_seed = stream.match(reg_soundseed, true);
                                        if (is_seed !== null) {
                                            tokentype = 'SOUND';
                                            state.current_line_wip_array.push([is_seed[0].trim(), tokentype]);
                                            state.tokenIndex++;
                                        } else {
                                            let match = errorFallbackMatchToken(stream);
                                            //depending on whether the verb is directional or not, we log different errors
                                            logError(`Ah I was expecting a sound seed here after ${state.current_line_wip_array[state.current_line_wip_array.length - 1][0].toUpperCase()}, but I don't know what to make of "${match[0].trim().toUpperCase()}".`, state.lineNumber);
                                            tokentype = 'ERROR';
                                            state.current_line_wip_array.push("ERROR");
                                        }
                                    }
                                }
                            }
                        }

                        if (stream.eol()) {
                            processSoundsLine(state);
                        }

                        return tokentype;
                    }
                case 'collisionlayers':
                    {
                        if (sol) {
                            //create new collision layer
                            state.collisionLayers.push([]);
                            //empty current_line_wip_array
                            state.current_line_wip_array = [];
                            state.tokenIndex = 0;
                        }

                        let match_name = stream.match(reg_name, true);
                        if (match_name === null) {
                            //then strip spaces and commas
                            let prepos = stream.pos;
                            stream.match(reg_csv_separators, true);
                            if (stream.pos == prepos) {
                                logError("error detected - unexpected character " + stream.peek(), state.lineNumber);
                                stream.next();
                            }
                            return null;
                        } else {
                            //have a name: let's see if it's valid
                            let candname = match_name[0].trim();

                            let substitutor = function (n) {
                                n = n.toLowerCase();
                                if (n in state.objects) {
                                    return [n];
                                }


                                for (let i = 0; i < state.legend_synonyms.length; i++) {
                                    let a = state.legend_synonyms[i];
                                    if (a[0] === n) {
                                        return substitutor(a[1]);
                                    }
                                }

                                for (let i = 0; i < state.legend_aggregates.length; i++) {
                                    let a = state.legend_aggregates[i];
                                    if (a[0] === n) {
                                        logError('"' + n + '" is an aggregate (defined using "and"), and cannot be added to a single layer because its constituent objects must be able to coexist.', state.lineNumber);
                                        return [];
                                    }
                                }
                                for (let i = 0; i < state.legend_properties.length; i++) {
                                    let a = state.legend_properties[i];
                                    if (a[0] === n) {
                                        let result = [];
                                        for (let j = 1; j < a.length; j++) {
                                            if (a[j] === n) {
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
                            if (candname === 'background') {
                                if (state.collisionLayers.length > 0 && state.collisionLayers[state.collisionLayers.length - 1].length > 0) {
                                    logError("Background must be in a layer by itself.", state.lineNumber);
                                }
                                state.tokenIndex = 1;
                            } else if (state.tokenIndex !== 0) {
                                logError("Background must be in a layer by itself.", state.lineNumber);
                            }

                            let ar = substitutor(candname);

                            if (state.collisionLayers.length === 0) {
                                //pre-empted by other messages
                                logError("no layers found.", state.lineNumber);
                                return 'ERROR';
                            }

                            let foundOthers = [];
                            let foundSelves = [];
                            for (let i = 0; i < ar.length; i++) {
                                let tcandname = ar[i];
                                for (let j = 0; j <= state.collisionLayers.length - 1; j++) {
                                    let clj = state.collisionLayers[j];
                                    if (clj.indexOf(tcandname) >= 0) {
                                        if (j !== state.collisionLayers.length - 1) {
                                            foundOthers.push(j);
                                        } else {
                                            foundSelves.push(j);
                                        }
                                    }
                                }
                            }
                            if (foundOthers.length > 0) {
                                let warningStr = 'Object "' + candname.toUpperCase() + '" included in multiple collision layers ( layers ';
                                for (let i = 0; i < foundOthers.length; i++) {
                                    warningStr += "#" + (foundOthers[i] + 1) + ", ";
                                }
                                warningStr += "#" + state.collisionLayers.length;
                                logWarning(warningStr + ' ). You should fix this!', state.lineNumber);
                            }

                            if (state.current_line_wip_array.indexOf(candname) >= 0) {
                                let warningStr = 'Object "' + candname.toUpperCase() + '" included explicitly multiple times in the same layer. Don\'t do that innit.';
                                logWarning(warningStr, state.lineNumber);
                            }
                            state.current_line_wip_array.push(candname);

                            state.collisionLayers[state.collisionLayers.length - 1] = state.collisionLayers[state.collisionLayers.length - 1].concat(ar);
                            if (ar.length > 0) {
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
                            let rule = reg_notcommentstart.exec(stream.string)[0];
                            state.rules.push([rule, state.lineNumber, mixedCase]);
                            state.tokenIndex = 0;//in rules, records whether bracket has been found or not
                        }

                        if (state.tokenIndex === -4) {
                            stream.skipToEnd();
                            return 'MESSAGE';
                        }
                        if (stream.match(/[\p{Z}\s]*->[\p{Z}\s]*/u, true)) {
                            return 'ARROW';
                        }
                        if (ch === '[' || ch === '|' || ch === ']' || ch === '+') {
                            if (ch !== '+') {
                                state.tokenIndex = 1;
                            }
                            stream.next();
                            stream.match(/[\p{Z}\s]*/u, true);
                            return 'BRACKET';
                        } else {
                            let m = stream.match(/[^\[\|\]\p{Z}\s]*/u, true)[0].trim();

                            if (state.tokenIndex === 0 && reg_loopmarker.exec(m)) {
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
                                } else if (m === '...') {
                                    return 'DIRECTION';
                                } else if (m === 'rigid') {
                                    return 'DIRECTION';
                                } else if (m === 'random') {
                                    return 'DIRECTION';
                                } else if (commandwords.indexOf(m) >= 0) {
                                    if (m === 'message') {
                                        state.tokenIndex = -4;
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
                            let tokenized = reg_notcommentstart.exec(stream.string);
                            let splitted = tokenized[0].split(/[\p{Z}\s]/u);
                            let filtered = splitted.filter(function (v) { return v !== '' });
                            filtered.push(state.lineNumber);

                            state.winconditions.push(filtered);
                            state.tokenIndex = -1;
                        }
                        state.tokenIndex++;

                        let match = stream.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);
                        if (match === null) {
                            logError('incorrect format of win condition.', state.lineNumber);
                            stream.match(reg_notcommentstart, true);
                            return 'ERROR';

                        } else {
                            let candword = match[0].trim();
                            if (state.tokenIndex === 0) {
                                if (reg_winconditionquantifiers.exec(candword)) {
                                    return 'LOGICWORD';
                                }
                                else {
                                    logError('Expecting the start of a win condition ("ALL","SOME","NO") but got "' + candword.toUpperCase() + "'.", state.lineNumber);
                                    return 'ERROR';
                                }
                            }
                            else if (state.tokenIndex === 2) {
                                if (candword != 'on') {
                                    logError('Expecting the word "ON" but got "' + candword.toUpperCase() + "\".", state.lineNumber);
                                    return 'ERROR';
                                } else {
                                    return 'LOGICWORD';
                                }
                            }
                            else if (state.tokenIndex === 1 || state.tokenIndex === 3) {
                                if (state.names.indexOf(candword) === -1) {
                                    logError('Error in win condition: "' + candword.toUpperCase() + '" is not a valid object name.', state.lineNumber);
                                    return 'ERROR';
                                } else {
                                    return 'NAME';
                                }
                            } else {
                                logError("Error in win condition: I don't know what to do with " + candword.toUpperCase() + ".", state.lineNumber);
                                return 'ERROR';
                            }
                        }
                        break;
                    }
                case 'levels':
                    {
                        if (sol) {
                            if (stream.match(/[\p{Z}\s]*message\b[\p{Z}\s]*/u, true)) {
                                state.tokenIndex = -4;//-4/2 = message/level
                                let newdat = ['\n', mixedCase.slice(stream.pos).trim(), state.lineNumber];
                                if (state.levels[state.levels.length - 1].length === 0) {
                                    state.levels.splice(state.levels.length - 1, 0, newdat);
                                } else {
                                    state.levels.push(newdat);
                                }
                                return 'MESSAGE_VERB';//a duplicate of the previous section as a legacy thing for #589 
                            } else if (stream.match(/[\p{Z}\s]*message[\p{Z}\s]*/u, true)) {//duplicating previous section because of #589
                                logWarning("You probably meant to put a space after 'message' innit.  That's ok, I'll still interpret it as a message, but you probably want to put a space there.", state.lineNumber);
                                state.tokenIndex = -4;//-4/2 = message/level
                                let newdat = ['\n', mixedCase.slice(stream.pos).trim(), state.lineNumber];
                                if (state.levels[state.levels.length - 1].length === 0) {
                                    state.levels.splice(state.levels.length - 1, 0, newdat);
                                } else {
                                    //don't seem to ever reach this.
                                    state.levels.push(newdat);
                                }
                                return 'MESSAGE_VERB';
                            } else {
                                let matches = stream.match(reg_notcommentstart, false);
                                if (matches === null || matches.length === 0) {
                                    //not sure if it's possible to get here.
                                    logError("Detected a comment where I was expecting a level. Oh gosh; if this is to do with you using '(' as a character in the legend, please don't do that ^^", state.lineNumber);
                                    state.commentLevel++;
                                    stream.skipToEnd();
                                    return 'comment';
                                } else {
                                    let line = matches[0].trim();
                                    state.tokenIndex = 2;
                                    let lastlevel = state.levels[state.levels.length - 1];
                                    if (lastlevel[0] == '\n') {
                                        state.levels.push([state.lineNumber, line]);
                                    } else {
                                        if (lastlevel.length == 0) {
                                            lastlevel.push(state.lineNumber);
                                        }
                                        lastlevel.push(line);

                                        if (lastlevel.length > 1) {
                                            if (line.length != lastlevel[1].length) {
                                                logWarning("Maps must be rectangular, yo (In a level, the length of each row must be the same).", state.lineNumber);
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
                            let ch = stream.peek();
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
                        if (sol || state.sol_after_comment) {
                            state.tokenIndex = 0;
                        }
                        if (state.tokenIndex == 0) {
                            let match = stream.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);
                            if (match !== null) {
                                let token = match[0].trim();
                                if (sol) {
                                    if (['title', 'author', 'homepage', 'background_color', 'text_color', 'key_repeat_interval', 'realtime_interval', 'again_interval', 'flickscreen', 'zoomscreen', 'color_palette', 'youtube'].indexOf(token) >= 0) {

                                        if (token === 'author' || token === 'homepage' || token === 'title') {
                                            stream.string = mixedCase;
                                        }

                                        if (token === "youtube") {
                                            logWarning("Unfortunately, YouTube support hasn't been working properly for a long time - it was always a hack and it hasn't gotten less hacky over time, so I can no longer pretend to support it.", state.lineNumber);
                                        }

                                        let m2 = stream.match(reg_notcommentstart, false);

                                        if (m2 !== null) {
                                            state.metadata.push(token);
                                            state.metadata.push(m2[0].trim());
                                            if (token in state.metadata_lines) {
                                                let otherline = state.metadata_lines[token];
                                                logWarning(`You've already defined a ${token.toUpperCase()} in the prelude on line <a onclick="jumpToLine(${otherline})>${otherline}</a>.`, state.lineNumber);
                                            }
                                            state.metadata_lines[token] = state.lineNumber;
                                        } else {
                                            logError('MetaData "' + token + '" needs a value.', state.lineNumber);
                                        }
                                        state.tokenIndex = 1;
                                        return 'METADATA';
                                    } else if (['run_rules_on_level_start', 'norepeat_action', 'require_player_movement', 'debug', 'verbose_logging', 'throttle_movement', 'noundo', 'noaction', 'norestart', 'scanline'].indexOf(token) >= 0) {
                                        state.metadata.push(token);
                                        state.metadata.push("true");
                                        state.tokenIndex = -1;


                                        let m2 = stream.match(reg_notcommentstart, false);

                                        if (m2 !== null) {
                                            let extra = m2[0].trim();
                                            logWarning('MetaData ' + token.toUpperCase() + ' doesn\'t take any parameters, but you went and gave it "' + extra + '".', state.lineNumber);
                                        }

                                        return 'METADATA';
                                    } else {
                                        logError('Unrecognised stuff in the prelude.', state.lineNumber);
                                        return 'ERROR';
                                    }
                                } else if (state.tokenIndex == -1) {
                                    //no idea how to get here. covered with a similar error message above.
                                    logError('MetaData "' + token + '" has no parameters.', state.lineNumber);
                                    return 'ERROR';
                                }
                                return 'METADATA';
                            } else {
                                //garbage found
                                logError(`Unrecognised stuff "${stream.string}" in the prelude.`, state.lineNumber);
                            }
                        } else {
                            stream.match(reg_notcommentstart, true);
                            state.tokenIndex++;

                            let key = state.metadata[state.metadata.length - 2];
                            let val = state.metadata[state.metadata.length - 1];

                            if (state.tokenIndex > 2) {
                                logWarning("Error: you can't embed comments in metadata values. Anything after the comment will be ignored.", state.lineNumber);
                                return 'ERROR';
                            }
                            if (key === "background_color" || key === "text_color") {
                                let candcol = val.trim().toLowerCase();
                                if (candcol in colorPalettes.arnecolors) {
                                    return 'COLOR COLOR-' + candcol.toUpperCase();
                                } else if (candcol === "transparent") {
                                    return 'COLOR FADECOLOR';
                                } else if ((candcol.length === 4) || (candcol.length === 7)) {
                                    let color = candcol.match(/#[0-9a-fA-F]+/);
                                    if (color !== null) {
                                        return 'MULTICOLOR' + color[0];
                                    }
                                }

                            }
                            return "METADATATEXT";
                        }
                        break;
                    }
            }


            if (stream.eol()) {
                //don't know how to reach this.
                return null;
            }

            if (!stream.eol()) {
                stream.next();
                return null;
            }
        },
        startState: function () {
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
