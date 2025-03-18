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

function colorToHex(palette, str) {
    str = str.trim();
    if (str in palette) {
        return palette[str];
    }

    return str;
}


function generateSpriteMatrix(dat) {

    let result = [];
    for (let i = 0; i < dat.length; i++) {
        let row = [];
        for (let j = 0; j < dat.length; j++) {
            let ch = dat[i].charAt(j);
            if (ch === '.') {
                row.push(-1);
            } else {
                row.push(ch);
            }
        }
        result.push(row);
    }
    return result;
}

let debugMode;
let colorPalette;

function generateExtraMembers(state) {

    //annotate objects with layers
    //assign ids at the same time
    state.idDict = [];
    let idcount = 0;
    for (let layerIndex = 0; layerIndex < state.collisionLayers.length; layerIndex++) {
        for (let j = 0; j < state.collisionLayers[layerIndex].length; j++) {
            let n = state.collisionLayers[layerIndex][j];
            if (n in state.objects) {
                let o = state.objects[n];
                o.layer = layerIndex;
                o.id = idcount;
                state.idDict[idcount] = n;
                idcount++;
            }
        }
    }

    //set object count
    state.objectCount = idcount;

    //calculate blank mask template
    let layerCount = state.collisionLayers.length;
    let blankMask = [];
    for (let i = 0; i < layerCount; i++) {
        blankMask.push(-1);
    }

    // how many words do our bitvecs need to hold?
    STRIDE_OBJ = Math.ceil(state.objectCount / 32) | 0;
    STRIDE_MOV = Math.ceil(layerCount / 5) | 0;
    LAYER_COUNT = layerCount;
    state.STRIDE_OBJ = STRIDE_OBJ;
    state.STRIDE_MOV = STRIDE_MOV;
    state.LAYER_COUNT = LAYER_COUNT;
    RebuildGameArrays();
    
    //get colorpalette name
    debugMode = false;
    verbose_logging = false;
    throttle_movement = false;
    colorPalette = colorPalettes.arnecolors;
    for (let i = 0; i < state.metadata.length; i += 2) {
        let key = state.metadata[i];
        let val = state.metadata[i + 1];
        if (key === 'color_palette') {
            if (val in colorPalettesAliases) {
                val = colorPalettesAliases[val];
            }
            if (colorPalettes[val] === undefined) {
                logError('Palette "' + val + '" not found, defaulting to arnecolors.', 0);
            } else {
                colorPalette = colorPalettes[val];
            }
        } else if (key === 'debug') {
            if (IDE && unitTesting === false) {
                debugMode = true;
                cache_console_messages = true;
            }
        } else if (key === 'verbose_logging') {
            if (IDE && unitTesting === false) {
                verbose_logging = true;
                cache_console_messages = true;
            }
        } else if (key === 'throttle_movement') {
            throttle_movement = true;
        }
    }

    //convert colors to hex
    for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            //convert color to hex
            let o = state.objects[n];
            if (o.colors.length > 10) {
                logError("a sprite cannot have more than 10 colors.  Why you would want more than 10 is beyond me.", o.lineNumber + 1);
            }
            for (let i = 0; i < o.colors.length; i++) {
                let c = o.colors[i];
                if (isColor(c)) {
                    c = colorToHex(colorPalette, c);
                    o.colors[i] = c;
                } else {
                    logError('Invalid color specified for object "' + n + '", namely "' + o.colors[i] + '".', o.lineNumber + 1);
                    o.colors[i] = '#ff00ff'; // magenta error color
                }
            }
        }
    }

    //generate sprite matrix
    for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            let o = state.objects[n];
            if (o.colors.length === 0) {
                logError('color not specified for object "' + n + '".', o.lineNumber);
                o.colors = ["#ff00ff"];
            }
            if (o.spritematrix.length === 0) {
                o.spritematrix = [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ];
            } else {
                if (o.spritematrix.length !== 5 || o.spritematrix[0].length !== 5 || o.spritematrix[1].length !== 5 || o.spritematrix[2].length !== 5 || o.spritematrix[3].length !== 5 || o.spritematrix[4].length !== 5) {
                    logWarning("Sprite graphics must be 5 wide and 5 high exactly.", o.lineNumber);
                }
                o.spritematrix = generateSpriteMatrix(o.spritematrix);
            }
        }
    }

    let glyphOrder = [];
    //calculate glyph dictionary
    let glyphDict = {};
    for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            let o = state.objects[n];
            let mask = blankMask.concat([]);
            mask[o.layer] = o.id;
            glyphDict[n] = mask;
            glyphOrder.push([o.lineNumber, n]);
        }
    }

    let added = true;
    while (added) {
        added = false;

        //then, synonyms
        for (let i = 0; i < state.legend_synonyms.length; i++) {
            let dat = state.legend_synonyms[i];
            let key = dat[0];
            let val = dat[1];
            if ((!(key in glyphDict) || (glyphDict[key] === undefined)) && (glyphDict[val] !== undefined)) {
                added = true;
                glyphDict[key] = glyphDict[val];
                glyphOrder.push([dat.lineNumber, key]);
            }
        }

        //then, aggregates
        for (let i = 0; i < state.legend_aggregates.length; i++) {
            let dat = state.legend_aggregates[i];
            let key = dat[0];
            let vals = dat.slice(1);
            let allVallsFound = true;
            for (let j = 0; j < vals.length; j++) {
                let v = vals[j];
                if (glyphDict[v] === undefined) {
                    allVallsFound = false;
                    break;
                }
            }
            if ((!(key in glyphDict) || (glyphDict[key] === undefined)) && allVallsFound) {
                let mask = blankMask.concat([]);

                for (let j = 1; j < dat.length; j++) {
                    let n = dat[j];
                    let o = state.objects[n];
                    if (o === undefined) {
                        logError('Object not found with name ' + n, state.lineNumber);
                    }
                    if (mask[o.layer] === -1) {
                        mask[o.layer] = o.id;
                    } else {
                        if (o.layer === undefined) {
                            logError('Object "' + n.toUpperCase() + '" has been defined, but not assigned to a layer.', dat.lineNumber);
                        } else {
                            let n1 = n.toUpperCase();
                            let n2 = state.idDict[mask[o.layer]].toUpperCase();
                            // if (n1 !== n2) {
                            logError(
                                'Trying to create an aggregate object (something defined in the LEGEND section using AND) with both "' +
                                n1 + '" and "' + n2 + '", which are on the same layer and therefore can\'t coexist.',
                                dat.lineNumber
                            );
                            // }
                        }
                    }
                }
                added = true;
                glyphDict[dat[0]] = mask;
                glyphOrder.push([dat.lineNumber, key]);
            }
        }
    }

    //sort glyphs line number
    glyphOrder.sort((a, b) => a[0] - b[0]);
    glyphOrder = glyphOrder.map(a => a[1]);

    state.glyphDict = glyphDict;
    state.glyphOrder = glyphOrder;

    let aggregatesDict = {};
    for (let i = 0; i < state.legend_aggregates.length; i++) {
        let entry = state.legend_aggregates[i];
        aggregatesDict[entry[0]] = entry.slice(1);
    }
    state.aggregatesDict = aggregatesDict;

    let propertiesDict = {};
    for (let i = 0; i < state.legend_properties.length; i++) {
        let entry = state.legend_properties[i];
        propertiesDict[entry[0]] = entry.slice(1);
    }
    state.propertiesDict = propertiesDict;

    //calculate lookup dictionaries
    let synonymsDict = {};
    for (let i = 0; i < state.legend_synonyms.length; i++) {
        let entry = state.legend_synonyms[i];
        let key = entry[0];
        let value = entry[1];
        if (value in aggregatesDict) {
            aggregatesDict[key] = aggregatesDict[value];
        } else if (value in propertiesDict) {
            propertiesDict[key] = propertiesDict[value];
        } else if (key !== value) {
            synonymsDict[key] = value;
        }
    }
    state.synonymsDict = synonymsDict;

    let modified = true;
    while (modified) {
        modified = false;
        for (let n in synonymsDict) {
            if (synonymsDict.hasOwnProperty(n)) {
                let value = synonymsDict[n];
                if (value in propertiesDict) {
                    delete synonymsDict[n];
                    propertiesDict[n] = propertiesDict[value];
                    modified = true;
                } else if (value in aggregatesDict) {
                    delete aggregatesDict[n];
                    aggregatesDict[n] = aggregatesDict[value];
                    modified = true;
                } else if (value in synonymsDict) {
                    synonymsDict[n] = synonymsDict[value];
                }
            }
        }

        for (let n in propertiesDict) {
            if (propertiesDict.hasOwnProperty(n)) {
                let values = propertiesDict[n];
                for (let i = 0; i < values.length; i++) {
                    let value = values[i];
                    if (value in synonymsDict) {
                        values[i] = synonymsDict[value];
                        modified = true;
                    } else if (value in propertiesDict) {
                        values.splice(i, 1);
                        let newvalues = propertiesDict[value];
                        for (let j = 0; j < newvalues.length; j++) {
                            let newvalue = newvalues[j];
                            if (values.indexOf(newvalue) === -1) {
                                values.push(newvalue);
                            }
                        }
                        modified = true;
                    }
                    if (value in aggregatesDict) {
                        logError('Trying to define property "' + n.toUpperCase() + '" in terms of aggregate "' + value.toUpperCase() + '".');
                    }
                }
            }
        }


        for (let n in aggregatesDict) {
            if (aggregatesDict.hasOwnProperty(n)) {
                let values = aggregatesDict[n];
                for (let i = 0; i < values.length; i++) {
                    let value = values[i];
                    if (value in synonymsDict) {
                        values[i] = synonymsDict[value];
                        modified = true;
                    } else if (value in aggregatesDict) {
                        values.splice(i, 1);
                        let newvalues = aggregatesDict[value];
                        for (let j = 0; j < newvalues.length; j++) {
                            let newvalue = newvalues[j];
                            if (values.indexOf(newvalue) === -1) {
                                values.push(newvalue);
                            }
                        }
                        modified = true;
                    }
                    if (value in propertiesDict) {
                        logError('Trying to define aggregate "' + n.toUpperCase() + '" in terms of property "' + value.toUpperCase() + '".');
                    }
                }
            }
        }
    }

    /* determine which properties specify objects all on one layer */
    state.propertiesSingleLayer = {};
    for (let key in propertiesDict) {
        if (propertiesDict.hasOwnProperty(key)) {
            let values = propertiesDict[key];
            let sameLayer = true;
            for (let i = 1; i < values.length; i++) {
                if ((state.objects[values[i - 1]].layer !== state.objects[values[i]].layer)) {
                    sameLayer = false;
                    break;
                }
            }
            if (sameLayer) {
                state.propertiesSingleLayer[key] = state.objects[values[0]].layer;
            }
        }
    }

    if (state.idDict[0] === undefined && state.collisionLayers.length > 0) {
        logError('You need to have some objects defined');
    }

    //set default background object
    let backgroundid;
    let backgroundlayer;
    if (state.objects.background === undefined) {
        if ('background' in state.synonymsDict) {
            let n = state.synonymsDict['background'];
            let o = state.objects[n];
            backgroundid = o.id;
            backgroundlayer = o.layer;
        } else if ('background' in state.propertiesDict) {
            let backgrounddef = state.propertiesDict['background'];
            let n = backgrounddef[0];
            let o = state.objects[n];
            backgroundid = o.id;
            backgroundlayer = o.layer;
            for (let i = 1; i < backgrounddef.length; i++) {
                let nnew = backgrounddef[i];
                let onew = state.objects[nnew];
                if (onew.layer !== backgroundlayer) {
                    let lineNumber = state.original_line_numbers['background'];
                    logError('Background objects must be on the same layer', lineNumber);
                }
            }
        } else if ('background' in state.aggregatesDict) {
            let o = state.objects[state.idDict[0]];
            backgroundid = o.id;
            backgroundlayer = o.layer;
            let lineNumber = state.original_line_numbers['background'];
            logError("background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or').", lineNumber);
        } else {
            //background doesn't exist. Error already printed elsewhere.
            let o = state.objects[state.idDict[0]];
            if (o != null) {
                backgroundid = o.id;
                backgroundlayer = o.layer;
            }
            logError("Seriously, you have to define something to be the background.");
        }
    } else {
        backgroundid = state.objects.background.id;
        backgroundlayer = state.objects.background.layer;
    }
    state.backgroundid = backgroundid;
    state.backgroundlayer = backgroundlayer;
}

function levelFromString(state, level) {
    let backgroundlayer = state.backgroundlayer;
    let backgroundid = state.backgroundid;
    let backgroundLayerMask = state.layerMasks[backgroundlayer];
    let o = new Level(level[0], level[1].length, level.length - 1, state.collisionLayers.length, null);
    o.objects = new Int32Array(o.width * o.height * STRIDE_OBJ);

    for (let i = 0; i < o.width; i++) {
        for (let j = 0; j < o.height; j++) {
            let ch = level[j + 1].charAt(i);
            if (ch.length === 0) {
                ch = level[j + 1].charAt(level[j + 1].length - 1);
            }
            let mask = state.glyphDict[ch];

            if (mask === undefined) {
                if (state.propertiesDict[ch] === undefined) {
                    logError('Error, symbol "' + ch + '", used in map, not found.', level[0] + j);
                } else {
                    logError('Error, symbol "' + ch + '" is defined using OR, and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of AND?', level[0] + j);
                }
                return o;
            }

            let maskint = new BitVec(STRIDE_OBJ);
            mask = mask.concat([]);
            for (let z = 0; z < o.layerCount; z++) {
                if (mask[z] >= 0) {
                    maskint.ibitset(mask[z]);
                }
            }
            for (let w = 0; w < STRIDE_OBJ; ++w) {
                o.objects[STRIDE_OBJ * (i * o.height + j) + w] = maskint.data[w];
            }
        }
    }

    let levelBackgroundMask = o.calcBackgroundMask(state);
    for (let i = 0; i < o.n_tiles; i++) {
        let cell = o.getCell(i);
        if (!backgroundLayerMask.anyBitsInCommon(cell)) {
            cell.ior(levelBackgroundMask);
            o.setCell(i, cell);
        }
    }
    return o;
}
//also assigns glyphDict
function levelsToArray(state) {
    let levels = state.levels;
    let processedLevels = [];

    for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
        let level = levels[levelIndex];
        if (level.length === 0) {
            continue;
        }
        if (level[0] === '\n') {

            let o = {
                message: level[1]
            };
            splitMessage = wordwrap(o.message, intro_template[0].length);
            if (splitMessage.length > 12) {
                logWarning('Message too long to fit on screen.', level[2]);
            }

            processedLevels.push(o);
        } else {
            let o = levelFromString(state, level);
            processedLevels.push(o);
        }

    }

    state.levels = processedLevels;
}

let directionaggregates = {
    'horizontal': ['left', 'right'],
    'horizontal_par': ['left', 'right'],
    'horizontal_perp': ['left', 'right'],
    'vertical': ['up', 'down'],
    'vertical_par': ['up', 'down'],
    'vertical_perp': ['up', 'down'],
    'moving': ['up', 'down', 'left', 'right', 'action'],
    'orthogonal': ['up', 'down', 'left', 'right'],
    'perpendicular': ['^', 'v'],
    'parallel': ['<', '>']
};

let relativeDirections = ['^', 'v', '<', '>', 'perpendicular', 'parallel'];
let simpleAbsoluteDirections = ['up', 'down', 'left', 'right'];
let simpleRelativeDirections = ['^', 'v', '<', '>'];
let reg_directions_only = /^(\>|\<|\^|v|up|down|left|right|moving|stationary|no|randomdir|random|horizontal|vertical|orthogonal|perpendicular|parallel|action)$/;
//redeclaring here, i don't know why
let commandwords = ["sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10", "cancel", "checkpoint", "restart", "win", "message", "again"];
let commandwords_sfx = ["sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10"];


function directionalRule(rule) {
    for (let i = 0; i < rule.lhs.length; i++) {
        let cellRow = rule.lhs[i];
        if (cellRow.length > 1) {
            return true;
        }
        for (let j = 0; j < cellRow.length; j++) {
            let cell = cellRow[j];
            for (let k = 0; k < cell.length; k += 2) {
                if (relativeDirections.indexOf(cell[k]) >= 0) {
                    return true;
                }
            }
        }
    }
    for (let i = 0; i < rule.rhs.length; i++) {
        let cellRow = rule.rhs[i];
        for (let j = 0; j < cellRow.length; j++) {
            let cell = cellRow[j];
            for (let k = 0; k < cell.length; k += 2) {
                if (relativeDirections.indexOf(cell[k]) >= 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

function findIndexAfterToken(str, tokens, tokenIndex) {
    str = str.toLowerCase();
    let curIndex = 0;
    for (let i = 0; i <= tokenIndex; i++) {
        let token = tokens[i];
        curIndex = str.indexOf(token, curIndex) + token.length;
    }
    return curIndex;
}
function rightBracketToRightOf(tokens, i) {
    for (; i < tokens.length; i++) {
        if (tokens[i] === "]") {
            return true;
        }
    }
    return false;
}

function processRuleString(rule, state, curRules) {
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
    let line = rule[0];
    let lineNumber = rule[1];
    let origLine = rule[2];

    // STEP ONE, TOKENIZE
    line = line.replace(/\[/g, ' [ ').replace(/\]/g, ' ] ').replace(/\|/g, ' | ').replace(/\-\>/g, ' -> ');
    line = line.trim();
    if (line[0] === '+') {
        line = line.substring(0, 1) + " " + line.substring(1, line.length);
    }
    let tokens = line.split(/\s/).filter(function (v) { return v !== '' });

    if (tokens.length === 0) {
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
    let parsestate = 0;
    let directions = [];

    let curcell = null; // [up, cat, down mouse]
    let curcellrow = []; // [  [up, cat]  [ down, mouse ] ]

    let incellrow = false;

    let appendGroup = false;
    let rhs = false;
    let lhs_cells = [];
    let rhs_cells = [];
    let late = false;
    let rigid = false;
    let groupNumber = lineNumber;
    let commands = [];
    let randomRule = false;
    let has_plus = false;

    if (tokens.length === 1) {
        if (tokens[0] === "startloop") {
            let rule_line = {
                bracket: 1
            }
            return rule_line;
        } else if (tokens[0] === "endloop") {
            let rule_line = {
                bracket: -1
            }
            return rule_line;
        }
    }

    if (tokens.indexOf('->') === -1) {
        logError("A rule has to have an arrow in it.  There's no arrow here! Consider reading up about rules - you're clearly doing something weird", lineNumber);
    }

    curcell = [];
    let bracketbalance = 0;
    for (let i = 0; i < tokens.length; i++) {
        let token = tokens[i];
        switch (parsestate) {
            case 0:
                {
                    //read initial directions
                    if (token === '+') {
                        has_plus = true;
                        if (groupNumber === lineNumber) {
                            if (curRules.length === 0) {
                                logError('The "+" symbol, for joining a rule with the group of the previous rule, needs a previous rule to be applied to.', lineNumber);
                                has_plus = false;
                            }
                            if (i !== 0) {
                                logError('The "+" symbol, for joining a rule with the group of the previous rule, must be the first symbol on the line ', lineNumber);
                            }
                            if (has_plus) {
                                groupNumber = curRules[curRules.length - 1].groupNumber;
                            }
                        } else {
                            logError('Two "+"s (the "append to previous rule group" symbol) applied to the same rule.', lineNumber);
                        }
                    } else if (token in directionaggregates) {
                        directions = directions.concat(directionaggregates[token]);
                    } else if (token === 'late') {
                        late = true;
                    } else if (token === 'rigid') {
                        rigid = true;
                    } else if (token === 'random') {
                        randomRule = true;
                        if (has_plus) {
                            logError(`A rule-group can only be marked random by the opening rule in the group (aka, a '+' and 'random' can't appear as rule modifiers on the same line).  Why? Well, you see "random" isn't a property of individual rules, but of whole rule groups.  It indicates that a single possible application of some rule from the whole group should be applied at random.`, lineNumber)
                        }

                    } else if (simpleAbsoluteDirections.indexOf(token) >= 0) {
                        directions.push(token);
                    } else if (simpleRelativeDirections.indexOf(token) >= 0) {
                        logError('You cannot use relative directions (\"^v<>\") to indicate in which direction(s) a rule applies.  Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions', lineNumber);
                    } else if (token === '[') {
                        if (directions.length === 0) {
                            directions = directions.concat(directionaggregates['orthogonal']);
                        }
                        parsestate = 1;
                        i--;
                    } else {
                        logError("The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions).  It seems you've just entered \"" + token.toUpperCase() + '\".', lineNumber);
                    }
                    break;
                }
            case 1:
                {
                    if (token === '[') {
                        bracketbalance++;
                        if (bracketbalance > 1) {
                            logWarning("Multiple opening brackets without closing brackets.  Something fishy here.  Every '[' has to be closed by a ']', and you can't nest them.", lineNumber);
                        }
                        if (curcell.length > 0) {
                            logError('Error, malformed cell rule - encountered a "["" before previous bracket was closed', lineNumber);
                        }
                        incellrow = true;
                        curcell = [];
                    } else if (reg_directions_only.exec(token)) {
                        if (curcell.length % 2 === 1) {
                            logError("Error, an item can only have one direction/action at a time, but you're looking for several at once!", lineNumber);
                        } else if (!incellrow) {
                            logWarning("Invalid syntax. Directions should be placed at the start of a rule.", lineNumber);
                        } else if (late && token !== 'no' && token !== 'random' && token !== 'randomdir') {
                            logError("Movements cannot appear in late rules.", lineNumber);
                        } else {
                            curcell.push(token);
                        }
                    } else if (token === '|') {
                        if (!incellrow) {
                            logWarning('Janky syntax.  "|" should only be used inside cell rows (the square brackety bits).', lineNumber);
                        } else if (curcell.length % 2 === 1) {
                            logError('In a rule, if you specify a movement, it has to act on an object.', lineNumber);
                        } else {
                            curcellrow.push(curcell);
                            curcell = [];
                        }
                    } else if (token === ']') {

                        bracketbalance--;
                        if (bracketbalance < 0) {
                            logWarning("Multiple closing brackets without corresponding opening brackets.  Something fishy here.  Every '[' has to be closed by a ']', and you can't nest them.", lineNumber);
                            return null;
                        }

                        if (curcell.length % 2 === 1) {
                            if (curcell[0] === '...') {
                                logError('Cannot end a rule with ellipses.', lineNumber);
                            } else {
                                logError('In a rule, if you specify a movement, it has to act on an object.', lineNumber);
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
                        incellrow = false;
                    } else if (token === '->') {

                        if (groupNumber !== lineNumber) {
                            let parentrule = curRules[curRules.length - 1];
                            if (parentrule.late !== late) {
                                logWarning('Oh gosh you can mix late and non-late rules in a rule-group if you really want to, but gosh why would you want to do that?  What do you expect to accomplish?', lineNumber);
                            }
                        }

                        if (incellrow) {
                            logWarning('Encountered an unexpected "->" inside square brackets.  It\'s used to separate states, it has no place inside them >:| .', lineNumber);
                        } else if (rhs) {
                            logError('Error, you can only use "->" once in a rule; it\'s used to separate before and after states.', lineNumber);
                            return null;
                        } else {
                            rhs = true;
                        }
                    } else if (state.names.indexOf(token) >= 0) {
                        if (!incellrow) {
                            logWarning("Invalid token " + token.toUpperCase() + ". Object names should only be used within cells (square brackets).", lineNumber);
                        } else if (curcell.length % 2 === 0) {
                            curcell.push('');
                            curcell.push(token);
                        } else if (curcell.length % 2 === 1) {
                            curcell.push(token);
                        }
                    } else if (token === '...') {
                        if (!incellrow) {
                            logWarning("Invalid syntax, ellipses should only be used within cells (square brackets).", lineNumber);
                        } else {
                            curcell.push(token);
                            curcell.push(token);
                        }
                    } else if (commandwords.indexOf(token) >= 0) {
                        if (rhs === false) {
                            logError("Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", lineNumber);
                        } else if (incellrow || rightBracketToRightOf(tokens, i)) {//only a warning for legacy support reasons.
                            logWarning("Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", lineNumber);
                        }
                        if (token === 'message') {
                            let messageIndex = findIndexAfterToken(origLine, tokens, i);
                            let messageStr = origLine.substring(messageIndex).trim();
                            if (messageStr === "") {
                                messageStr = " ";
                                //needs to be nonempty or the system gets confused and thinks it's a whole level message rather than an interstitial.
                            }
                            commands.push([token, messageStr]);
                            i = tokens.length;
                        } else {
                            if (commandwords_sfx.indexOf(token) >= 0) {
                                //check defined
                                let found = false;
                                for (let j = 0; j < state.sounds.length; j++) {
                                    let sound = state.sounds[j];
                                    if (sound[0][0] === token) {
                                        found = true;
                                    }
                                }
                                if (!found) {
                                    logWarning('Sound effect "' + token + '" not defined.', lineNumber);
                                }
                            }
                            commands.push([token]);
                        }
                    } else {
                        logError('Error, malformed cell rule - was looking for cell contents, but found "' + token + '".  What am I supposed to do with this, eh, please tell me that.', lineNumber);
                    }
                }

        }
    }

    if (late && rigid) {
        logError("Late rules cannot be marked as rigid (rigid rules are all about dealing with the consequences of unresolvable movements, and late rules can't even have movements).", lineNumber);
    }

    if (lhs_cells.length != rhs_cells.length) {
        if (commands.length > 0 && rhs_cells.length === 0) {
            //ok
        } else {
            logWarning('Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right', lineNumber);
        }
    } else {
        for (let i = 0; i < lhs_cells.length; i++) {
            if (lhs_cells[i].length != rhs_cells[i].length) {
                logError('In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells).', lineNumber);
                state.invalid = true;
            }
            if (lhs_cells[i].length === 0) {
                logError("You have an totally empty pattern on the left-hand side.  This will match *everything*.  You certainly don't want this.");
            }
        }
    }

    if (lhs_cells.length === 0) {
        logError('This rule refers to nothing.  What the heck? :O', lineNumber);
    }

    let rule_line = {
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

    if (directionalRule(rule_line) === false && rule_line.directions.length > 1) {
        rule_line.directions.splice(1);
    }

    //next up - replace relative directions with absolute direction

    return rule_line;
}

function deepCloneHS(HS) {
    let cloneHS = HS.map(function (arr) { return arr.map(function (deepArr) { return deepArr.slice(); }); });
    return cloneHS;
}

function deepCloneRule(rule) {
    let clonedRule = {
        direction: rule.direction,
        lhs: deepCloneHS(rule.lhs),
        rhs: deepCloneHS(rule.rhs),
        lineNumber: rule.lineNumber,
        late: rule.late,
        rigid: rule.rigid,
        groupNumber: rule.groupNumber,
        commands: rule.commands,
        randomRule: rule.randomRule
    };
    return clonedRule;
}

function rulesToArray(state) {
    let oldrules = state.rules;
    let rules = [];
    let loops = [];
    for (let i = 0; i < oldrules.length; i++) {
        let lineNumber = oldrules[i][1];
        let newrule = processRuleString(oldrules[i], state, rules);
        if (newrule === null) {
            continue;//error in processing string.
        }
        if (newrule.bracket !== undefined) {
            loops.push([lineNumber, newrule.bracket]);
            continue;
        }
        rules.push(newrule);
    }
    state.loops = loops;

    //now expand out rules with multiple directions
    let rules2 = [];
    for (let i = 0; i < rules.length; i++) {
        let rule = rules[i];
        let ruledirs = rule.directions;
        for (let j = 0; j < ruledirs.length; j++) {
            let dir = ruledirs[j];
            // The following block is never getting hit by any tests. 
            // Presumably in the past it was used to expand out rules with
            // multiple directions, but now that's done somewhere else.
            if (dir in directionaggregates && directionalRule(rule)) {
                let dirs = directionaggregates[dir];
                for (let k = 0; k < dirs.length; k++) {
                    let modifiedrule = deepCloneRule(rule);
                    modifiedrule.direction = dirs[k];
                    rules2.push(modifiedrule);
                }
            } else {
                let modifiedrule = deepCloneRule(rule);
                modifiedrule.direction = dir;
                rules2.push(modifiedrule);
            }
        }
    }

    for (let i = 0; i < rules2.length; i++) {
        let rule = rules2[i];
        //remove relative directions
        convertRelativeDirsToAbsolute(rule);
        //optional: replace up/left rules with their down/right equivalents
        rewriteUpLeftRules(rule);
        //replace aggregates with what they mean
        atomizeAggregates(state, rule);

        if (state.invalid) {
            return;
        }

        //replace synonyms with what they mean
        rephraseSynonyms(state, rule);
    }

    let rules3 = [];
    //expand property rules
    for (let i = 0; i < rules2.length; i++) {
        let rule = rules2[i];
        rules3 = rules3.concat(concretizeMovingRule(state, rule, rule.lineNumber));
    }

    let rules4 = [];
    for (let i = 0; i < rules3.length; i++) {
        let rule = rules3[i];
        rules4 = rules4.concat(concretizePropertyRule(state, rule, rule.lineNumber));

    }

    for (let i = 0; i < rules4.length; i++) {
        let rule = rules4[i];
        makeSpawnedObjectsStationary(state, rule, rule.lineNumber);
    }
    state.rules = rules4;
}

function containsEllipsis(rule) {
    for (let i = 0; i < rule.lhs.length; i++) {
        for (let j = 0; j < rule.lhs[i].length; j++) {
            if (rule.lhs[i][j][1] === '...')
                return true;
        }
    }
    return false;
}

function rewriteUpLeftRules(rule) {
    if (containsEllipsis(rule)) {
        return;
    }

    if (rule.direction === 'up') {
        rule.direction = 'down';
    } else if (rule.direction === 'left') {
        rule.direction = 'right';
    } else {
        return;
    }

    for (let i = 0; i < rule.lhs.length; i++) {
        rule.lhs[i].reverse();
        if (rule.rhs.length > 0) {
            rule.rhs[i].reverse();
        }
    }
}

//expands all properties to list of all things it could be, filterio
function getPossibleObjectsFromCell(state, cell) {
    const result = [];
    for (let j = 0; j < cell.length; j += 2) {
        const dir = cell[j];
        const name = cell[j + 1];
        if (name in state.objects) {
            result.push(name);
        }
        else if (name in state.propertiesDict) {
            const aliases = state.propertiesDict[name];
            for (let k = 0; k < aliases.length; k++) {
                const alias = aliases[k];
                result.push(alias);
            }
        }
    }
    return result;
}

function getPropertiesFromCell(state, cell) {
    let result = [];
    for (let j = 0; j < cell.length; j += 2) {
        let dir = cell[j];
        let name = cell[j + 1];
        if (dir === "random") {
            continue;
        }
        if (name in state.propertiesDict) {
            result.push(name);
        }
    }
    return result;
}

//returns you a list of object names in that cell that're moving
function getMovings(state, cell) {
    let result = [];
    for (let j = 0; j < cell.length; j += 2) {
        let dir = cell[j];
        let name = cell[j + 1];
        if (dir in directionaggregates) {
            result.push([name, dir]);
        }
    }
    return result;
}

function concretizePropertyInCell(cell, property, concreteType) {
    for (let j = 0; j < cell.length; j += 2) {
        if (cell[j + 1] === property && cell[j] !== "random") {
            cell[j + 1] = concreteType;
        }
    }
}

function concretizeMovingInCell(cell, ambiguousMovement, nameToMove, concreteDirection) {
    for (let j = 0; j < cell.length; j += 2) {
        if (cell[j] === ambiguousMovement && cell[j + 1] === nameToMove) {
            cell[j] = concreteDirection;
        }
    }
}

function concretizeMovingInCellByAmbiguousMovementName(cell, ambiguousMovement, concreteDirection) {
    for (let j = 0; j < cell.length; j += 2) {
        if (cell[j] === ambiguousMovement) {
            cell[j] = concreteDirection;
        }
    }
}

function expandNoPrefixedProperties(state, cell) {
    let expanded = [];
    for (let i = 0; i < cell.length; i += 2) {
        let dir = cell[i];
        let name = cell[i + 1];

        if (dir === 'no' && (name in state.propertiesDict)) {
            let aliases = state.propertiesDict[name];
            for (let j = 0; j < aliases.length; j++) {
                let alias = aliases[j];
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

function concretizePropertyRule(state, rule, lineNumber) {

    //step 1, rephrase rule to change "no flying" to "no cat no bat"
    for (let i = 0; i < rule.lhs.length; i++) {
        let cur_cellrow_l = rule.lhs[i];
        for (let j = 0; j < cur_cellrow_l.length; j++) {
            cur_cellrow_l[j] = expandNoPrefixedProperties(state, cur_cellrow_l[j]);
            if (rule.rhs.length > 0)
                rule.rhs[i][j] = expandNoPrefixedProperties(state, rule.rhs[i][j]);
        }
    }

    //are there any properties we could avoid processing?
    // e.g. [> player | movable] -> [> player | > movable],
    // 		doesn't need to be split up (assuming single-layer player/block aggregates)

    // we can't manage this if they're being used to disambiguate
    let ambiguousProperties = {};

    for (let j = 0; j < rule.rhs.length; j++) {
        let row_l = rule.lhs[j];
        let row_r = rule.rhs[j];
        for (let k = 0; k < row_r.length; k++) {
            let properties_l = getPropertiesFromCell(state, row_l[k]);
            let properties_r = getPropertiesFromCell(state, row_r[k]);
            for (let prop_n = 0; prop_n < properties_r.length; prop_n++) {
                let property = properties_r[prop_n];
                if (properties_l.indexOf(property) === -1) {
                    ambiguousProperties[property] = true;
                }
            }
        }
    }

    let shouldremove;
    let result = [rule];
    let modified = true;
    while (modified) {
        modified = false;
        for (let i = 0; i < result.length; i++) {
            //only need to iterate through lhs
            let cur_rule = result[i];
            shouldremove = false;
            for (let j = 0; j < cur_rule.lhs.length && !shouldremove; j++) {
                let cur_rulerow = cur_rule.lhs[j];
                for (let k = 0; k < cur_rulerow.length && !shouldremove; k++) {
                    let cur_cell = cur_rulerow[k];
                    let properties = getPropertiesFromCell(state, cur_cell);
                    for (let prop_n = 0; prop_n < properties.length; ++prop_n) {
                        let property = properties[prop_n];

                        if (state.propertiesSingleLayer.hasOwnProperty(property) &&
                            ambiguousProperties[property] !== true) {
                            // we don't need to explode this property
                            continue;
                        }

                        let aliases = state.propertiesDict[property];

                        shouldremove = true;
                        modified = true;

                        //just do the base property, let future iterations take care of the others

                        for (let l = 0; l < aliases.length; l++) {
                            let concreteType = aliases[l];
                            let newrule = deepCloneRule(cur_rule);
                            newrule.propertyReplacement = {};
                            for (let prop in cur_rule.propertyReplacement) {
                                if (cur_rule.propertyReplacement.hasOwnProperty(prop)) {
                                    let propDat = cur_rule.propertyReplacement[prop];
                                    newrule.propertyReplacement[prop] = [propDat[0], propDat[1]];
                                }
                            }

                            concretizePropertyInCell(newrule.lhs[j][k], property, concreteType);
                            if (newrule.rhs.length > 0) {
                                concretizePropertyInCell(newrule.rhs[j][k], property, concreteType); //do for the corresponding rhs cell as well
                            }

                            if (newrule.propertyReplacement[property] === undefined) {
                                newrule.propertyReplacement[property] = [concreteType, 1];
                            } else {
                                newrule.propertyReplacement[property][1] = newrule.propertyReplacement[property][1] + 1;
                            }

                            result.push(newrule);
                        }

                        break;
                    }
                }
            }
            if (shouldremove) {
                result.splice(i, 1);
                i--;
            }
        }
    }


    for (let i = 0; i < result.length; i++) {
        //for each rule
        let cur_rule = result[i];
        if (cur_rule.propertyReplacement === undefined) {
            continue;
        }

        //for each property replacement in that rule
        for (let property of Object.keys(cur_rule.propertyReplacement)) {
            let replacementInfo = cur_rule.propertyReplacement[property];
            let concreteType = replacementInfo[0];
            let occurrenceCount = replacementInfo[1];
            if (occurrenceCount === 1) {
                //do the replacement
                for (let j = 0; j < cur_rule.rhs.length; j++) {
                    let cellRow_rhs = cur_rule.rhs[j];
                    for (let k = 0; k < cellRow_rhs.length; k++) {
                        let cell = cellRow_rhs[k];
                        concretizePropertyInCell(cell, property, concreteType);
                    }
                }
            }
        }
    }

    //if any properties remain on the RHSes, bleep loudly
    let rhsPropertyRemains = '';
    for (let i = 0; i < result.length; i++) {
        let cur_rule = result[i];
        delete cur_rule.propertyReplacement;
        for (let j = 0; j < cur_rule.rhs.length; j++) {
            let cur_rulerow = cur_rule.rhs[j];
            for (let k = 0; k < cur_rulerow.length; k++) {
                let cur_cell = cur_rulerow[k];
                let properties = getPropertiesFromCell(state, cur_cell);
                for (let prop_n = 0; prop_n < properties.length; prop_n++) {
                    if (ambiguousProperties.hasOwnProperty(properties[prop_n])) {
                        rhsPropertyRemains = properties[prop_n];
                    }
                }
            }
        }
    }


    if (rhsPropertyRemains.length > 0) {
        logError('This rule has a property on the right-hand side, "' + rhsPropertyRemains.toUpperCase() + "\", that can't be inferred from the left-hand side.  (either for every property on the right there has to be a corresponding one on the left in the same cell, OR, if there's a single occurrence of a particular property name on the left, all properties of the same name on the right are assumed to be the same).", lineNumber);
        return [];
    }

    return result;
}

function makeSpawnedObjectsStationary(state, rule, lineNumber) {
    //movement not getting correctly cleared from tile #492
    //[ > Player | ] -> [ Crate | Player ] if there was a player already in the second cell, it's not replaced with a stationary player.
    //if there are properties remaining by this stage, just ignore them ( c.f. "[ >  Moveable | Moveable ] -> [ > Moveable | > Moveable ]" in block faker, what's left in this form) - this only happens IIRC when the properties span a single layer so it's)
    //if am object without moving-annotations appears on the RHS, and that object is not present on the lhs (either explicitly as an object, or implicitly in a property), add a 'stationary'
    if (rule.late) {
        return;
    }

    for (let j = 0; j < rule.rhs.length; j++) {
        let row_l = rule.lhs[j];
        let row_r = rule.rhs[j];
        for (let k = 0; k < row_r.length; k++) {
            let cell = row_r[k];

            //this is super intricate. uff. 
            let objects_l = getPossibleObjectsFromCell(state, row_l[k]);
            let layers = objects_l.map(n => state.objects[n].layer);
            for (let l = 0; l < cell.length; l += 2) {
                let dir = cell[l];
                if (dir !== "") {
                    continue;
                }
                let name = cell[l + 1];
                if (name in state.propertiesDict || objects_l.indexOf(name) >= 0) {
                    continue;
                }
                let r_layer = state.objects[name].layer;
                if (layers.indexOf(r_layer) === -1) {
                    cell[l] = 'stationary';
                }
            }
        }
    }

}

function concretizeMovingRule(state, rule, lineNumber) {

    let shouldremove;
    let result = [rule];
    let modified = true;
    while (modified) {
        modified = false;
        for (let i = 0; i < result.length; i++) {
            //only need to iterate through lhs
            let cur_rule = result[i];
            shouldremove = false;
            for (let j = 0; j < cur_rule.lhs.length; j++) {
                let cur_rulerow = cur_rule.lhs[j];
                for (let k = 0; k < cur_rulerow.length; k++) {
                    let cur_cell = cur_rulerow[k];
                    let movings = getMovings(state, cur_cell); //finds aggregate directions
                    if (movings.length > 0) {
                        shouldremove = true;
                        modified = true;

                        //just do the base property, let future iterations take care of the others
                        let cand_name = movings[0][0];
                        let ambiguous_dir = movings[0][1];
                        let concreteDirs = directionaggregates[ambiguous_dir];
                        for (let l = 0; l < concreteDirs.length; l++) {
                            let concreteDirection = concreteDirs[l];
                            let newrule = deepCloneRule(cur_rule);

                            //deep copy replacements
                            newrule.movingReplacement = {};
                            for (let moveTerm in cur_rule.movingReplacement) {
                                if (cur_rule.movingReplacement.hasOwnProperty(moveTerm)) {
                                    let moveDat = cur_rule.movingReplacement[moveTerm];
                                    newrule.movingReplacement[moveTerm] = [moveDat[0], moveDat[1], moveDat[2], moveDat[3], moveDat[4], moveDat[5]];
                                }
                            }
                            newrule.aggregateDirReplacement = {};
                            for (let moveTerm in cur_rule.aggregateDirReplacement) {
                                if (cur_rule.aggregateDirReplacement.hasOwnProperty(moveTerm)) {
                                    let moveDat = cur_rule.aggregateDirReplacement[moveTerm];
                                    newrule.aggregateDirReplacement[moveTerm] = [moveDat[0], moveDat[1], moveDat[2]];
                                }
                            }

                            concretizeMovingInCell(newrule.lhs[j][k], ambiguous_dir, cand_name, concreteDirection);
                            if (newrule.rhs.length > 0) {
                                concretizeMovingInCell(newrule.rhs[j][k], ambiguous_dir, cand_name, concreteDirection); //do for the corresponding rhs cell as well
                            }

                            if (newrule.movingReplacement[cand_name + ambiguous_dir] === undefined) {
                                newrule.movingReplacement[cand_name + ambiguous_dir] = [concreteDirection, 1, ambiguous_dir, cand_name, j, k];
                            } else {
                                let mr = newrule.movingReplacement[cand_name + ambiguous_dir];
                                if (j !== mr[4] || k !== mr[5]) {
                                    mr[1] = mr[1] + 1;
                                }
                            }
                            if (newrule.aggregateDirReplacement[ambiguous_dir] === undefined) {
                                newrule.aggregateDirReplacement[ambiguous_dir] = [concreteDirection, 1, ambiguous_dir];
                            } else {
                                newrule.aggregateDirReplacement[ambiguous_dir][1] = newrule.aggregateDirReplacement[ambiguous_dir][1] + 1;
                            }

                            result.push(newrule);
                        }
                    }
                }
            }
            if (shouldremove) {
                result.splice(i, 1);
                i--;
            }
        }
    }


    for (let i = 0; i < result.length; i++) {
        //for each rule
        let cur_rule = result[i];
        if (cur_rule.movingReplacement === undefined) {
            continue;
        }
        let ambiguous_movement_dict = {};
        //strict first - matches movement direction to objects
        //for each property replacement in that rule
        for (let cand_name in cur_rule.movingReplacement) {
            if (cur_rule.movingReplacement.hasOwnProperty(cand_name)) {
                let replacementInfo = cur_rule.movingReplacement[cand_name];
                let concreteMovement = replacementInfo[0];
                let occurrenceCount = replacementInfo[1];
                let ambiguousMovement = replacementInfo[2];
                let ambiguousMovement_attachedObject = replacementInfo[3];

                if (occurrenceCount === 1) {
                    //do the replacement
                    for (let j = 0; j < cur_rule.rhs.length; j++) {
                        let cellRow_rhs = cur_rule.rhs[j];
                        for (let k = 0; k < cellRow_rhs.length; k++) {
                            let cell = cellRow_rhs[k];
                            concretizeMovingInCell(cell, ambiguousMovement, ambiguousMovement_attachedObject, concreteMovement);
                        }
                    }
                }

            }
        }

        //I don't fully understand why the following part is needed (and I wrote this yesterday), but it's not obviously malicious.
        let ambiguous_movement_names_dict = {};
        for (let cand_name in cur_rule.aggregateDirReplacement) {
            if (cur_rule.aggregateDirReplacement.hasOwnProperty(cand_name)) {
                let replacementInfo = cur_rule.aggregateDirReplacement[cand_name];
                let concreteMovement = replacementInfo[0];
                let occurrenceCount = replacementInfo[1];
                let ambiguousMovement = replacementInfo[2];
                //are both the following boolean bits necessary, or just the latter? ah well, no harm it seems.
                if ((ambiguousMovement in ambiguous_movement_names_dict) || (occurrenceCount !== 1)) {
                    ambiguous_movement_names_dict[ambiguousMovement] = "INVALID";
                } else {
                    ambiguous_movement_names_dict[ambiguousMovement] = concreteMovement
                }
            }
        }

        //for each ambiguous word, if there's a single ambiguous movement specified in the whole lhs, then replace that wholesale
        for (let ambiguousMovement in ambiguous_movement_dict) {
            if (ambiguous_movement_dict.hasOwnProperty(ambiguousMovement) && ambiguousMovement !== "INVALID") {
                concreteMovement = ambiguous_movement_dict[ambiguousMovement];
                if (concreteMovement === "INVALID") {
                    continue;
                }
                for (let j = 0; j < cur_rule.rhs.length; j++) {
                    let cellRow_rhs = cur_rule.rhs[j];
                    for (let k = 0; k < cellRow_rhs.length; k++) {
                        let cell = cellRow_rhs[k];
                        concretizeMovingInCellByAmbiguousMovementName(cell, ambiguousMovement, concreteMovement);
                    }
                }
            }
        }


        //further replacements - if a movement word appears once on the left, can use to disambiguate remaining ones on the right
        for (let ambiguousMovement in ambiguous_movement_names_dict) {
            if (ambiguous_movement_names_dict.hasOwnProperty(ambiguousMovement) && ambiguousMovement !== "INVALID") {
                let concreteMovement = ambiguous_movement_names_dict[ambiguousMovement];
                if (concreteMovement === "INVALID") {
                    continue;
                }
                for (let j = 0; j < cur_rule.rhs.length; j++) {
                    let cellRow_rhs = cur_rule.rhs[j];
                    for (let k = 0; k < cellRow_rhs.length; k++) {
                        let cell = cellRow_rhs[k];
                        concretizeMovingInCellByAmbiguousMovementName(cell, ambiguousMovement, concreteMovement);
                    }
                }
            }
        }
    }

    //if any properties remain on the RHSes, bleep loudly
    let rhsAmbiguousMovementsRemain = '';
    
    outerloop: for (const currentRule of result) {
        delete currentRule.movingReplacement;        
        for (const ruleRow of currentRule.rhs) {
            for (const cell of ruleRow) {
                const movings = getMovings(state, cell);
                if (movings.length > 0) {
                    rhsAmbiguousMovementsRemain = movings[0][1];
                    break outerloop;
                }
            }
        }
    }


    if (rhsAmbiguousMovementsRemain.length > 0) {
        logError('This rule has an ambiguous movement on the right-hand side, "' + rhsAmbiguousMovementsRemain + "\", that can't be inferred from the left-hand side.  (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that)).", lineNumber);
        state.invalid = true;
    }

    return result;
}

function rephraseSynonyms(state, rule) {
    const processCell = (cell) => {
        for (let i = 1; i < cell.length; i += 2) {
            const name = cell[i];
            if (name in state.synonymsDict) {
                cell[i] = state.synonymsDict[name];
            }
        }
    };

    for (let i = 0; i < rule.lhs.length; i++) {
        const cellrow_l = rule.lhs[i];
        const cellrow_r = rule.rhs[i];
        
        for (let j = 0; j < cellrow_l.length; j++) {
            processCell(cellrow_l[j]);
            if (rule.rhs.length > 0) {
                processCell(cellrow_r[j]);
            }
        }
    }
}

function atomizeAggregates(state, rule) {
    const processCellRow = (cellrow) => {
        for (let j = 0; j < cellrow.length; j++) {
            atomizeCellAggregates(state, cellrow[j], rule.lineNumber);
        }
    };

    for (let i = 0; i < rule.lhs.length; i++) {
        processCellRow(rule.lhs[i]);
    }
    for (let i = 0; i < rule.rhs.length; i++) {
        processCellRow(rule.rhs[i]);
    }
}

function atomizeCellAggregates(state, cell, lineNumber) {
    for (let i = 0; i < cell.length; i += 2) {
        const dir = cell[i];
        const name = cell[i + 1];
        if (name in state.aggregatesDict) {
            if (dir === 'no') {
                logError("You cannot use 'no' to exclude the aggregate object " + name.toUpperCase() + " (defined using 'AND'), only regular objects, or properties (objects defined using 'OR').  If you want to do this, you'll have to write it out yourself the long way.", lineNumber);
            }
            const equivs = state.aggregatesDict[name];
            cell[i + 1] = equivs[0];
            for (let j = 1; j < equivs.length; j++) {
                cell.push(dir); //push the direction
                cell.push(equivs[j]);
            }
        }
    }
}

function convertRelativeDirsToAbsolute(rule) {
    const forward = rule.direction;
    for (let i = 0; i < rule.lhs.length; i++) {
        const cellrow = rule.lhs[i];
        for (let j = 0; j < cellrow.length; j++) {
            const cell = cellrow[j];
            absolutifyRuleCell(forward, cell);
        }
    }
    for (let i = 0; i < rule.rhs.length; i++) {
        const cellrow = rule.rhs[i];
        for (let j = 0; j < cellrow.length; j++) {
            const cell = cellrow[j];
            absolutifyRuleCell(forward, cell);
        }
    }
}

const relativeDirs = ['^', 'v', '<', '>', 'parallel', 'perpendicular']; //used to index the following
//I use _par/_perp just to keep track of providence for replacement purposes later.
const relativeDict = {
    'right': ['up', 'down', 'left', 'right', 'horizontal_par', 'vertical_perp'],
    'up': ['left', 'right', 'down', 'up', 'vertical_par', 'horizontal_perp'],
    'down': ['right', 'left', 'up', 'down', 'vertical_par', 'horizontal_perp'],
    'left': ['down', 'up', 'right', 'left', 'horizontal_par', 'vertical_perp']
};

function absolutifyRuleCell(forward, cell) {
    for (let i = 0; i < cell.length; i += 2) {
        const c = cell[i];
        const index = relativeDirs.indexOf(c);
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

const dirMasks = {
    'up': parseInt('00001', 2),
    'down': parseInt('00010', 2),
    'left': parseInt('00100', 2),
    'right': parseInt('01000', 2),
    'moving': parseInt('01111', 2),
    'no': parseInt('00011', 2),
    'randomdir': parseInt('00101', 2),
    'random': parseInt('10010', 2),
    'action': parseInt('10000', 2),
    '': parseInt('00000', 2)
};

function rulesToMask(state) {
    const layerCount = state.collisionLayers.length;
    const layerTemplate = Array(layerCount).fill(null);
    const STRIDE_5 = 5; // Magic number for bit shifting

    outerloop: for (let ruleIndex = 0; ruleIndex < state.rules.length; ruleIndex++) {
        const rule = state.rules[ruleIndex];
        
        for (let rowIndex = 0; rowIndex < rule.lhs.length; rowIndex++) {
            const [cellrow_l, cellrow_r] = [rule.lhs[rowIndex], rule.rhs[rowIndex]];
            
            for (let colIndex = 0; colIndex < cellrow_l.length; colIndex++) {
                const cell_l = cellrow_l[colIndex];
                const layersUsed_l = [...layerTemplate];
                
                // Initialize bit vectors for the current cell
                const bitVectors = {
                    objectsPresent: new BitVec(STRIDE_OBJ),
                    objectsMissing: new BitVec(STRIDE_OBJ),
                    movementsPresent: new BitVec(STRIDE_MOV),
                    movementsMissing: new BitVec(STRIDE_MOV),
                    objectlayers_l: new BitVec(STRIDE_MOV)
                };
                
                const anyObjectsPresent = [];

                // Process left-hand side cell
                for (let i = 0; i < cell_l.length; i += 2) {
                    const [object_dir, object_name] = [cell_l[i], cell_l[i + 1]];
                    
                    // Handle special cases
                    if (object_dir === '...') {
                        if (cell_l.length !== 2) {
                            logError("You can't have anything in with an ellipsis. Sorry.", rule.lineNumber);
                            state.rules.splice(ruleIndex, 1);
                            ruleIndex--;
                            continue outerloop;
                        }
                        if (colIndex === 0 || colIndex === cellrow_l.length - 1) {
                            logError("There's no point in putting an ellipsis at the very start or the end of a rule", rule.lineNumber);
                        }
                        if (rule.rhs.length > 0) {
                            const rhscell = cellrow_r[colIndex];
                            if (rhscell.length !== 2 || rhscell[0] !== '...') {
                                logError("An ellipsis on the left must be matched by one in the corresponding place on the right.", rule.lineNumber);
                            }
                        }
                        bitVectors.objectsPresent = ellipsisPattern;
                        break;
                    }
                    
                    if (object_dir === 'random') {
                        logError("RANDOM cannot be matched on the left-hand side, it can only appear on the right", rule.lineNumber);
                        continue;
                    }

                    // Process regular object
                    const object = state.objects[object_name];
                    const objectMask = state.objectMasks[object_name];
                    const layerIndex = object ? (object.layer | 0) : state.propertiesSingleLayer[object_name];

                    if (typeof layerIndex === "undefined") {
                        logError(`Oops! ${object_name.toUpperCase()} not assigned to a layer.`, rule.lineNumber);
                    }

                    if (object_dir === 'no') {
                        bitVectors.objectsMissing.ior(objectMask);
                    } else {
                        const existingname = layersUsed_l[layerIndex];
                        if (existingname !== null) {
                            rule.discard = [object_name.toUpperCase(), existingname.toUpperCase()];
                        }

                        layersUsed_l[layerIndex] = object_name;

                        if (object) {
                            bitVectors.objectsPresent.ior(objectMask);
                            bitVectors.objectlayers_l.ishiftor(0x1f, STRIDE_5 * layerIndex);
                        } else {
                            anyObjectsPresent.push(objectMask);
                        }

                        const movementMask = object_dir === 'stationary' ? 
                            bitVectors.movementsMissing : bitVectors.movementsPresent;
                        movementMask.ishiftor(object_dir === 'stationary' ? 0x1f : dirMasks[object_dir], 
                                           STRIDE_5 * layerIndex);
                    }
                }

                // Handle ellipsis pattern
                if (bitVectors.objectsPresent === ellipsisPattern) {
                    cellrow_l[colIndex] = ellipsisPattern;
                    continue;
                }

                // Create cell pattern
                cellrow_l[colIndex] = new CellPattern([
                    bitVectors.objectsPresent,
                    bitVectors.objectsMissing,
                    anyObjectsPresent,
                    bitVectors.movementsPresent,
                    bitVectors.movementsMissing,
                    null
                ]);

                // Check for invalid patterns
                if (bitVectors.objectsPresent.anyBitsInCommon(bitVectors.objectsMissing)) {
                    const ln = rule.lineNumber;
                    const hasAdjacentRule = (ruleIndex > 0 && state.rules[ruleIndex - 1].lineNumber === ln) || 
                                         (ruleIndex + 1 < state.rules.length && state.rules[ruleIndex + 1].lineNumber === ln);
                    
                    if (!hasAdjacentRule) {
                        logWarning('This rule has some content of the form "X no X" (either directly or maybe indirectly - check closely how the terms are defined if nothing stands out) which can never match and so the rule is getting removed during compilation.', rule.lineNumber);
                    }
                    state.rules.splice(ruleIndex, 1);
                    ruleIndex--;
                    continue;
                }

                if (rule.rhs.length === 0) continue;
                const cell_r = cellrow_r[colIndex];

                // Check for mismatched ellipsis
                if (cell_r[0] === '...' && cell_l[0] !== '...') {
                    logError("An ellipsis on the right must be matched by one in the corresponding place on the left.", rule.lineNumber);
                }
                
                // Validate ellipsis in right-hand side
                for (let i = 0; i < cell_r.length; i += 2) {
                    if (cell_r[i] === '...' && cell_r.length !== 2) {
                        logError("You can't have anything in with an ellipsis. Sorry.", rule.lineNumber);
                    }
                }

                const layersUsed_r = [...layerTemplate];
                const layersUsedRand_r = [...layerTemplate];

                const rhsBitVectors = {
                    objectsClear: new BitVec(STRIDE_OBJ),
                    objectsSet: new BitVec(STRIDE_OBJ),
                    movementsClear: new BitVec(STRIDE_MOV),
                    movementsSet: new BitVec(STRIDE_MOV),
                    objectlayers_r: new BitVec(STRIDE_MOV),
                    randomMask_r: new BitVec(STRIDE_OBJ),
                    postMovementsLayerMask_r: new BitVec(STRIDE_MOV),
                    randomDirMask_r: new BitVec(STRIDE_MOV)
                };

                // Process right-hand side cell
                for (let i = 0; i < cell_r.length; i += 2) {
                    const [object_dir, object_name] = [cell_r[i], cell_r[i + 1]];

                    if (object_dir === '...') break;
                    
                    if (object_dir === 'random') {
                        if (object_name in state.objectMasks) {
                            const mask = state.objectMasks[object_name];
                            rhsBitVectors.randomMask_r.ior(mask);
                            
                            const values = state.propertiesDict.hasOwnProperty(object_name) ? 
                                state.propertiesDict[object_name] : [object_name];
                            
                            if (values.length === 1) {
                                logWarning(`In this rule you're asking me to spawn a random ${object_name.toUpperCase()} for you, but that's already a concrete single object. You wanna be using random with properties (things defined in terms of OR in the legend) so there's some things to select between.`, rule.lineNumber);
                            }

                            for (const subobject of values) {
                                const layerIndex = state.objects[subobject].layer | 0;
                                const existingname = layersUsed_r[layerIndex];
                                
                                if (existingname !== null) {
                                    const [o1, o2] = [subobject.toUpperCase(), existingname.toUpperCase()];
                                    if (o1 !== o2) {
                                        logWarning(`This rule may try to spawn a ${o1} with random, but also requires a ${o2} be here, which is on the same layer - they shouldn't be able to coexist!`, rule.lineNumber);
                                    }
                                }
                                layersUsedRand_r[layerIndex] = subobject;
                            }
                        } else {
                            logError(`You want to spawn a random "${object_name.toUpperCase()}", but I don't know how to do that`, rule.lineNumber);
                        }
                        continue;
                    }

                    const object = state.objects[object_name];
                    const objectMask = state.objectMasks[object_name];
                    const layerIndex = object ? (object.layer | 0) : state.propertiesSingleLayer[object_name];

                    if (object_dir === 'no') {
                        rhsBitVectors.objectsClear.ior(objectMask);
                    } else {
                        const existingname = layersUsed_r[layerIndex] || layersUsedRand_r[layerIndex];
                        if (existingname !== null && !rule.hasOwnProperty('discard')) {
                            logError(`Rule matches object types that can't overlap: "${object_name.toUpperCase()}" and "${existingname.toUpperCase()}".`, rule.lineNumber);
                        }

                        layersUsed_r[layerIndex] = object_name;

                        if (object_dir.length > 0) {
                            rhsBitVectors.postMovementsLayerMask_r.ishiftor(0x1f, STRIDE_5 * layerIndex);
                        }

                        const layerMask = state.layerMasks[layerIndex];

                        if (object) {
                            rhsBitVectors.objectsSet.ibitset(object.id);
                            rhsBitVectors.objectsClear.ior(layerMask);
                            rhsBitVectors.objectlayers_r.ishiftor(0x1f, STRIDE_5 * layerIndex);
                        }

                        if (object_dir === 'stationary') {
                            rhsBitVectors.movementsClear.ishiftor(0x1f, STRIDE_5 * layerIndex);
                        } else if (object_dir === 'randomdir') {
                            rhsBitVectors.randomDirMask_r.ishiftor(dirMasks[object_dir], STRIDE_5 * layerIndex);
                        } else {
                            rhsBitVectors.movementsSet.ishiftor(dirMasks[object_dir], STRIDE_5 * layerIndex);
                        }
                    }
                }

                // Clear old objects and movements if needed
                if (!bitVectors.objectsPresent.bitsSetInArray(rhsBitVectors.objectsSet.data)) {
                    rhsBitVectors.objectsClear.ior(bitVectors.objectsPresent);
                }
                if (!bitVectors.movementsPresent.bitsSetInArray(rhsBitVectors.movementsSet.data)) {
                    rhsBitVectors.movementsClear.ior(bitVectors.movementsPresent);
                }

                // Handle layer-specific clearing
                for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
                    if (layersUsed_l[layerIndex] !== null && layersUsed_r[layerIndex] === null) {
                        rhsBitVectors.objectsClear.ior(state.layerMasks[layerIndex]);
                        rhsBitVectors.postMovementsLayerMask_r.ishiftor(0x1f, STRIDE_5 * layerIndex);
                    }
                }

                bitVectors.objectlayers_l.iclear(rhsBitVectors.objectlayers_r);
                rhsBitVectors.postMovementsLayerMask_r.ior(bitVectors.objectlayers_l);

                // Set replacement if any changes would occur
                const hasChanges = !rhsBitVectors.objectsClear.iszero() || 
                                 !rhsBitVectors.objectsSet.iszero() || 
                                 !rhsBitVectors.movementsClear.iszero() || 
                                 !rhsBitVectors.movementsSet.iszero() || 
                                 !rhsBitVectors.postMovementsLayerMask_r.iszero() || 
                                 !rhsBitVectors.randomMask_r.iszero() || 
                                 !rhsBitVectors.randomDirMask_r.iszero();

                if (hasChanges) {
                    const target_cell_pattern = cellrow_l[colIndex];
                    target_cell_pattern.replacement = new CellReplacement([
                        rhsBitVectors.objectsClear,
                        rhsBitVectors.objectsSet,
                        rhsBitVectors.movementsClear,
                        rhsBitVectors.movementsSet,
                        rhsBitVectors.postMovementsLayerMask_r,
                        rhsBitVectors.randomMask_r,
                        rhsBitVectors.randomDirMask_r
                    ]);
                    target_cell_pattern.replace = target_cell_pattern.generateReplaceFunction(STRIDE_OBJ, STRIDE_MOV,rule);
                }
            }
        }
    }
}

function cellRowMasks(rule) {
    const ruleMasks = [];
    const lhs = rule[1];
    for (let i = 0; i < lhs.length; i++) {
        const cellRow = lhs[i];
        const rowMask = new BitVec(STRIDE_OBJ);
        for (let j = 0; j < cellRow.length; j++) {
            if (cellRow[j] === ellipsisPattern)
                continue;
            rowMask.ior(cellRow[j].objectsPresent);
        }
        ruleMasks.push(rowMask);
    }
    return ruleMasks;
}

function cellRowMasks_Movements(rule) {
    const ruleMasks_mov = [];
    const lhs = rule[1];
    for (let i = 0; i < lhs.length; i++) {
        const cellRow = lhs[i];
        const rowMask = new BitVec(STRIDE_MOV);
        for (let j = 0; j < cellRow.length; j++) {
            if (cellRow[j] === ellipsisPattern)
                continue;
            rowMask.ior(cellRow[j].movementsPresent);
        }
        ruleMasks_mov.push(rowMask);
    }
    return ruleMasks_mov;
}

function collapseRules(groups) {
    for (let gn = 0; gn < groups.length; gn++) {
        const rules = groups[gn];
        for (let i = 0; i < rules.length; i++) {
            const oldrule = rules[i];
            const newrule = [0, [], oldrule.rhs.length > 0, oldrule.lineNumber /*ellipses,group number,rigid,commands,randomrule,[cellrowmasks]*/];
            const ellipses = [];
            for (let j = 0; j < oldrule.lhs.length; j++) {
                ellipses.push(0);
            }

            newrule[0] = dirMasks[oldrule.direction];
            for (let j = 0; j < oldrule.lhs.length; j++) {
                const cellrow_l = oldrule.lhs[j];
                for (let k = 0; k < cellrow_l.length; k++) {
                    if (cellrow_l[k] === ellipsisPattern) {
                        ellipses[j]++;
                        if (ellipses[j] > 2) {
                            logError("You can't use more than two ellipses in a single cell match pattern.", oldrule.lineNumber);
                        } else {
                            if (k > 0 && cellrow_l[k - 1] === ellipsisPattern) {
                                logWarning("Why would you go and have two ellipses in a row like that? It's exactly the same as just having a single ellipsis, right?", oldrule.lineNumber);
                            }
                        }
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
            newrule.push(cellRowMasks_Movements(newrule));
            rules[i] = new Rule(newrule);
        }
    }
}



function ruleGroupDiscardOverlappingTest(ruleGroup) {
    if (ruleGroup.length === 0)
        return;

    let discards = [];

    for (let i = 0; i < ruleGroup.length; i++) {
        let rule = ruleGroup[i];
        if (rule.hasOwnProperty('discard')) {

            let beforesame = i === 0 ? false : ruleGroup[i - 1].lineNumber === rule.lineNumber;
            let aftersame = i === (ruleGroup.length - 1) ? false : ruleGroup[i + 1].lineNumber === rule.lineNumber;

            ruleGroup.splice(i, 1);

            let found = false;
            for (let j = 0; j < discards.length; j++) {
                let discard = discards[j];
                if (discard[0] === rule.discard[0] && discard[1] === rule.discard[1]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                discards.push(rule.discard)
            }

            //if rule before isn't of same linenumber, and rule after isn't of same linenumber, 
            //then a rule has been totally erased and you should throw an error!
            if (!(beforesame || aftersame) || ruleGroup.length === 0) {

                const example = discards[0];

                let parenthetical = "";
                if (discards.length > 1) {
                    parenthetical = " (ditto for ";
                    for (let j = 1; j < discards.length; j++) {
                        if (j > 1) {
                            parenthetical += ", "

                            if (j === discards.length - 1) {
                                parenthetical += "and ";
                            }
                        }

                        const thisdiscard = discards[j];
                        const p1 = thisdiscard[0];
                        const p2 = thisdiscard[1];
                        parenthetical += `${p1}/${p2}`;

                        if (j === 3 && discards.length > 4) {
                            parenthetical += " etc.";
                            break;
                        }
                    }
                    parenthetical += ")";
                }

                logError(`${example[0]} and ${example[1]} can never overlap${parenthetical}, but this rule requires that to happen, so it's being culled.`, rule.lineNumber);
            }
            i--;
        }
    }
}

function arrangeRulesByGroupNumber(state) {
    let aggregates = {};
    let aggregates_late = {};
    for (let i = 0; i < state.rules.length; i++) {
        let rule = state.rules[i];
        let targetArray = aggregates;
        if (rule.late) {
            targetArray = aggregates_late;
        }

        if (targetArray[rule.groupNumber] === undefined) {
            targetArray[rule.groupNumber] = [];
        }
        targetArray[rule.groupNumber].push(rule);
    }

    let result = [];
    for (let groupNumber in aggregates) {
        if (aggregates.hasOwnProperty(groupNumber)) {
            let ruleGroup = aggregates[groupNumber];
            ruleGroupDiscardOverlappingTest(ruleGroup);
            if (ruleGroup.length > 0) {
                result.push(ruleGroup);
            }
        }
    }
    let result_late = [];
    for (let groupNumber in aggregates_late) {
        if (aggregates_late.hasOwnProperty(groupNumber)) {
            let ruleGroup = aggregates_late[groupNumber];
            ruleGroupDiscardOverlappingTest(ruleGroup);
            if (ruleGroup.length > 0) {
                result_late.push(ruleGroup);
            }
        }
    }
    state.rules = result;

    //check that there're no late movements with direction requirements on the lhs
    state.lateRules = result_late;
}

function generateRigidGroupList(state) {
    let rigidGroupIndex_to_GroupIndex = [];
    let groupIndex_to_RigidGroupIndex = [];
    let groupNumber_to_GroupIndex = [];
    let groupNumber_to_RigidGroupIndex = [];
    let rigidGroups = [];
    for (let i = 0; i < state.rules.length; i++) {
        let ruleset = state.rules[i];
        let rigidFound = false;
        for (let j = 0; j < ruleset.length; j++) {
            let rule = ruleset[j];
            if (rule.rigid) {
                rigidFound = true;
            }
        }
        rigidGroups[i] = rigidFound;
        if (rigidFound) {
            let groupNumber = ruleset[0].groupNumber;
            groupNumber_to_GroupIndex[groupNumber] = i;
            let rigid_group_index = rigidGroupIndex_to_GroupIndex.length;
            groupIndex_to_RigidGroupIndex[i] = rigid_group_index;
            groupNumber_to_RigidGroupIndex[groupNumber] = rigid_group_index;
            rigidGroupIndex_to_GroupIndex.push(i);
        }
    }
    if (rigidGroupIndex_to_GroupIndex.length > 30) {
        let group_index = rigidGroupIndex_to_GroupIndex[30];
        logError("There can't be more than 30 rigid groups (rule groups containing rigid members).", state.rules[group_index][0].lineNumber);
    }
    state.hasRigidRules = groupIndex_to_RigidGroupIndex.length > 0;
    state.rigidGroups = rigidGroups;
    state.rigidGroupIndex_to_GroupIndex = rigidGroupIndex_to_GroupIndex;
    state.groupNumber_to_RigidGroupIndex = groupNumber_to_RigidGroupIndex;
    state.groupIndex_to_RigidGroupIndex = groupIndex_to_RigidGroupIndex;
}

function isObjectDefined(state, name) {

    let result = name in state.objects ||
        (state.aggregatesDict !== undefined && (name in state.aggregatesDict)) ||
        (state.propertiesDict !== undefined && (name in state.propertiesDict)) ||
        (state.synonymsDict !== undefined && (name in state.synonymsDict));

    if (state.legend_aggregates !== undefined) {
        for (let i = 0; i < state.legend_aggregates.length; i++) {
            if (state.legend_aggregates[i][0] === name) {
                result = true;
                break;
            }
        }
    }
    if (state.legend_properties !== undefined) {
        for (let i = 0; i < state.legend_properties.length; i++) {
            if (state.legend_properties[i][0] === name) {
                result = true;
                break;
            }
        }
    }
    if (state.legend_synonyms !== undefined) {
        for (let i = 0; i < state.legend_synonyms.length; i++) {
            if (state.legend_synonyms[i][0] === name) {
                result = true;
                break;
            }
        }
    }
    return result;
}

function getMaskFromName(state, name) {
    const objectMask = new BitVec(STRIDE_OBJ);
    if (name in state.objects) {
        const o = state.objects[name];
        objectMask.ibitset(o.id);
    }

    if (name in state.aggregatesDict) {
        const objectnames = state.aggregatesDict[name];
        for (let i = 0; i < objectnames.length; i++) {
            const n = objectnames[i];
            const o = state.objects[n];
            objectMask.ibitset(o.id);
        }
    }

    if (name in state.propertiesDict) {
        const objectnames = state.propertiesDict[name];
        for (let i = 0; i < objectnames.length; i++) {
            const n = objectnames[i];
            const o = state.objects[n];
            objectMask.ibitset(o.id);
        }
    }

    if (name in state.synonymsDict) {
        const n = state.synonymsDict[name];
        const o = state.objects[n];
        objectMask.ibitset(o.id);
    }

    if (objectMask.iszero()) {
        logErrorNoLine(`Error, didn't find any object called ${name}, either in the objects section, or the legends section.`);
    }
    return objectMask;
}

function generateMasks(state) {
    state.playerMask = getMaskFromName(state, 'player');

    let layerMasks = [];
    let layerCount = state.collisionLayers.length;
    for (let layer = 0; layer < layerCount; layer++) {
        let layerMask = new BitVec(STRIDE_OBJ);
        for (let j = 0; j < state.objectCount; j++) {
            let n = state.idDict[j];
            let o = state.objects[n];
            if (o.layer === layer) {
                layerMask.ibitset(o.id);
            }
        }
        layerMasks.push(layerMask);
    }
    state.layerMasks = layerMasks;

    let objectMask = {};
    for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            let o = state.objects[n];
            objectMask[n] = new BitVec(STRIDE_OBJ);
            objectMask[n].ibitset(o.id);
        }
    }

    // Synonyms can depend on properties, and properties can depend on synonyms.
    // Process them in order by combining & sorting by linenumber.

    let synonyms_and_properties = state.legend_synonyms.concat(state.legend_properties);
    synonyms_and_properties.sort(function (a, b) {
        return a.lineNumber - b.lineNumber;
    });

    for (let i = 0; i < synonyms_and_properties.length; i++) {
        let synprop = synonyms_and_properties[i];
        if (synprop.length === 2) {
            // synonym (a = b)
            objectMask[synprop[0]] = objectMask[synprop[1]];
        } else {
            // property (a = b or c)
            let val = new BitVec(STRIDE_OBJ);
            for (let j = 1; j < synprop.length; j++) {
                let n = synprop[j];
                val.ior(objectMask[n]);
            }
            objectMask[synprop[0]] = val;
        }
    }

    //use \n as a delimeter for internal-only objects
    let all_obj = new BitVec(STRIDE_OBJ);
    all_obj.inot();
    objectMask["\nall\n"] = all_obj;

    state.objectMasks = objectMask;


    state.aggregateMasks = {};

    //set aggregate masks similarly
    for (let aggregateName of Object.keys(state.aggregatesDict)) {
        let objectnames = state.aggregatesDict[aggregateName];

        let aggregateMask = new BitVec(STRIDE_OBJ);
        for (let i = 0; i < objectnames.length; i++) {
            let n = objectnames[i];
            let o = state.objects[n];
            aggregateMask.ior(objectMask[n]);
        }
        state.aggregateMasks[aggregateName] = aggregateMask;
    }
}

function checkObjectsAreLayered(state) {
    outer: for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            let found = false;
            for (let i = 0; i < state.collisionLayers.length; i++) {
                const layer = state.collisionLayers[i];
                for (let j = 0; j < layer.length; j++) {
                    if (layer[j] === n) {
                        found = true;
                        continue outer;
                    }
                }
            }
            if (found === false) {
                const o = state.objects[n];
                logError('Object "' + n.toUpperCase() + '" has been defined, but not assigned to a layer.', o.lineNumber);
            }
        }
    }
}

function isInt(value) {
    return !isNaN(value) && (function (x) { return (x | 0) === x; })(parseFloat(value))
}

function twiddleMetaData(state) {
    const newmetadata = {};
    for (let i = 0; i < state.metadata.length; i += 2) {
        const key = state.metadata[i];
        const val = state.metadata[i + 1];
        newmetadata[key] = val;
    }

    const getIntCheckedPositive = function (s, lineNumber) {
        if (!isFinite(s) || !isInt(s)) {
            logWarning(`Wasn't able to make sense of "${s}" as a (whole number) dimension.`, lineNumber);
            return NaN;
        }
        const result = parseInt(s);
        if (isNaN(result)) {
            logWarning(`Wasn't able to make sense of "${s}" as a dimension.`, lineNumber);
        }
        if (result <= 0) {
            logWarning(`The dimension given to me (you gave "${s}") is baad - it should be greater than 0.`, lineNumber);
        }
        return result;
    }

    const getCoords = function (val, lineNumber) {
        const coords = val.split('x');
        if (coords.length !== 2) {
            logWarning("Dimensions must be of the form AxB.", lineNumber);
            return null;
        } else {
            const intcoords = [getIntCheckedPositive(coords[0], lineNumber), getIntCheckedPositive(coords[1], lineNumber)];
            if (!isFinite(coords[0]) || !isFinite(coords[1]) || isNaN(intcoords[0]) || isNaN(intcoords[1])) {
                logWarning(`Couldn't understand the dimensions given to me (you gave "${val}") - should be of the form AxB.`, lineNumber);
                return null;
            } else {
                if (intcoords[0] <= 0 || intcoords[1] <= 0) {
                    logWarning(`The dimensions given to me (you gave "${val}") are baad - they should be > 0.`, lineNumber);
                }
                return intcoords;
            }
        }
    }

    if (newmetadata.flickscreen !== undefined) {
        const val = newmetadata.flickscreen;
        newmetadata.flickscreen = getCoords(val, state.metadata_lines.flickscreen);
        if (newmetadata.flickscreen === null) {
            delete newmetadata.flickscreen;
        }
    }
    if (newmetadata.zoomscreen !== undefined) {
        const val = newmetadata.zoomscreen;
        newmetadata.zoomscreen = getCoords(val, state.metadata_lines.zoomscreen);
        if (newmetadata.zoomscreen === null) {
            delete newmetadata.zoomscreen;
        }
    }

    state.metadata = newmetadata;
}

function processWinConditions(state) {
    //	[-1/0/1 (no,some,all),ob1,ob2] (ob2 is background by default)
    let newconditions = [];
    for (let i = 0; i < state.winconditions.length; i++) {
        let wincondition = state.winconditions[i];
        if (wincondition.length === 0) {
            //I feel like here should never be reached, right? Not sure if it warrants an error though.
            return;
        }
        let num = 0;
        switch (wincondition[0]) {
            case 'no':
                { num = -1; break; }
            case 'all':
                { num = 1; break; }
        }

        let lineNumber = wincondition[wincondition.length - 1];

        let n1 = wincondition[1];
        let n2;
        if (wincondition.length === 5) {
            n2 = wincondition[3];
        } else {
            n2 = '\nall\n';
        }

        let mask1 = 0;
        let mask2 = 0;
        let aggr1 = false;
        let aggr2 = false;

        if (wincondition.length <= 2) {
            logError('Win conditions is badly formatted - needs to look something like "No Fruit", "All Target On Crate", "Some Fruit", "Some Gold on Chest", "No Gold on Chest", or the like.', lineNumber);
        }
        else if (n1 in state.objectMasks) {
            aggr1 = false;
            mask1 = state.objectMasks[n1];
        } else if (n1 in state.aggregateMasks) {
            aggr1 = true;
            mask1 = state.aggregateMasks[n1];
        } else {
            logError('Unwelcome term "' + n1 + '" found in win condition. I don\'t know what I\'m supposed to do with this. ', lineNumber);
        }
        if (n2 in state.objectMasks) {
            aggr2 = false;
            mask2 = state.objectMasks[n2];
        } else if (n2 in state.aggregateMasks) {
            aggr2 = true;
            mask2 = state.aggregateMasks[n2];
        } else {
            logError('Unwelcome term "' + n1 + '" found in win condition. I don\'t know what I\'m supposed to do with this. ', lineNumber);
        }
        let newcondition = [num, mask1, mask2, lineNumber, aggr1, aggr2];
        newconditions.push(newcondition);
    }
    state.winconditions = newconditions;
}

function printCellRow(cellRow) {
    let result = "[ ";
    for (let i = 0; i < cellRow.length; i++) {
        if (i > 0) {
            result += "| ";
        }
        let cell = cellRow[i];
        for (let j = 0; j < cell.length; j += 2) {
            let direction = cell[j];
            let object = cell[j + 1]
            if (direction === "...") {
                result += direction + " ";
            } else {
                result += direction + " " + object + " ";
            }
        }
    }
    result += "] ";
    return result;
}

function cacheRuleStringRep(rule) {
    let result = "(<a onclick=\"jumpToLine('" + rule.lineNumber.toString() + "');\"  href=\"javascript:void(0);\">" + rule.lineNumber + "</a>) " + rule.direction.toString().toUpperCase() + " ";
    if (rule.rigid) {
        result = "RIGID " + result + " ";
    }
    if (rule.randomRule) {
        result = "RANDOM " + result + " ";
    }
    if (rule.late) {
        result = "LATE " + result + " ";
    }
    for (let i = 0; i < rule.lhs.length; i++) {
        let cellRow = rule.lhs[i];
        result = result + printCellRow(cellRow);
    }
    result = result + "-> ";
    for (let i = 0; i < rule.rhs.length; i++) {
        let cellRow = rule.rhs[i];
        result = result + printCellRow(cellRow);
    }
    for (let i = 0; i < rule.commands.length; i++) {
        let command = rule.commands[i];
        if (command.length === 1) {
            result = result + command[0].toString();
        } else {
            result = result + '(' + command[0].toString() + ", " + command[1].toString() + ') ';
        }
    }
    //print commands next
    rule.stringRep = result;
}

function cacheAllRuleNames(state) {

    for (let i = 0; i < state.rules.length; i++) {
        let rule = state.rules[i];
        cacheRuleStringRep(rule);
    }
}

function printRules(state) {
    let output = "";
    let loopIndex = -1;
    let outsideLoop = true
    let discardcount = 0;
    for (let i = 0; i < state.rules.length; i++) {
        let rule = state.rules[i];

        if (!outsideLoop) {
            //decide if we print ENDLOOP
            if (loopIndex + 1 < state.loops.length) {
                let nextLoop = state.loops[loopIndex + 1];
                if (nextLoop[0] < rule.lineNumber) {
                    output += "ENDLOOP<br>";
                    outsideLoop = true;
                    loopIndex++
                }
            }
        }
        // We *don't* have an else here because we might have 
        // two loops side-by-side.
        // ( cf  https://github.com/increpare/PuzzleScript/issues/1048 )
        if (outsideLoop) {

            // if there are multiple empty startloop/endloop pairs in a row,
            // we should skip past them
            // (e.g. https://www.puzzlescript.net/editor.html?hack=7e521a3c8d22f5dc5643ad5852f6cd22)
            if (loopIndex + 1 < state.loops.length) {
                let nextLoop = state.loops[loopIndex + 1];
                let loopEnd = state.loops[loopIndex + 2];
                while (loopIndex + 1 < state.loops.length && loopEnd[0] < rule.lineNumber) {
                    loopIndex += 2;
                    nextLoop = state.loops[loopIndex + 1];
                    loopEnd = state.loops[loopIndex + 2];
                }
            }

            //trying to decide if we print STARTLOOP
            if (loopIndex + 1 < state.loops.length) {
                let nextLoop = state.loops[loopIndex + 1];
                if (nextLoop[0] < rule.lineNumber) {
                    output += "STARTLOOP<br>";
                    outsideLoop = false;
                    loopIndex++;
                }
            }
        }

        if (rule.hasOwnProperty('discard')) {
            discardcount++;
        } else {
            let sameGroupAsPrevious = i > 0 && state.rules[i - 1].groupNumber === rule.groupNumber;
            if (sameGroupAsPrevious) {
                output += '+ ';
            } else {
                output += '&nbsp;&nbsp;';
            }
            output += rule.stringRep + "<br>";
        }
    }
    if (!outsideLoop) {
        output += "ENDLOOP<br>";
    }

    output += "===========<br>";
    output = "<br>Rule Assembly : (" + (state.rules.length - discardcount) + " rules)<br>===========<br>" + output;
    consolePrint(output);
}

function removeDuplicateRules(state) {
    let record = {};
    let newrules = [];
    let lastgroupnumber = -1;
    for (let i = state.rules.length - 1; i >= 0; i--) {
        let r = state.rules[i];
        let groupnumber = r.groupNumber;
        if (groupnumber !== lastgroupnumber) {
            record = {};
        }
        let r_string = r.stringRep;
        if (record.hasOwnProperty(r_string)) {
            state.rules.splice(i, 1);
        } else {
            record[r_string] = true;
        }
        lastgroupnumber = groupnumber;
    }
}

function calculateLoopPoints(state, rulegroup_collection) {
    let loopPoint = {};
    for (let j = 0; j < state.loops.length; j += 2) {
        let loop_start_line = state.loops[j][0]; //for each startloop/endloop
        let loop_end_line = state.loops[j + 1][0];
        let init_group_index = -1
        for (let group_i = 0; group_i < rulegroup_collection.length; group_i++) {
            let ruleGroup = rulegroup_collection[group_i];

            let firstRule = ruleGroup[0];
            let lastRule = ruleGroup[ruleGroup.length - 1];

            let firstRuleLine = firstRule.lineNumber;
            let lastRuleLine = lastRule.lineNumber;

            if (firstRuleLine <= loop_start_line && loop_start_line <= lastRuleLine) {
                logError("Found a loop point in the middle of a rule. You probably don't want to do this, right?", loop_start_line)
            } else if (firstRuleLine <= loop_end_line && loop_end_line <= lastRuleLine) {
                logError("Found a loop point in the middle of a rule. You probably don't want to do this, right?", loop_start_line)
            }

            let rule_before_loop = loop_start_line > firstRuleLine;
            let rule_in_loop = loop_start_line <= firstRuleLine && firstRuleLine <= loop_end_line;
            let rule_after_loop = loop_end_line < firstRuleLine;

            if (rule_after_loop)
                break;


            if (rule_in_loop) {
                if (init_group_index === -1) {
                    init_group_index = group_i
                }
                // only the last rulegroup in a loop should be the loop point - 
                // this is a bit lazy, but basically we look back, and if the 
                // previous rule-group has the same loop point, we remove it.
                if (group_i > 0 && loopPoint[group_i - 1] !== undefined && loopPoint[group_i - 1] === init_group_index) {
                    loopPoint[group_i - 1] = undefined;
                }
                loopPoint[group_i] = init_group_index;
            }
        }
    }
    return loopPoint;
}

function generateLoopPoints(state) {
    //run through to check loops aren't nested and are properly closed
    for (let group_i = 0; group_i < state.loops.length; group_i++) {
        let loop = state.loops[group_i];
        if (group_i % 2 === 0) {
            if (loop[1] === -1) {
                logError("Found an ENDLOOP, but I'm not in a loop?", loop[0]);
            }
        } else {
            if (loop[1] === 1) {
                logError("Found a STARTLOOP, but I'm already inside a loop? (Puzzlescript can't nest loops, FWIW).", loop[0]);
            }
        }
    }
    if ((state.loops.length % 2) !== 0) {
        logError("Yo I found a STARTLOOP without a corresponding ENDLOOP.", state.loops[state.loops.length - 1][0]);
    }


    state.loopPoint = calculateLoopPoints(state, state.rules);
    state.lateLoopPoint = calculateLoopPoints(state, state.lateRules);

}

let soundDirectionIndicatorMasks = {
    'up': parseInt('00001', 2),
    'down': parseInt('00010', 2),
    'left': parseInt('00100', 2),
    'right': parseInt('01000', 2),
    'horizontal': parseInt('01100', 2),
    'vertical': parseInt('00011', 2),
    'orthogonal': parseInt('01111', 2),
    '___action____': parseInt('10000', 2)
};

let soundDirectionIndicators = ["up", "down", "left", "right", "horizontal", "vertical", "orthogonal", "___action____"];


function generateSoundData(state) {
    let sfx_Events = {};
    let sfx_CreationMasks = [];
    let sfx_DestructionMasks = [];
    let sfx_MovementMasks = state.collisionLayers.map(x => []);
    let sfx_MovementFailureMasks = [];

    for (let i = 0; i < state.sounds.length; i++) {
        let sound = state.sounds[i];
        if (sound.length <= 1) {
            //don't see that this would ever be triggered
            continue;
        }
        let lineNumber = sound[sound.length - 1];

        if (sound.length === 2) {
            logWarning('incorrect sound declaration.', lineNumber);
            continue;
        }

        const v0 = sound[0][0].trim();
        const t0 = sound[0][1].trim();
        const v1 = sound[1][0].trim();
        const t1 = sound[1][1].trim();

        let seed = sound[sound.length - 2][0];
        let seed_t = sound[sound.length - 2][1];
        if (seed_t !== 'SOUND') {
            // unreachable?
            // seems to be pre-empted by "Was expecting a soundverb here 
            // (MOVE, DESTROY, CANTMOVE, or the like), but found something else" message
            logError("Expecting sfx data, instead found \"" + seed + "\".", lineNumber);
        }

        if (t0 === "SOUNDEVENT") {

            //pretty sure neither of the following are reachable, they're caught by the parser before.
            if (sound.length > 4) {
                logError("too much stuff to define a sound event.", lineNumber);
            } else {
                //out of an abundance of caution, doing a fallback warning rather than expanding the scope of the error #779
                if (sound.length > 3) {
                    logWarning("too much stuff to define a sound event.", lineNumber);
                }
            }

            if (sfx_Events[v0] !== undefined) {
                logWarning(v0.toUpperCase() + " already declared.", lineNumber);
            }
            sfx_Events[v0] = seed;

        } else {
            let target = v0;
            let verb = v1;
            let directions = [];
            for (let j = 2; j < sound.length - 2; j++) {//avoid last sound declaration as well as the linenumber element at the end
                if (sound[j][1] === 'DIRECTION') {
                    directions.push(sound[j][0]);
                } else {
                    //Don't think I can get here, but just in case
                    logError(`Expected a direction here, but found instead "$(sound[j][0])".`, lineNumber);
                }
            }
            if (directions.length > 0 && (verb !== 'move' && verb !== 'cantmove')) {
                //this is probably unreachable, as the parser catches it before it gets here
                logError('Incorrect sound declaration - cannot have directions (UP/DOWN/etc.) attached to non-directional sound verbs (CREATE is not directional, but MOVE is directional).', lineNumber);
            }

            if (verb === 'action') {
                verb = 'move';
                directions = ['___action____'];
            }

            if (directions.length === 0) {
                directions = ["orthogonal"];
            }


            if (target in state.aggregatesDict) {
                logError('cannot assign sound events to aggregate objects (declared with "and"), only to regular objects, or properties, things defined in terms of "or" ("' + target + '").', lineNumber);
            } else if (target in state.objectMasks) {

            } else {
                //probably unreachable
                logError('Object "' + target + '" not found.', lineNumber);
            }

            let objectMask = state.objectMasks[target];

            let directionMask = 0;
            for (let j = 0; j < directions.length; j++) {
                directions[j] = directions[j].trim();
                let direction = directions[j];
                if (soundDirectionIndicators.indexOf(direction) === -1) {
                    //pre-emted by parser
                    logError('Was expecting a direction, instead found "' + direction + '".', lineNumber);
                } else {
                    let soundDirectionMask = soundDirectionIndicatorMasks[direction];
                    directionMask |= soundDirectionMask;
                }
            }


            let targets = [target];
            let modified = true;
            while (modified) {
                modified = false;
                for (let k = 0; k < targets.length; k++) {
                    let t = targets[k];
                    if (t in state.synonymsDict) {
                        targets[k] = state.synonymsDict[t];
                        modified = true;
                    } else if (t in state.propertiesDict) {
                        modified = true;
                        let props = state.propertiesDict[t];
                        targets.splice(k, 1);
                        k--;
                        for (let l = 0; l < props.length; l++) {
                            targets.push(props[l]);
                        }
                    }
                }
            }

            //if verb in soundverbs_directional
            if (verb === 'move' || verb === 'cantmove') {
                for (let j = 0; j < targets.length; j++) {
                    let targetName = targets[j];
                    let targetDat = state.objects[targetName];
                    let targetLayer = targetDat.layer;
                    //if not found, continue - probably from the error ""aggr" is an aggregate (defined using "and"), and cannot be added to a single layer because its constituent objects must be able to coexist."
                    if (targetLayer === undefined) {
                        continue;
                    }
                    let shiftedDirectionMask = new BitVec(STRIDE_MOV);
                    shiftedDirectionMask.ishiftor(directionMask, 5 * targetLayer);

                    let o = {
                        objectMask: objectMask,
                        directionMask: shiftedDirectionMask,
                        layer: targetLayer,
                        seed: seed
                    };

                    if (verb === 'move') {
                        sfx_MovementMasks[targetLayer].push(o);
                    } else {
                        sfx_MovementFailureMasks.push(o);
                    }
                }
            }



            let targetArray;
            switch (verb) {
                case "create":
                    {
                        let o = {
                            objectMask: objectMask,
                            seed: seed
                        }
                        sfx_CreationMasks.push(o);
                        break;
                    }
                case "destroy":
                    {
                        let o = {
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


function formatHomePage(state) {
    if ('background_color' in state.metadata) {
        state.bgcolor = colorToHex(colorPalette, state.metadata.background_color);
    } else {
        state.bgcolor = "#000000";
    }
    if ('text_color' in state.metadata) {
        state.fgcolor = colorToHex(colorPalette, state.metadata.text_color);
    } else {
        state.fgcolor = "#FFFFFF";
    }

    if (isColor(state.fgcolor) === false) {
        logError("text_color in incorrect format - found " + state.fgcolor + ", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').  Defaulting to white.", state.metadata_lines.text_color)
        state.fgcolor = "#FFFFFF";
    }
    if (isColor(state.bgcolor) === false) {
        logError("background_color in incorrect format - found " + state.bgcolor + ", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').  Defaulting to black.", state.metadata_lines.background_color)
        state.bgcolor = "#000000";
    }

    if (canSetHTMLColors) {

        if ('background_color' in state.metadata) {
            document.body.style.backgroundColor = state.bgcolor;
        }

        if ('text_color' in state.metadata) {
            let separator = document.getElementById("separator");
            if (separator != null) {
                separator.style.color = state.fgcolor;
            }

            let aElements = document.getElementsByTagName("a");
            for (let i = 0; i < aElements.length; i++) {
                aElements[i].style.color = state.fgcolor;
            }

            let h1Elements = document.getElementsByTagName("h1");
            for (let i = 0; i < h1Elements.length; i++) {
                h1Elements[i].style.color = state.fgcolor;
            }
        }
    }

    if ('homepage' in state.metadata) {
        let url = state.metadata['homepage'];
        url = url.replace("http://", "");
        url = url.replace("https://", "");
        state.metadata['homepage'] = url;
    }
}

let MAX_ERRORS = 5;

function loadFile(str) {
    let processor = new codeMirrorFn();
    let state = processor.startState();

    let lines = str.split('\n');
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        state.lineNumber = i + 1;
        let ss = new CodeMirror.StringStream(line, 4);
        do {
            processor.token(ss, state);

            if (errorCount > MAX_ERRORS) {
                consolePrint("too many errors, aborting compilation");
                return;
            }
        }
        while (ss.eol() === false);
    }

    //check if player defined
    if (!isObjectDefined(state, "player")) {
        logErrorNoLine("Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!");
    }
    //check if background
    if (!isObjectDefined(state, "background")) {
        logErrorNoLine("Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!");
    }

    if (state.collisionLayers.length === 0) {
        logError("No collision layers defined.  All objects need to be in collision layers.");
        return null;
    }

    generateExtraMembers(state);
    generateMasks(state);
    levelsToArray(state);
    rulesToArray(state);
    if (state.invalid > 0) {
        return null;
    }

    cacheAllRuleNames(state);
    removeDuplicateRules(state);
    rulesToMask(state);

    if (debugMode) {
        printRules(state);
    }

    arrangeRulesByGroupNumber(state);
    collapseRules(state.rules);
    collapseRules(state.lateRules);

    generateRigidGroupList(state);

    processWinConditions(state);
    checkObjectsAreLayered(state);

    twiddleMetaData(state);

    generateLoopPoints(state);

    generateSoundData(state);

    formatHomePage(state);

    addSpecializedFunctions(state);

    //delete intermediate representations
    delete state.commentLevel;
    delete state.line_should_end;
    delete state.line_should_end_because;
    delete state.sol_after_comment;
    delete state.names;
    delete state.abbrevNames;
    delete state.objects_candname;
    delete state.objects_section;
    delete state.objects_spritematrix;
    delete state.section;
    delete state.subsection;
    delete state.tokenIndex;
    delete state.current_line_wip_array;
    delete state.visitedSections;
    delete state.loops;
    /*
    let lines = stripComments(str);
    window.console.log(lines);
    let sections = generateSections(lines);
    window.console.log(sections);
    let sss = generateSemiStructuredSections(sections);*/
    return state;
}


function addSpecializedFunctions(state) {
    const OBJECT_SIZE = Math.ceil(state.objectCount / 32);
    const MOVEMENT_SIZE = Math.ceil(state.collisionLayers.length / 5);
    state.moveEntitiesAtIndex = generate_moveEntitiesAtIndex(OBJECT_SIZE, MOVEMENT_SIZE);
    state.calculateRowColMasks = generate_calculateRowColMasks(OBJECT_SIZE, MOVEMENT_SIZE);
    state.resolveMovements = generate_resolveMovements(OBJECT_SIZE, MOVEMENT_SIZE);
    state.matchCellRow = generateMatchCellRow(OBJECT_SIZE, MOVEMENT_SIZE);
    state.matchCellRowWildCard = generateMatchCellRowWildCard(OBJECT_SIZE, MOVEMENT_SIZE);
}

function compile(command, text, randomseed) {
    forceRegenImages = true;
    if (command === undefined) {
        command = ["restart"];
    }
    if (randomseed === undefined) {
        randomseed = null;
    }
    lastDownTarget = canvas;

    if (text === undefined) {
        let code = window.form1.code;

        let editor = code.editorreference;

        text = editor.getValue() + "\n";
    }
    if (canDump === true) {
        compiledText = text;
    }

    errorCount = 0;
    compiling = true;
    errorStrings = [];
    consolePrint('=================================');
    let state;
    try {
        state = loadFile(text);
    } catch (error) {
        consolePrint(error);
        console.log(error);
    } finally {
        compiling = false;
    }

    if (state && state.levels && state.levels.length === 0) {
        logError('No levels found.  Add some levels!', undefined, true);
    }



    if (errorCount > 0) {
        if (IDE === false) {
            if (state === null) {
                consoleError('<span class="systemMessage">Errors detected during compilation; I can\'t salvage anything playable from it.  If this is an older game, and you think it just broke because of recent changes in the puzzlescript engine, please consider dropping an email to analytic@gmail.com with a link to the game and I\'ll try make sure it\'s back working ASAP.</span>');
            } else {
                consoleError('<span class="systemMessage">Errors detected during compilation; the game may not work correctly. If this is an older game, and you think it just broke because of recent changes in the puzzlescript engine, please consider dropping an email to analytic@gmail.com with a link to the game and I\'ll try make sure it\'s back working ASAP.</span>');
            }
        } else {
            if (state === null) {
                consoleError('<span class="systemMessage">Errors detected during compilation; I can\'t salvage anything playable from it.</span>');
            } else {
                consoleError('<span class="systemMessage">Errors detected during compilation; the game may not work correctly.</span>');
            }
        }
        if (errorCount > MAX_ERRORS) {
            return;
        }
    } else {
        let ruleCount = 0;
        for (let i = 0; i < state.rules.length; i++) {
            ruleCount += state.rules[i].length;
        }
        for (let i = 0; i < state.lateRules.length; i++) {
            ruleCount += state.lateRules[i].length;
        }
        if (command[0] === "restart") {
            consolePrint('<span class="systemMessage">Successful Compilation, generated ' + ruleCount + ' instructions.</span>');
        } else {
            consolePrint('<span class="systemMessage">Successful live recompilation, generated ' + ruleCount + ' instructions.</span>');

        }



        if (IDE) {
            if (state.metadata.title !== undefined) {
                document.title = "PuzzleScript - " + state.metadata.title;
            }
        }
    }

    if (state !== null) {//otherwise error
        setGameState(state, command, randomseed);
    }

    clearInputHistory();

    consoleCacheDump();

    manage_compilation_caches();

}

const cache_CHECK_RATE=20;
let cache_checkCount=0;
function manage_compilation_caches() {    
    cache_checkCount = (cache_checkCount + 1) % cache_CHECK_RATE;
    if (cache_checkCount !== 0) {
        return;
    }
    // console.log("CACHE_CELLPATTERN_MATCHFUNCTION size: " + Object.keys(CACHE_CELLPATTERN_MATCHFUNCTION).length);
    // console.log("CACHE_MOVEENTITIESATINDEX size: " + Object.keys(CACHE_MOVEENTITIESATINDEX).length);
    // console.log("CACHE_CALCULATEROWCOLMASKS size: " + Object.keys(CACHE_CALCULATEROWCOLMASKS).length);
    // console.log("CACHE_RULE_CELLROWMATCHESFUNCTION size: " + Object.keys(CACHE_RULE_CELLROWMATCHESFUNCTION).length);
    // console.log("CACHE_CELLPATTERN_REPLACEFUNCTION size: " + Object.keys(CACHE_CELLPATTERN_REPLACEFUNCTION).length);
    // console.log("CACHE_MATCHCELLROW size: " + Object.keys(CACHE_MATCHCELLROW).length);
    // console.log("CACHE_MATCHCELLROWWILDCARD size: " + Object.keys(CACHE_MATCHCELLROWWILDCARD).length);
    // console.log("CACHE_RULE_APPLYAT size: " + Object.keys(CACHE_RULE_APPLYAT).length);
    // console.log("CACHE_RESOLVEMOVEMENTS size: " + Object.keys(CACHE_RESOLVEMOVEMENTS).length);
    // console.log("CACHE_RULE_FINDMATCHES size: " + Object.keys(CACHE_RULE_FINDMATCHES).length);



    //CACHE_CELLPATTERN_MATCHFUNCTION>10000, reset, for the others the limit is say 200
    if (Object.keys(CACHE_CELLPATTERN_MATCHFUNCTION).length > 10000) {
        CACHE_CELLPATTERN_MATCHFUNCTION = {};
    }
    if (Object.keys(CACHE_MOVEENTITIESATINDEX).length > 200) {
        CACHE_MOVEENTITIESATINDEX = {};
    }
    if (Object.keys(CACHE_CALCULATEROWCOLMASKS).length > 200) {
        CACHE_CALCULATEROWCOLMASKS = {};
    }
    if (Object.keys(CACHE_RULE_CELLROWMATCHESFUNCTION).length > 1000) {
        CACHE_RULE_CELLROWMATCHESFUNCTION = {};
    }
    if (Object.keys(CACHE_CELLPATTERN_REPLACEFUNCTION).length > 200) {
        CACHE_CELLPATTERN_REPLACEFUNCTION = {};
    }
    if (Object.keys(CACHE_MATCHCELLROW).length > 200) {
        CACHE_MATCHCELLROW = {};
    }
    if (Object.keys(CACHE_MATCHCELLROWWILDCARD).length > 200) {
        CACHE_MATCHCELLROWWILDCARD = {};
    }
    if (Object.keys(CACHE_RULE_APPLYAT).length > 200) {
        CACHE_RULE_APPLYAT = {};
    }
    if (Object.keys(CACHE_RESOLVEMOVEMENTS).length > 200) {
        CACHE_RESOLVEMOVEMENTS = {};
    }
    if (Object.keys(CACHE_RULE_FINDMATCHES).length > 200) {
        CACHE_RULE_FINDMATCHES = {};
    }
}



function qualifyURL(url) {
    let a = document.createElement('a');
    a.href = url;
    return a.href;
}


