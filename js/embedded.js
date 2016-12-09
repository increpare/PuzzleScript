var DIR_UP     = 0b00001;
var DIR_DOWN   = 0b00010;
var DIR_LEFT   = 0b00100;
var DIR_RIGHT  = 0b01000;
var DIR_ACTION = 0b10000;
var LAYER_COUNT = 3;

String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.replace(new RegExp(search, 'g'), replacement);
};

function ARDU_spriteGlyphs(){
    var strb=`PROGMEM const byte tiles_b[][8] = {\n`;
    var strw=`PROGMEM const byte tiles_w[][8] = {\n`;
    for (var i = 0; i < sprites.length; i++) {
        strb+="\t{\n"
        strw+="\t{\n"
        if (sprites[i] == undefined) {
            continue;
        }

        var s = sprites[i];

        for (var j=0;j<8;j++){
            var black="";
            var white="";
            for (var k=0;k<8;k++){
                var p = s.dat[k][j]
                var c = (p==-1||p==".")?".":s.colors[ p ];                
                c = c.toLowerCase();
                if (c==="black"||c==="#000000"){
                    black="1"+black;
                } else {
                    black="0"+black;
                }
                if (c==="white"||c==="#ffffff"){
                    white="1"+white;
                } else {
                    white="0"+white;
                }
            }
            strb+="\t\t0b"+black+",\n"
            strw+="\t\t0b"+white+",\n"
        }
        strb+="\t},\n"
        strw+="\t},\n"
    }
    strb+=`};\n`
    strw+=`};\n`
    return `const int GLYPH_COUNT = ` +sprites.length+`;\n\n`+strb+'\n'+strw;
}

function ARDU_levelDat(){
	var levelDat = `PROGMEM const byte levels[][128] {\n`
	for (var i=0;i<state.levels.length;i++){
		var level=state.levels[i]
		levelDat+="\t{\n\t\t"
        var levelw=level.width;
        var levelh=level.height;
        if (levelw>16){
            consoleError(`ARDUBOY ERROR:\nOne of your levels has width ${levelw} (max on arduboy is 16).`)
        }
        if (levelh>8){
            consoleError(`ARDUBOY ERROR:\nOne of your levels has height ${levelh} (max on arduboy is 8).`)
        }

        var centeredLevelGrid=[];
        for (var j=0;j<8;j++){
            for (var k=0;k<16;k++){
                centeredLevelGrid.push(0);
            }
        }
        var xpadding=Math.floor((16-levelw)/2);
        var ypadding=Math.floor((8-levelh)/2);
        for (var j=0;j<levelh;j++){            
            for (var k=0;k<levelw;k++){
                var fromidx = j+levelh*k;
                var toidx = j+ypadding+8*(k+xpadding);
                centeredLevelGrid[toidx]=level.objects[fromidx]
            }
        }

		for (var j=0;j<8;j++){            
			for (var k=0;k<16;k++){
				var idx = j+8*k
				levelDat+=centeredLevelGrid[idx]+","
			}
			levelDat+="\n"
			if (j<7){
				levelDat+="\t\t"
			}
		}
		levelDat+="\t},\n"
	}
	levelDat+="};\n"
	return levelDat
}

function printInt(n,bits){
    var result = n.toString(2);
    while(result.length<bits){
        result = '0'+result;;
    }
    return '0b'+result;
}

function ARDU_playerConstants(){
    var result="";

    result+= "const byte PLAYER_MASK = "+printInt(state.playerMask.data[0],8)+";\n";
    
    var playerLayers = getLayersOfMask(state.playerMask)
    var playerLayerMask = 0;
    for (var i=0;i<playerLayers.length;i++){
        var l = playerLayers.length;
        playerLayerMask+=0b11111<<(5*playerLayers[0]);
    }
    result += "const word PLAYER_LAYERMASK = "+printInt(playerLayerMask,16)+";\n";


    result+="\n";
    result += "const word LAYERMASK[] = {\n"
    for (var i=0;i<state.collisionLayers.length;i++){
        var clayer = state.collisionLayers[i];
        var lMask = 0;

        for (var n=0;n<state.objectCount;n++){
            var obN=state.idDict[n];
            if (clayer.indexOf(obN)>=0){
                lMask |= (1<<n);
            }
        }
        result+="\t"+printInt(lMask,16)+",\n";

    }
    result += "};\n";

    return result;
}


function GenerateEllipsisMatchPattern(d,p,depth,flip=false){
    if (flip===true){
        p = p.slice().reverse();
    }
    var ellipsisIndex = getEllipsisIndex(p);

    var tests="";
    for (var l=0;l<ellipsisIndex;l++){
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];

        var _cellObjects = `level[i_L_${depth}+${l*d}]`
        var _cellMovements = `movementMask[i_L_${depth}+${l*d}]`
        if (tests.length>0){
            tests+=" && ";
        }
        if (objectPresent!==0){            
            if (tests.length>0){
                tests+=" && ";
            }
            tests+=`( ${_cellObjects} & ${objectPresent} )`;
            if (movementPresent!==0){
                tests+=` && ( ${_cellMovements} & ${movementPresent})`;                            
            }
        }
        for (var i=0;i<anyObjectsPresent.length;i++){
            var op = anyObjectsPresent[i].data[0];
            if (op!==0){
                if (tests.length>0){
                    tests+=" && ";
                }
                tests+=`( ${_cellObjects} & 0b${op.toString(2)} )`;
            }
        }
        if (movementMissing!==0){
            if (tests.length>0){
                tests+=" && ";
            }
            tests += `( !( ${_cellMovements}&${movementMissing} ) )`;
        }
    }

    var l0=ellipsisIndex+1;
    for (var l=l0;l<p.length;l++){
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];

        var _cellObjects = `level[i_R_${depth}+${(l-l0)*d}]`
        var _cellMovements = `movementMask[i_R_${depth}+${(l-l0)*d}]`
        if (objectPresent!==0){            
            if (tests.length>0){
                tests+=" && ";
            }
            tests+=`( ${_cellObjects} & ${objectPresent} )`;
            if (movementPresent!==0){
                tests+=` && ( ${_cellMovements} & ${movementPresent})`;                            
            }
        }
        if (objectMissing!==0){
            if (tests.length>0){
                tests+=" && ";
            }
            tests+=`!( ${_cellObjects} & ${objectMissing} )`;
        }
        for (var i=0;i<anyObjectsPresent.length;i++){
            var op = anyObjectsPresent[i].data[0];
            if (op!==0){
                if (tests.length>0){
                    tests+=" && ";
                }
                tests+=`( ${_cellObjects} & 0b${op.toString(2)} )`;
            }
        }
        if (movementMissing!==0){
            if (tests.length>0){
                tests+=" && ";
            }
            tests += `( !( ${_cellMovements}&${movementMissing} ) )`;
        }
    }
    return tests;
}

function GenerateMatchPattern(d,p,depth){
    var tests="";
    for (var l=0;l<p.length;l++){
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];
        var anyObjectsPresent=c.anyObjectsPresent;
        var movementMissing=c.movementsMissing.data[0];

        var _cellObjects = `level[i${depth}+`+l*d+`]`
        var _cellMovements = `movementMask[i${depth}+`+l*d+`]`
        if (objectPresent!==0){            
            if (tests.length>0){
                tests+=" && ";
            }
            tests+=`( ${_cellObjects} & ${objectPresent} )`;
            if (movementPresent!==0){
                tests+=` && ( ${_cellMovements} & ${movementPresent})`;                            
            }
        }
        if (objectMissing!==0){
            if (tests.length>0){
                tests+=" && ";
            }
            tests+=`!( ${_cellObjects} & ${objectMissing} )`;
        }
        for (var i=0;i<anyObjectsPresent.length;i++){
            var op = anyObjectsPresent[i].data[0];
            if (op!==0){
                if (tests.length>0){
                    tests+=" && ";
                }
                tests+=`( ${_cellObjects} & 0b${op.toString(2)} )`;
            }
        }
        if (movementMissing!==0){
            if (tests.length>0){
                tests+=" && ";
            }
            tests += `( !( ${_cellMovements}&${movementMissing} ) )`;
        }
    }
    return tests;
}


function GenerateEllipsisPatternReplacement(d,p,depth,flip){
    if (flip===true){
        p = p.slice().reverse();
    }
    var ellipsisIndex = getEllipsisIndex(p);
    var movementsMissing = [];
    var movementsPresent = [];
    var objectsMissing = [];
    var objectsPresent = [];
    var test="";
    for (var l=0;l<ellipsisIndex;l++){
        if (l>0){
            test+="\n";
        }
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];

        var objectsClear=c.replacement.objectsClear.data[0];
        var objectsSet=c.replacement.objectsSet.data[0];
        var movementsClear=c.replacement.movementsClear.data[0];
        var movementsLayerMask=c.replacement.movementsLayerMask.data[0];
        var movementsSet=c.replacement.movementsSet.data[0];
        var randomDirMask=c.replacement.randomDirMask.data[0];
        var randomEntityMask=c.replacement.randomEntityMask.data[0];


        var objectsPreserve=(~objectsClear)>>> 0;
        var movementsPresent=(~movementsClear)>>> 0;

        var lvlName = `level[i_L_${depth}+`+l*d+`]`
        test += lvlName + " = "
        test += `(${lvlName}&${objectsPreserve})|${objectsSet};\n`
        var mvmtName = `movementMask[i_L_${depth}+`+l*d+`]`
        test += mvmtName + " = "
        test += `(${mvmtName}&${movementsPresent})|${movementsSet};`


        if (randomEntityMask!==0){
            test+=`\n`;

            var entityCount=CountBits(randomEntityMask);
            test+=`switch (random(0,${entityCount})){\n`
            var ni=0;
            for (var i=0;i<entityCount;i++){
                var targetObjMask = 1<<GetPositionofKthOne(randomEntityMask,i);
                test+=`    case ${i}:\n`
                test+=`    {\n`
                test+=`        level[i${depth}+`+l*d+`] |= 0b${targetObjMask.toString(2)};\n`
                test+=`        break;\n`
                test+=`    }\n`
                ni++;
            }
            test+=`}`
        }
        if (randomDirMask!==0){
            test+=`\n`;

            for (var i=0;i<LAYER_COUNT;i++){
                if ((randomDirMask&(1<<i)) !=0){
                    test+=`movementMask[i_L_${depth}+`+l*d+`] |= (1<<random(0,4))<<${5*i};`;
                }
            }
        }
    }
    var l0=ellipsisIndex+1;
    for (var l=l0;l<p.length;l++){
        if (l>0){
            test+="\n";
        }
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];

        var objectsClear=c.replacement.objectsClear.data[0];
        var objectsSet=c.replacement.objectsSet.data[0];
        var movementsClear=c.replacement.movementsClear.data[0];
        var movementsLayerMask=c.replacement.movementsLayerMask.data[0];
        var movementsSet=c.replacement.movementsSet.data[0];
        var randomDirMask=c.replacement.randomDirMask.data[0];
        var randomEntityMask=c.replacement.randomEntityMask.data[0];


        var objectsPreserve=(~objectsClear)>>> 0;
        var movementsPresent=(~movementsClear)>>> 0;

        var lvlName = `level[i_R_${depth}+`+(l-l0)*d+`]`
        test += lvlName + " = "
        test += `(${lvlName}&${objectsPreserve})|${objectsSet};\n`
        var mvmtName = `movementMask[i_R_${depth}+`+(l-l0)*d+`]`
        test += mvmtName + " = "
        test += `(${mvmtName}&${movementsPresent})|${movementsSet};`


        if (randomEntityMask!==0){
            test+=`\n`;

            var entityCount=CountBits(randomEntityMask);
            test+=`switch (random(0,${entityCount})){\n`
            var ni=0;
            for (var i=0;i<entityCount;i++){
                var targetObjMask = 1<<GetPositionofKthOne(randomEntityMask,i);
                test+=`    case ${i}:\n`
                test+=`    {\n`
                test+=`        level[i${depth}+`+l*d+`] |= 0b${targetObjMask.toString(2)};\n`
                test+=`        break;\n`
                test+=`    }\n`
                ni++;
            }
            test+=`}`
        }
        if (randomDirMask!==0){
            test+=`\n`;

            for (var i=0;i<LAYER_COUNT;i++){
                if ((randomDirMask&(1<<(5*i))) !=0){
                    test+=`movementMask[i_R_${depth}+`+l*d+`] |= (1<<random(0,4))<<${5*i};`;
                }
            }
        }
    }

    return test;
}

function CountBits(n){
    var result=0;
    for (var i=0;i<state.objectCount;i++){
        if ((n&(1<<i))!==0){
            result++;
        }
    }
    return result;
}

function GetPositionofKthOne(n,k){
    var count=0;
    for (var i=0;i<state.objectCount;i++){        
        if ((n&(1<<i))!==0){
            if (k===count){
                return i;
            }
            count++;
        }
    }
    return -1;
}

function GeneratePatternReplacement(d,p,depth){
    var movementsMissing = [];
    var movementsPresent = [];
    var objectsMissing = [];
    var objectsPresent = [];
    var test="";
    for (var l=0;l<p.length;l++){
        if (l>0){
            test+="\n";
        }
        var c = p[l]
        var movementMissing=c.movementsMissing.data[0];
        var movementPresent=c.movementsPresent.data[0];
        var objectMissing=c.objectsMissing.data[0];
        var objectPresent=c.objectsPresent.data[0];

        var objectsClear=c.replacement.objectsClear.data[0];
        var objectsSet=c.replacement.objectsSet.data[0];
        var movementsClear=c.replacement.movementsClear.data[0];
        var movementsLayerMask=c.replacement.movementsLayerMask.data[0];
        var movementsSet=c.replacement.movementsSet.data[0];
        var randomEntityMask=c.replacement.randomEntityMask.data[0];
        var randomDirMask=c.replacement.randomDirMask.data[0];


        var objectsPreserve=(~objectsClear)>>> 0;
        var movementsPresent=(~movementsClear)>>> 0;

        var lvlName = `level[i${depth}+`+l*d+`]`
        test += lvlName + " = "
        test += `(${lvlName}&${objectsPreserve})|${objectsSet};\n`
        var mvmtName = `movementMask[i${depth}+`+l*d+`]`
        test += mvmtName + " = "
        test += `( ${mvmtName}&${movementsPresent} ) | ${movementsSet};`


        var randomEntityMask=c.replacement.randomEntityMask.data[0];
        var randomDirMask=c.replacement.randomDirMask.data[0];

        if (randomEntityMask!==0){
            test+=`\n`;

            var entityCount=CountBits(randomEntityMask);
            test+=`switch (random(0,${entityCount})){\n`
            var ni=0;
            for (var i=0;i<entityCount;i++){
                var targetObjMask = 1<<GetPositionofKthOne(randomEntityMask,i);
                test+=`    case ${i}:\n`
                test+=`    {\n`
                test+=`        level[i${depth}+`+l*d+`] |= 0b${targetObjMask.toString(2)};\n`
                test+=`        break;\n`
                test+=`    }\n`
                ni++;
            }
            test+=`}`
        }
        if (randomDirMask!==0){
            test+=`\n`;

            for (var i=0;i<LAYER_COUNT;i++){
                if ((randomDirMask&(1<<(5*i))) !=0){
                    test+=`movementMask[i${depth}+`+l*d+`] |= (1<<random(0,4))<<${5*i};`;
                }
            }
        }
    }
    return test;
}

function indent(s,amount){
    for (var i=0;i<amount;i++){
        s=s.replace(/^/gm, "    ");
    }
    return s;
}

function getEllipsisIndex(pattern){
    for (var i=0;i<pattern.length;i++){
        var p = pattern[i];
        if (p.length===1&&p[0]===ellipsisPattern[0]){
            return i;
        }
    }
    return -1;
}


function generateEllipsisMatchString(pattern,ruleDir,depth){
    var ellipsisIndex = getEllipsisIndex(pattern);

    var pat1_startIndex = 0;
    var pat1_length = ellipsisIndex;

    var pat2_startIndex = ellipsisIndex+1;
    var pat2_length = pattern.length-pat2_startIndex;
    
    var d=0;
    var flip=false;
    switch (ruleDir){
        case DIR_UP:
        {
            d=16;
            flip=true;
            break;
        }
        case DIR_DOWN:
        {
            d=16;
            break;
        }
        case DIR_LEFT:
        {
            d=1;
            flip=true;
            break;
        }
        case DIR_RIGHT:
        {
            d=1;
            break;
        }
    }

    var test = GenerateEllipsisMatchPattern(d,pattern,depth,flip)
    var replacement = GenerateEllipsisPatternReplacement(d,pattern,depth,flip)
    var l = pattern.length;
    var maxY=8;
    var maxX=16;
    if (d===1){
        maxX-=l-2;//-2 because of ellipses
    } else {
        maxY-=l-2;                    
    }

    var patternTest="";

    var patternLoop = ""
    
    switch(ruleDir){        
        case DIR_RIGHT: {
            patternLoop+=`for (byte y${depth}=0;y${depth}<${maxY};y${depth}++){\n`
            patternLoop+=`    for (byte x${depth}=0;x${depth}<${maxX};x${depth}++){\n`
            patternLoop+=`        for (byte k${depth}=0;k${depth}<(${maxX}-x${depth});k${depth}++){`

            patternTest += `byte i_L_${depth} = x${depth}+16*y${depth};\n`
            patternTest += `byte i_R_${depth} = x${depth}+k${depth}+${ellipsisIndex}+16*y${depth};\n`
            patternTest += `if (${test}){`
            break;
        }
        case DIR_LEFT: {
            patternLoop+=`for (byte y${depth}=0;y${depth}<${maxY};y${depth}++){\n`
            patternLoop+=`    for (byte x${depth}=0;x${depth}<${maxX};x${depth}++){\n`
            patternLoop+=`        for (char k${depth}=x${depth};k${depth}>=0;k${depth}--){`

            patternTest +=`byte i_L_${depth} = k${depth}+16*y${depth};\n`
            patternTest +=`byte i_R_${depth} = x${depth}+${ellipsisIndex}+16*y${depth};\n`;
            patternTest +=`if (${test}){`
            break;
        }
        case DIR_DOWN:{
            patternLoop+=`for (byte y${depth}=0;y${depth}<${maxY};y${depth}++){\n`
            patternLoop+=`    for (byte x${depth}=0;x${depth}<${maxX};x${depth}++){\n`
            patternLoop+=`        for (byte k${depth}=0;k${depth}<(${maxY}-y${depth});k${depth}++){`  

            patternTest += `byte i_L_${depth} = x${depth}+16*y${depth};\n`
            patternTest += `byte i_R_${depth} = x${depth}+16*(y${depth}+k${depth}+${ellipsisIndex});\n`
            patternTest += `if (${test}){`     
            break;
        }
        case DIR_UP:{
            patternLoop+=`for (byte y${depth}=0;y${depth}<${maxY};y${depth}++){\n`
            patternLoop+=`    for (byte x${depth}=0;x${depth}<${maxX};x${depth}++){\n`
            patternLoop+=`        for (char k${depth}=y${depth};k${depth}>=0;k${depth}--){`

            patternTest += `byte i_L_${depth} = x${depth}+16*k${depth};\n`
            patternTest += `byte i_R_${depth} = x${depth}+16*(y${depth}+${ellipsisIndex});\n`
            patternTest += `if (${test}){`     
            break;
        }
    }

var patternEnd = 
`            }
        }
    }
}`;
    var patternReplace=replacement;

    return [patternLoop,patternTest,patternReplace,patternEnd];
}

function generateMatchString(pattern,ruleDir,depth){
    if (getEllipsisIndex(pattern)>=0){
        return generateEllipsisMatchString(pattern,ruleDir,depth)
    }
    var d = ruleDir===8?1:16;//right is 8, otherwise down

    var test = GenerateMatchPattern(d,pattern,depth)
    var replacement = GeneratePatternReplacement(d,pattern,depth)
    var l = pattern.length;
    var maxY=8;
    var maxX=16;
    if (d===1){
        maxX-=l-1;
    } else {
        maxY-=l-1;                    
    }

    var patternLoop = 
`for (byte y${depth}=0;y${depth}<${maxY};y${depth}++){
    for (byte x${depth}=0;x${depth}<${maxX};x${depth}++){`

var patternTest = 
`byte i${depth} = x${depth}+16*y${depth};
if (${test}){`

var patternEnd = 
`        }
    }
}`;
    var patternReplace=replacement;

    return [patternLoop,patternTest,patternReplace,patternEnd];
}

function ARDU_rulesDat(){
    var result="";
    var rulesTogether=state.rules.concat(state.lateRules);

    for (var i=0;i<rulesTogether.length;i++){
        var rg = rulesTogether[i];
        var late = i>=state.rules.length;
        for (var j=0;j<rg.length;j++){
            var r = rg[j];
            var ruleDir = r.direction;
                console.log("DIR " +ruleDir);
            var matchStrings=[];            
            for (var k=0;k<r.patterns.length;k++){   
                var pattern = r.patterns[k];
                matchStrings.push(generateMatchString(pattern,ruleDir,k));
            }

            var depth=1;           
            if (!late){ 
                result+=`bool applyRule${i}_${j}(){\n`;
            } else {
                result+=`bool applyLateRule${i-state.rules.length}_${j}(){\n`;                
            }
            for (var k=0;k<matchStrings.length;k++){
                var [loop,test,replace,post]=matchStrings[k];
                loop=indent(loop,depth);
                result+=loop+"\n";
                depth+=2;
                if (getEllipsisIndex(r.patterns[k])>=0){
                    depth++;
                }
            }
            for (var k=0;k<matchStrings.length;k++){
                var [loop,test,replace,post]=matchStrings[k];
                test=indent(test,depth);
                result+=test+"\n";
                depth++;
            }
            for (var k=0;k<matchStrings.length;k++){
                var [loop,test,replace,post]=matchStrings[k];
                replace=indent(replace,depth)+"\n"
                result+=replace;
            }
            while (depth>0){
                depth--;
                result+=indent("}",depth)+"\n";
            }
        }
    }
    console.log(result);
    return result;
}

function ARDU_applyRulesFns(){
    var result=`void processRules(){\n`;

    for (var i=0;i<state.rules.length;i++){
        var rg = state.rules[i];
        for (var j=0;j<rg.length;j++){
            result +=`\tapplyRule${i}_${j}();\n`
        }
    }

    result+="}\n"

    result+=`void processLateRules(){\n`;

    for (var i=0;i<state.lateRules.length;i++){
        var rg = state.lateRules[i];
        for (var j=0;j<rg.length;j++){
            result +=`\tapplyLateRule${i}_${j}();\n`
        }
    }

    result+="}\n"
    return result;
}

function ARDU_titleFunction(){
    var CHAR_WIDTH=6;
    var SCREEN_WIDTH=128;
    var SCREEN_HEIGHT=64;
    var LINEHEIGHT=10;

    function strOffset(s){
        if (s.length>21){
            consoleError(`ARDUBOY ERROR:\nstring "${s}" is too long (${s.length} characters). Max string length for arduboy is 21.` )
        }
      return Math.floor(SCREEN_WIDTH/2-(CHAR_WIDTH*s.length/2));
    }

    if (state.metadata.title!==undefined) {
        var s_title = state.metadata.title;
        var x_title = strOffset(s_title);
    }

    if (state.metadata.author!==undefined) {
        var s_by = `by ${state.metadata.author}`;
        var x_by = strOffset(s_by);
    }

    var s_start = "start game";
    var x_start = strOffset(s_start);

    var s_startSelected = "> start game <"
    var x_startSelected = strOffset(s_startSelected);

    var s_new = "new game";
    var x_new = strOffset(s_new);

    var s_newSelected = "> new game <"
    var x_newSelected = strOffset(s_newSelected);

    var s_continue = "continue game"
    var x_continue = strOffset(s_continue);

    var s_continueSelected = "> continue game <"
    var x_continueSelected = strOffset(s_continueSelected);

    var outputTxt = `
        byte titleSelection = 2;

        void drawTitle(){

          arduboy.setCursor(${x_title}, 0);
          arduboy.print(F("${s_title}"));
          

          arduboy.setCursor(${x_by}, ${LINEHEIGHT});
          arduboy.print(F("${s_by}"));

          switch (titleSelection){
            case 2:{
              arduboy.setCursor(${x_startSelected}, ${LINEHEIGHT*3});
              arduboy.print(F("${s_startSelected}"));
              break;
            }
            case 0:{
              arduboy.setCursor(${x_newSelected}, ${LINEHEIGHT*3-4});
              arduboy.print(F("${s_newSelected}"));
              arduboy.setCursor(${x_continue}, ${LINEHEIGHT*3-4+8});
              arduboy.print(F("${s_continue}"));
              break;
            }
            case 1:{
              arduboy.setCursor(${x_new}, ${LINEHEIGHT*3-4});
              arduboy.print(F("${s_new}"));
              arduboy.setCursor(${x_continueSelected}, ${LINEHEIGHT*3-4+8});
              arduboy.print(F("${s_continueSelected}"));
              break;
            }
          }

          arduboy.setCursor(0,64-15);
          arduboy.print(F("A:reset, B:action\\nA+B:restart")); 
          arduboy.display(true);
        }
        `;
        return outputTxt;
}

function ARDU_winConditionsDat(){
    if (state.winconditions.length===0){
        return "void checkWin(){}"
    }

    var outputTxt=`void checkWin(){\n`;


    for (var i=0;i<state.winconditions.length;i++){
        outputTxt+="\t{\n"
        var wc = state.winconditions[i];
        switch (wc[0]){
            case -1:{//NO
                outputTxt+=`
        for (byte i=0;i<128;i++){
            if ( !(level[i]&${wc[1].data[0]}) && !(level[i]&${wc[2].data[0]}) ){
                return;
            }
        }\n`;

                break;
            }
            case 0:{//SOME
                outputTxt+=`
        bool passedTest=false;
        for (byte i=0;i<128;i++){
            if ( !(level[i]&${wc[1].data[0]}) && !(level[i]&${wc[2].data[0]}) ){
                passedTest=true;
                break;
            }
            if (!passedTest){
                return;
            }
        }\n`;

                break;
            }
            case 1:{//ALL
                outputTxt+=`
        for (byte i=0;i<128;i++){
            if ( !(level[i]&${wc[1].data[0]}) && (level[i]&${wc[2].data[0]}) ){
                return;
            }
        }\n`;
                break;
            }
        }
        outputTxt+="\t}\n"    
    }

    outputTxt+=`
  waiting=true;
  waitfrom=millis();
}`;
    return outputTxt;
}

function exportEmbeddedClick(){


	var sourceCode = editor.getValue();

	compile("restart");
    
    var outputTxt=`
enum State {
  LEVEL,
  TITLE,
  MESSAGE
};

const byte LAYER_COUNT = 3;

State state=TITLE;
const byte DIR_UP     = 0b00001;
const byte DIR_DOWN   = 0b00010;
const byte DIR_LEFT   = 0b00100;
const byte DIR_RIGHT  = 0b01000;
const byte DIR_ACTION = 0b10000;

const word ALL_UP = DIR_UP+(DIR_UP<<5)+(DIR_UP<<10);
const word ALL_DOWN = DIR_DOWN+(DIR_DOWN<<5)+(DIR_DOWN<<10);
const word ALL_LEFT = DIR_LEFT+(DIR_LEFT<<5)+(DIR_LEFT<<10);
const word ALL_RIGHT = DIR_RIGHT+(DIR_RIGHT<<5)+(DIR_RIGHT<<10);
const word ALL_ACTION = DIR_ACTION+(DIR_ACTION<<5)+(DIR_ACTION<<10);

byte undoState[128];
byte level[128];
word movementMask[128];
byte rowCellContents[8];
byte colCellContents[16];
byte mapCellContents=0;
unsigned long waitfrom;
bool waiting=false;
`;

    var playerConsts = ARDU_playerConstants();
    outputTxt+=playerConsts;
    
    var titleFunction = ARDU_titleFunction();
    outputTxt+=titleFunction+"\n";

	var glyphText = ARDU_spriteGlyphs();
	outputTxt+=glyphText+"\n";
    
    var levelText = ARDU_levelDat();
    outputTxt+=levelText+"\n";
    
    var rulesText = ARDU_rulesDat();
    outputTxt+=rulesText+"\n";
    
    var applyRulesFns = ARDU_applyRulesFns();
    outputTxt+=applyRulesFns+"\n";
    
    var winConditionsText = ARDU_winConditionsDat();
    outputTxt+=winConditionsText+"\n";


    var title = "PuzzleScript";
    if (state.metadata.title!==undefined) {
        title=state.metadata.title;
    }


    var BB = get_blob();
    var blob = new BB([outputTxt], {type: "text/plain;charset=utf-8"});
    saveAs(blob, title+".ino");

}