/*
..................................
.............SOKOBAN..............
..................................
...........#.new game.#...........
..................................
.............continue.............
..................................
arrow keys to move................
x to action.......................
z to undo, r to restart...........
*/

var RandomGen = new RNG();

var intro_template = [
	"..................................",
	"..................................",
	"..................................",
	"......Puzzle Script Terminal......",
	"..............v 1.0...............",
	"..................................",
	"..................................",
	"..................................",
	".........insert cartridge.........",
	"..................................",
	"..................................",
	"..................................",
	".................................."
];

var messagecontainer_template = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..........X to continue...........",
	"..................................",
	".................................."
];

var titletemplate_firstgo = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..........#.start game.#..........",
	"..................................",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];

var titletemplate_select0 = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"...........#.new game.#...........",
	"..................................",
	".............continue.............",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];

var titletemplate_select1 = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	".............new game.............",
	"..................................",
	"...........#.continue.#...........",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];


var titletemplate_firstgo_selected = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"###########.start game.###########",
	"..................................",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];

var titletemplate_select0_selected = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"############.new game.############",
	"..................................",
	".............continue.............",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];

var titletemplate_select1_selected = [
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	"..................................",
	".............new game.............",
	"..................................",
	"############.continue.############",
	"..................................",
	".arrow keys to move...............",
	".X to action......................",
	".Z to undo, R to restart..........",
	".................................."];

var titleImage=[];
var titleWidth=titletemplate_select1[0].length;
var titleHeight=titletemplate_select1.length;
var textMode=true;
var titleScreen=true;
var titleMode=0;//1 means there are options
var titleSelection=0;
var titleSelected=false;

function unloadGame() {
	state=introstate;
	level = new Level(0, 5, 5, 2, null);
	level.objects = new Int32Array(0);
	generateTitleScreen();
	canvasResize();
	redraw();
}

function generateTitleScreen()
{
	titleMode=curlevel>0?1:0;
	
	if (state.levels.length===0) {
		titleImage=intro_template;
		return;
	}

	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
	}

	if (titleMode===0) {
		if (titleSelected) {
			titleImage = deepClone(titletemplate_firstgo_selected);		
		} else {
			titleImage = deepClone(titletemplate_firstgo);					
		}
	} else {
		if (titleSelection===0) {
			if (titleSelected) {
				titleImage = deepClone(titletemplate_select0_selected);		
			} else {
				titleImage = deepClone(titletemplate_select0);					
			}			
		} else {
			if (titleSelected) {
				titleImage = deepClone(titletemplate_select1_selected);		
			} else {
				titleImage = deepClone(titletemplate_select1);					
			}						
		}
	}

	var noAction = 'noaction' in state.metadata;	
	var noUndo = 'noundo' in state.metadata;
	var noRestart = 'norestart' in state.metadata;
	if (noUndo && noRestart) {
		titleImage[11]="..................................";
	} else if (noUndo) {
		titleImage[11]=".R to restart.....................";
	} else if (noRestart) {
		titleImage[11]=".Z to undo.....................";
	}
	if (noAction) {
		titleImage[10]="..................................";
	}
	for (var i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	var width = titleImage[0].length;
	var titlelines=wordwrap(title,titleImage[0].length);
	for (var i=0;i<titlelines.length;i++) {
		var titleline=titlelines[i];
		var titleLength=titleline.length;
		var lmargin = ((width-titleLength)/2)|0;
		var rmargin = width-titleLength-lmargin;
		var row = titleImage[1+i];
		titleImage[1+i]=row.slice(0,lmargin)+titleline+row.slice(lmargin+titleline.length);
	}
	if (state.metadata.author!==undefined) {
		var attribution="by "+state.metadata.author;
		var attributionsplit = wordwrap(attribution,titleImage[0].length);
		for (var i=0;i<attributionsplit.length;i++) {
			var line = attributionsplit[i];
			var row = titleImage[3+i];
			titleImage[3+i]=row.slice(0,width-line.length-1)+line+row[row.length-1];			
		}
	}

}

var introstate = {
	title: "2D Whale World",
	attribution: "increpare",
   	objectCount: 2,
   	metadata:[],
   	levels:[],
   	bgcolor:"#000000",
   	fgcolor:"#FFFFFF"
};

var state = introstate;

function deepClone(item) {
    if (!item) { return item; } // null, undefined values check

    var types = [ Number, String, Boolean ], 
        result;

    // normalizing primitives if someone did new String('aaa'), or new Number('444');
    types.forEach(function(type) {
        if (item instanceof type) {
            result = type( item );
        }
    });

    if (typeof result == "undefined") {
        if (Object.prototype.toString.call( item ) === "[object Array]") {
            result = [];
            item.forEach(function(child, index, array) { 
                result[index] = deepClone( child );
            });
        } else if (typeof item == "object") {
            // testing that this is DOM
            if (item.nodeType && typeof item.cloneNode == "function") {
                var result = item.cloneNode( true );    
            } else if (!item.prototype) { // check that this is a literal
                if (item instanceof Date) {
                    result = new Date(item);
                } else {
                    // it is an object literal
                    result = {};
                    for (var i in item) {
                        result[i] = deepClone( item[i] );
                    }
                }
            } else {
                // depending what you would like here,
                // just keep the reference, or create new object
/*                if (false && item.constructor) {
                    // would not advice to do that, reason? Read below
                    result = new item.constructor();
                } else */{
                    result = item;
                }
            }
        } else {
            result = item;
        }
    }

    return result;
}

function wordwrap( str, width ) {
 
    width = width || 75;
    var cut = true;
 
    if (!str) { return str; }
 
    var regex = '.{1,' +width+ '}(\\s|$)' + (cut ? '|.{' +width+ '}|.+$' : '|\\S+?(\\s|$)');
 
    return str.match( RegExp(regex, 'g') );
 
}

var splitMessage=[];

function drawMessageScreen() {
	titleMode=0;
	textMode=true;
	titleImage = deepClone(messagecontainer_template);

	for (var i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	var width = titleImage[0].length;

	var message;
	if (messagetext==="") {
		var leveldat = state.levels[curlevel];
		message = leveldat.message.trim();
	} else {
		message = messagetext;
	}
	splitMessage = wordwrap(message,titleImage[0].length);

	for (var i=0;i<splitMessage.length;i++) {
		var m = splitMessage[i];
		var row = 5-((splitMessage.length/2)|0)+i;
		var messageLength=m.length;
		var lmargin = ((width-messageLength)/2)|0;
		var rmargin = width-messageLength-lmargin;
		var rowtext = titleImage[row];
		titleImage[row]=rowtext.slice(0,lmargin)+m+rowtext.slice(lmargin+m.length);		
	}

	if (quittingMessageScreen) {
		titleImage[10]=titleImage[9];
	}		
	canvasResize();
}

var loadedLevelSeed=0;

function loadLevelFromLevelDat(state,leveldat,randomseed) {	
	if (randomseed==null) {
		randomseed = (Math.random() + Date.now()).toString();
	}
	loadedLevelSeed = randomseed;
	RandomGen = new RNG(loadedLevelSeed);
	forceRegenImages=true;
	titleScreen=false;
	titleMode=curlevel>0?1:0;
	titleSelection=curlevel>0?1:0;
	titleSelected=false;
    againing=false;
    if (leveldat===undefined) {
    	consolePrint("Trying to access a level that doesn't exist.",true);
    	return;
    }
    if (leveldat.message===undefined) {
    	titleMode=0;
    	textMode=false;
		level = leveldat.clone();
		RebuildLevelArrays();

	    backups=[]
	    restartTarget=backupLevel();

	    if ('run_rules_on_level_start' in state.metadata) {
			processInput(-1,true);
	    }
	} else {
		tryPlayShowMessageSound();
		drawMessageScreen();
    	canvasResize();
	}

	clearInputHistory();
}

function loadLevelFromState(state,levelindex,randomseed) {	
    var leveldat = state.levels[levelindex];    
	curlevel=levelindex;
    if (leveldat.message===undefined) {
	    if (levelindex=== 0){ 
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();			
		}
    }
    loadLevelFromLevelDat(state,leveldat,randomseed);
}

var sprites = [
{
    color: '#423563',
    dat: [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
},
{
    color: '#252342',
    dat: [
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0]
    ]
}
];


generateTitleScreen();
canvasResize();

function tryPlaySimpleSound(soundname) {
	if (state.sfx_Events[soundname]!==undefined) {
		var seed = state.sfx_Events[soundname];
		playSound(seed);
	}
}
function tryPlayTitleSound() {
	tryPlaySimpleSound("titlescreen");
}

function tryPlayStartGameSound() {
	tryPlaySimpleSound("startgame");
}

function tryPlayEndGameSound() {
	tryPlaySimpleSound("endgame");
}

function tryPlayStartLevelSound() {
	tryPlaySimpleSound("startlevel");
}

function tryPlayEndLevelSound() {
	tryPlaySimpleSound("endlevel");
}

function tryPlayUndoSound(){
	tryPlaySimpleSound("undo");
}

function tryPlayRestartSound(){
	tryPlaySimpleSound("restart");
}

function tryPlayShowMessageSound(){
	tryPlaySimpleSound("showmessage");
}

function tryPlayCloseMessageSound(){
	tryPlaySimpleSound("closemessage");
}

var backups=[];
var restartTarget;

function backupLevel() {
	var ret = {
		dat : new Int32Array(level.objects),
		width : level.width,
		height : level.height,
		oldflickscreendat: oldflickscreendat.concat([])
	};
	return ret;
}

function setGameState(_state, command, randomseed) {
	oldflickscreendat=[];
	timer=0;
	autotick=0;
	winning=false;
	againing=false;
    messageselected=false;
    STRIDE_MOV=_state.STRIDE_MOV;
    STRIDE_OBJ=_state.STRIDE_OBJ;
    
	if (command===undefined) {
		command=["restart"];
	}
	if (state.levels.length===0 && command.length>0 && command[0]==="rebuild")  {
		command=["restart"];
	}
	if (randomseed===undefined) {
		randomseed=null;
	}
	RandomGen = new RNG(randomseed);

	state = _state;
    window.console.log('setting game state :D ');
    backups=[];
    //set sprites
    sprites = [];
    for (var n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            var object = state.objects[n];
            var sprite = {
                colors: object.colors,
                dat: object.spritematrix
            };
            sprites[object.id] = sprite;
        }
    }
    if (state.metadata.realtime_interval!==undefined) {
    	autotick=0;
    	autotickinterval=state.metadata.realtime_interval*1000;
    } else {
    	autotick=0;
    	autotickinterval=0;
    }

    if (state.metadata.key_repeat_interval!==undefined) {
		repeatinterval=state.metadata.key_repeat_interval*1000;
    } else {
    	repeatinterval=150;
    }

    if (state.metadata.again_interval!==undefined) {
		againinterval=state.metadata.again_interval*1000;
    } else {
    	againinterval=150;
    }
    if (throttle_movement && autotickinterval===0) {
    	logWarning("throttle_movement is designed for use in conjunction with realtime_interval. Using it in other situations makes games gross and unresponsive, broadly speaking.  Please don't.");
    }
    norepeat_action = state.metadata.norepeat_action!==undefined;
    
    switch(command[0]){
    	case "restart":
    	{
		    winning=false;
		    timer=0;
		    titleScreen=true;
		    tryPlayTitleSound();
		    textMode=true;
		    titleSelection=curlevel>0?1:0;
		    titleSelected=false;
		    quittingMessageScreen=false;
		    quittingTitleScreen=false;
		    messageselected=false;
		    titleMode = 0;
		    if (curlevel>0) {
		    	titleMode=1;
		    }
		    generateTitleScreen();
		    break;
		}
		case "rebuild":
		{
			//do nothing
			break;
		}
		case "loadLevel":
		{
			var targetLevel = command[1];
			curlevel=i;
		    winning=false;
		    timer=0;
		    titleScreen=false;
		    textMode=false;
		    titleSelection=curlevel>0?1:0;
		    titleSelected=false;
		    quittingMessageScreen=false;
		    quittingTitleScreen=false;
		    messageselected=false;
		    titleMode = 0;
			loadLevelFromState(state,targetLevel,randomseed);
			break;
		}
		case "levelline":
		{
			var targetLine = command[1];
			for (var i=state.levels.length-1;i>=0;i--) {
				var level= state.levels[i];
				if(level.lineNumber<=targetLine+1) {
					curlevel=i;
				    winning=false;
				    timer=0;
				    titleScreen=false;
				    textMode=false;
				    titleSelection=curlevel>0?1:0;
				    titleSelected=false;
				    quittingMessageScreen=false;
				    quittingTitleScreen=false;
				    messageselected=false;
				    titleMode = 0;
					loadLevelFromState(state,i);
					break;
				}
			}
			break;
		}
	}
	
	if(command[0] !== "rebuild") {
		clearInputHistory();
	}
	canvasResize();



	if (canYoutube) {
		if ('youtube' in state.metadata) {
			var youtubeid=state.metadata['youtube'];
			var url = "https://youtube.googleapis.com/v/"+youtubeid+"?autoplay=1&loop=1&playlist="+youtubeid;
			ifrm = document.createElement("IFRAME");
			ifrm.setAttribute("src",url);
			ifrm.style.visibility="hidden";
			ifrm.style.width="500px";
			ifrm.style.height="500px";
			ifrm.style.position="absolute";
			ifrm.style.top="-1000px";
			ifrm.style.left="-1000px";
//			ifrm.style.display="none";
			document.body.appendChild(ifrm);
		}

		/*
		if ('youtube' in state.metadata) {
			var div_container = document.createElement('DIV');
			var div_front = document.createElement('DIV');
			div_front.style.zIndex=-100;	
			div_front.style.backgroundColor=state.bgcolor;
			div_front.style.position= "absolute";
			div_front.style.width="500px";
			div_front.style.height="500px";
			var div_back = document.createElement('DIV');
			div_back.style.zIndex=-200;
			div_back.style.position= "absolute";
			
			div_container.appendChild(div_back);
			div_container.appendChild(div_front);
			
			var youtubeid=state.metadata['youtube'];
			var url = "https://youtube.googleapis.com/v/"+youtubeid+"?autoplay=1&loop=1&playlist="+youtubeid;
			ifrm = document.createElement("IFRAME");
			ifrm.setAttribute("src",url);
			ifrm.style.visibility="hidden";
			ifrm.style.width="500px";
			ifrm.style.height="500px";
			ifrm.frameBorder="0";
//			ifrm.style.display="none";

			div_back.appendChild(ifrm);
			document.body.appendChild(div_container);
			*/
	}
	
}

function RebuildLevelArrays() {
	level.movements = new Int32Array(level.n_tiles * STRIDE_MOV);

    level.rigidMovementAppliedMask = [];
    level.rigidGroupIndexMask = [];
	level.rowCellContents = [];
	level.colCellContents = [];
	level.mapCellContents = new BitVec(STRIDE_OBJ);
	_movementsVec = new BitVec(STRIDE_MOV);

	_o1 = new BitVec(STRIDE_OBJ);
	_o2 = new BitVec(STRIDE_OBJ);
	_o2_5 = new BitVec(STRIDE_OBJ);
	_o3 = new BitVec(STRIDE_OBJ);
	_o4 = new BitVec(STRIDE_OBJ);
	_o5 = new BitVec(STRIDE_OBJ);
	_o6 = new BitVec(STRIDE_OBJ);
	_o7 = new BitVec(STRIDE_OBJ);
	_o8 = new BitVec(STRIDE_OBJ);
	_o9 = new BitVec(STRIDE_OBJ);
	_o10 = new BitVec(STRIDE_OBJ);
	_o11 = new BitVec(STRIDE_OBJ);
	_o12 = new BitVec(STRIDE_OBJ);
	_m1 = new BitVec(STRIDE_MOV);
	_m2 = new BitVec(STRIDE_MOV);
	_m3 = new BitVec(STRIDE_MOV);
	

    for (var i=0;i<level.height;i++) {
    	level.rowCellContents[i]=new BitVec(STRIDE_OBJ);	    	
    }
    for (var i=0;i<level.width;i++) {
    	level.colCellContents[i]=new BitVec(STRIDE_OBJ);	    	
    }

    for (var i=0;i<level.n_tiles;i++)
    {
        level.rigidMovementAppliedMask[i]=new BitVec(STRIDE_MOV);
        level.rigidGroupIndexMask[i]=new BitVec(STRIDE_MOV);
    }
}

var messagetext="";
function restoreLevel(lev) {
	oldflickscreendat=lev.oldflickscreendat.concat([]);

	level.objects = new Int32Array(lev.dat);
	if (level.width !== lev.width || level.height !== lev.height) {
		level.width = lev.width;
		level.height = lev.height;
		level.n_tiles = lev.width * lev.height;
		RebuildLevelArrays();
		//regenerate all other stride-related stuff
	}
	else 
	{
	// layercount doesn't change

		for (var i=0;i<level.n_tiles;i++) {
			level.movements[i]=0;
			level.rigidMovementAppliedMask[i]=0;
			level.rigidGroupIndexMask[i]=0;
		}	

	    for (var i=0;i<level.height;i++) {
	    	var rcc = level.rowCellContents[i];
	    	rcc.setZero();
	    }
	    for (var i=0;i<level.width;i++) {
	    	var ccc = level.colCellContents[i];
	    	ccc.setZero();
	    }
	}

    againing=false;
    level.commandQueue=[];
}

var zoomscreen=false;
var flickscreen=false;
var screenwidth=0;
var screenheight=0;


function DoRestart(force) {

	if (force!==true && ('norestart' in state.metadata)) {
		return;
	}
	if (force===false) {
		backups.push(backupLevel());
	}

	if (verbose_logging) {
		consolePrint("--- restarting ---",true);
	}

	restoreLevel(restartTarget);
	tryPlayRestartSound();

	if ('run_rules_on_level_start' in state.metadata) {
    	processInput(-1,true);
	}
	
	level.commandQueue=[];
}

function DoUndo(force) {
	if ((!levelEditorOpened)&&('noundo' in state.metadata && force!==true)) {
		return;
	}
	if (verbose_logging) {
		consolePrint("--- undoing ---",true);
	}
	if (backups.length>0) {
		var tobackup = backups[backups.length-1];
		restoreLevel(tobackup);
		backups = backups.splice(0,backups.length-1);
		if (! force) {
			tryPlayUndoSound();
		}
	}
}

function getPlayerPositions() {
    var result=[];
    var playerMask = state.playerMask;
    for (i=0;i<level.n_tiles;i++) {
        level.getCellInto(i,_o11);
        if (playerMask.anyBitsInCommon(_o11)) {
            result.push(i);
        }
    }
    return result;
}

function getLayersOfMask(cellMask) {
    var layers=[];
    for (var i=0;i<state.objectCount;i++) {
        if (cellMask.get(i)) {
            var n = state.idDict[i];
            var o = state.objects[n];
            layers.push(o.layer)
        }
    }
    return layers;
}

function moveEntitiesAtIndex(positionIndex, entityMask, dirMask) {
    var cellMask = level.getCell(positionIndex);
    cellMask.iand(entityMask);
    var layers = getLayersOfMask(cellMask);

    var movementMask = level.getMovements(positionIndex);
    for (var i=0;i<layers.length;i++) {
    	movementMask.ishiftor(dirMask, 5 * layers[i]);
    }
    level.setMovements(positionIndex, movementMask);
}


function startMovement(dir) {
	var movedany=false;
    var playerPositions = getPlayerPositions();
    for (var i=0;i<playerPositions.length;i++) {
        var playerPosIndex = playerPositions[i];
        moveEntitiesAtIndex(playerPosIndex,state.playerMask,dir);
    }
    return playerPositions;
}

var dirMasksDelta = {
     1:[0,-1],//up
     2:[0,1],//'down'  : 
     4:[-1,0],//'left'  : 
     8:[1,0],//'right' : 
     15:[0,0],//'?' : 
     16:[0,0],//'action' : 
     3:[0,0]//'no'
};

var dirMaskName = {
     1:'up',
     2:'down'  ,
     4:'left'  , 
     8:'right',  
     15:'?' ,
     16:'action',
     3:'no'
};

var seedsToPlay_CanMove=[];
var seedsToPlay_CantMove=[];

function repositionEntitiesOnLayer(positionIndex,layer,dirMask) 
{
    var delta = dirMasksDelta[dirMask];

    var dx = delta[0];
    var dy = delta[1];
    var tx = ((positionIndex/level.height)|0);
    var ty = ((positionIndex%level.height));
    var maxx = level.width-1;
    var maxy = level.height-1;

    if ( (tx===0&&dx<0) || (tx===maxx&&dx>0) || (ty===0&&dy<0) || (ty===maxy&&dy>0)) {
    	return false;
    }

    var targetIndex = (positionIndex+delta[1]+delta[0]*level.height)%level.n_tiles;

    var layerMask = state.layerMasks[layer];
    var targetMask = level.getCellInto(targetIndex,_o7);
    var sourceMask = level.getCellInto(positionIndex,_o8);

    if (layerMask.anyBitsInCommon(targetMask) && (dirMask!=16)) {
        return false;
    }

	for (var i=0;i<state.sfx_MovementMasks.length;i++) {
		var o = state.sfx_MovementMasks[i];
		var objectMask = o.objectMask;
		if (objectMask.anyBitsInCommon(sourceMask)) {
			var movementMask = level.getMovements(positionIndex);
			var directionMask = o.directionMask;
			if (movementMask.anyBitsInCommon(directionMask) && seedsToPlay_CanMove.indexOf(o.seed)===-1) {
				seedsToPlay_CanMove.push(o.seed);
			}
		}
	}

    var movingEntities = sourceMask.clone();
    sourceMask.iclear(layerMask);
    movingEntities.iand(layerMask);
    targetMask.ior(movingEntities);

    level.setCell(positionIndex, sourceMask);
    level.setCell(targetIndex, targetMask);

    var colIndex=(targetIndex/level.height)|0;
	var rowIndex=(targetIndex%level.height);
    level.colCellContents[colIndex].ior(movingEntities);
    level.rowCellContents[rowIndex].ior(movingEntities);
    level.mapCellContents.ior(layerMask);
    return true;
}

function repositionEntitiesAtCell(positionIndex) {
    var movementMask = level.getMovements(positionIndex);
    if (movementMask.iszero())
        return false;

    var moved=false;
    for (var layer=0;layer<level.layerCount;layer++) {
        var layerMovement = movementMask.getshiftor(0x1f, 5*layer);
        if (layerMovement!==0) {
            var thismoved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement);
            if (thismoved) {
                movementMask.ishiftclear(layerMovement, 5*layer);
                moved = true;
            }
        }
    }

   	level.setMovements(positionIndex, movementMask);

    return moved;
}


function Level(lineNumber, width, height, layerCount, objects) {
	this.lineNumber = lineNumber;
	this.width = width;
	this.height = height;
	this.n_tiles = width * height;
	this.objects = objects;
	this.layerCount = layerCount;
	this.commandQueue = [];
}

Level.prototype.clone = function() {
	var clone = new Level(this.lineNumber, this.width, this.height, this.layerCount, null);
	clone.objects = new Int32Array(this.objects);
	return clone;
}

Level.prototype.getCell = function(index) {
	return new BitVec(this.objects.subarray(index * STRIDE_OBJ, index * STRIDE_OBJ + STRIDE_OBJ));
}

Level.prototype.getCellInto = function(index,targetarray) {
	for (var i=0;i<STRIDE_OBJ;i++) {
		targetarray.data[i]=this.objects[index*STRIDE_OBJ+i];	
	}
	return targetarray;
}

Level.prototype.setCell = function(index, vec) {
	for (var i = 0; i < vec.data.length; ++i) {
		this.objects[index * STRIDE_OBJ + i] = vec.data[i];
	}
}

var _movementsVec;

Level.prototype.getMovements = function(index) {
	for (var i=0;i<STRIDE_MOV;i++) {
		_movementsVec.data[i]=this.movements[index*STRIDE_MOV+i];	
	}
	return _movementsVec;
}

Level.prototype.setMovements = function(index, vec) {
	for (var i = 0; i < vec.data.length; ++i) {
		this.movements[index * STRIDE_MOV + i] = vec.data[i];
	}
}

var ellipsisPattern = ['ellipsis'];

function BitVec(init) {
	this.data = new Int32Array(init);
	return this;
}

BitVec.prototype.cloneInto = function(target) {
	for (var i=0;i<this.data.length;++i) {
		target.data[i]=this.data[i];
	}
	return target;
}
BitVec.prototype.clone = function() {
	return new BitVec(this.data);
}

BitVec.prototype.iand = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] &= other.data[i];
	}
}

BitVec.prototype.ior = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] |= other.data[i];
	}
}

BitVec.prototype.iclear = function(other) {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i] &= ~other.data[i];
	}
}

BitVec.prototype.ibitset = function(ind) {
	this.data[ind>>5] |= 1 << (ind & 31);
}

BitVec.prototype.ibitclear = function(ind) {
	this.data[ind>>5] &= ~(1 << (ind & 31));
}

BitVec.prototype.get = function(ind) {
	return (this.data[ind>>5] & 1 << (ind & 31)) !== 0;
}

BitVec.prototype.getshiftor = function(mask, shift) {
	var toshift = shift & 31;
	var ret = this.data[shift>>5] >>> (toshift);
	if (toshift) {
		ret |= this.data[(shift>>5)+1] << (32 - toshift);
	}
	return ret & mask;
}

BitVec.prototype.ishiftor = function(mask, shift) {
	var toshift = shift&31;
	var low = mask << toshift;
	this.data[shift>>5] |= low;
	if (toshift) {
		var high = mask >> (32 - toshift);
		this.data[(shift>>5)+1] |= high;
	}
}

BitVec.prototype.ishiftclear = function(mask, shift) {
	var toshift = shift & 31;
	var low = mask << toshift;
	this.data[shift>>5] &= ~low;
	if (toshift){
		var high = mask >> (32 - (shift & 31));
		this.data[(shift>>5)+1] &= ~high;
	}
}

BitVec.prototype.equals = function(other) {
	if (this.data.length !== other.data.length)
		return false;
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i] !== other.data[i])
			return false;
	}
	return true;
}

BitVec.prototype.setZero = function() {
	for (var i = 0; i < this.data.length; ++i) {
		this.data[i]=0;
	}
}

BitVec.prototype.iszero = function() {
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i])
			return false;
	}
	return true;
}

BitVec.prototype.bitsSetInArray = function(arr) {
	for (var i = 0; i < this.data.length; ++i) {
		if ((this.data[i] & arr[i]) !== this.data[i]) {
			return false;
		}
	}
	return true;
}

BitVec.prototype.bitsClearInArray = function(arr) {
	for (var i = 0; i < this.data.length; ++i) {
		if (this.data[i] & arr[i]) {
			return false;
		}
	}
	return true;
}

BitVec.prototype.anyBitsInCommon = function(other) {
	return !this.bitsClearInArray(other.data);
}

function Rule(rule) {
	this.direction = rule[0]; 		/* direction rule scans in */
	this.patterns = rule[1];		/* lists of CellPatterns to match */
	this.hasReplacements = rule[2];
	this.lineNumber = rule[3];		/* rule source for debugging */
	this.isEllipsis = rule[4];		/* true if pattern has ellipsis */
	this.groupNumber = rule[5];		/* execution group number of rule */
	this.isRigid = rule[6];
	this.commands = rule[7];		/* cancel, restart, sfx, etc */
	this.isRandom = rule[8];
	this.cellRowMasks = rule[9];
	this.cellRowMatches = [];
	for (var i=0;i<this.patterns.length;i++) {
		this.cellRowMatches.push(this.generateCellRowMatchesFunction(this.patterns[i],this.isEllipsis[i]));
	}
	/* TODO: eliminate isRigid, groupNumber, isRandom
	from this class by moving them up into a RuleGroup class */
}


Rule.prototype.generateCellRowMatchesFunction = function(cellRow,hasEllipsis)  {
	if (hasEllipsis==false) {
		var delta = dirMasksDelta[this.direction];
		var d0 = delta[0];
		var d1 = delta[1];
		var cr_l = cellRow.length;

			/*
			hard substitute in the first one - if I substitute in all of them, firefox chokes.
			*/
		var fn = "var d = "+d1+"+"+d0+"*level.height;\n";
		var mul = STRIDE_OBJ === 1 ? '' : '*'+STRIDE_OBJ;	
		for (var i = 0; i < STRIDE_OBJ; ++i) {
			fn += 'var cellObjects' + i + ' = level.objects[i' + mul + (i ? '+'+i: '') + '];\n';
		}
		mul = STRIDE_MOV === 1 ? '' : '*'+STRIDE_MOV;
		for (var i = 0; i < STRIDE_MOV; ++i) {
			fn += 'var cellMovements' + i + ' = level.movements[i' + mul + (i ? '+'+i: '') + '];\n';
		}
		fn += "return "+cellRow[0].generateMatchString('0_');// cellRow[0].matches(i)";
		for (var cellIndex=1;cellIndex<cr_l;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches((i+"+cellIndex+"*d)%level.n_tiles)";
		}
		fn+=";";

		if (fn in matchCache) {
			return matchCache[fn];
		}
		//console.log(fn.replace(/\s+/g, ' '));
		return matchCache[fn] = new Function("cellRow","i",fn);
	} else {
		var delta = dirMasksDelta[this.direction];
		var d0 = delta[0];
		var d1 = delta[1];
		var cr_l = cellRow.length;


		var fn = "var d = "+d1+"+"+d0+"*level.height;\n";
		fn += "var result = [];\n"
		fn += "if(cellRow[0].matches(i)";
		var cellIndex=1;
		for (;cellRow[cellIndex]!==ellipsisPattern;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches((i+"+cellIndex+"*d)%level.n_tiles)";
		}
		cellIndex++;
		fn+=") {\n";
		fn+="\tfor (var k=kmin;k<kmax;k++) {\n"
		fn+="\t\tif(cellRow["+cellIndex+"].matches((i+d*(k+"+(cellIndex-1)+"))%level.n_tiles)";
		cellIndex++;
		for (;cellIndex<cr_l;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches((i+d*(k+"+(cellIndex-1)+"))%level.n_tiles)";			
		}
		fn+="){\n";
		fn+="\t\t\tresult.push([i,k]);\n";
		fn+="\t\t}\n"
		fn+="\t}\n";				
		fn+="}\n";		
		fn+="return result;"


		if (fn in matchCache) {
			return matchCache[fn];
		}
		//console.log(fn.replace(/\s+/g, ' '));
		return matchCache[fn] = new Function("cellRow","i","kmax","kmin",fn);
	}
//say cellRow has length 3, with a split in the middle
/*
function cellRowMatchesWildcardFunctionGenerate(direction,cellRow,i, maxk, mink) {

	var result = [];
	var matchfirsthalf = cellRow[0].matches(i);
	if (matchfirsthalf) {
		for (var k=mink;k<maxk;k++) {
			if (cellRow[2].matches((i+d*(k+0))%level.n_tiles)) {
				result.push([i,k]);
			}
		}
	}
	return result;
}
*/
	

}


Rule.prototype.toJSON = function() {
	/* match construction order for easy deserialization */
	return [
		this.direction, this.patterns, this.hasReplacements, this.lineNumber, this.isEllipsis,
		this.groupNumber, this.isRigid, this.commands, this.isRandom, this.cellRowMasks
	];
};

var STRIDE_OBJ = 1;
var STRIDE_MOV = 1;

function CellPattern(row) {
	this.objectsPresent = row[0];
	this.objectsMissing = row[1];
	this.anyObjectsPresent = row[2];
	this.movementsPresent = row[3];
	this.movementsMissing = row[4];
	this.matches = this.generateMatchFunction();
	this.replacement = row[5];
};

function CellReplacement(row) {
	this.objectsClear = row[0];
	this.objectsSet = row[1];
	this.movementsClear = row[2];
	this.movementsSet = row[3];
	this.movementsLayerMask = row[4];
	this.randomEntityMask = row[5];
	this.randomDirMask = row[6];
};


var matchCache = {};



CellPattern.prototype.generateMatchString = function() {
	var fn = "(true";
	for (var i = 0; i < Math.max(STRIDE_OBJ, STRIDE_MOV); ++i) {
		var co = 'cellObjects' + i;
		var cm = 'cellMovements' + i;
		var op = this.objectsPresent.data[i];
		var om = this.objectsMissing.data[i];
		var mp = this.movementsPresent.data[i];
		var mm = this.movementsMissing.data[i];
		if (op) {
			if (op&(op-1))
				fn += '\t\t&& ((' + co + '&' + op + ')===' + op + ')\n';
			else
				fn += '\t\t&& (' + co + '&' + op + ')\n';
		}
		if (om)
			fn += '\t\t&& !(' + co + '&' + om + ')\n';
		if (mp) {
			if (mp&(mp-1))
				fn += '\t\t&& ((' + cm + '&' + mp + ')===' + mp + ')\n';
			else
				fn += '\t\t&& (' + cm + '&' + mp + ')\n';
		}
		if (mm)
			fn += '\t\t&& !(' + cm + '&' + mm + ')\n';
	}
	for (var j = 0; j < this.anyObjectsPresent.length; j++) {
		fn += "\t\t&& (0";
		for (var i = 0; i < STRIDE_OBJ; ++i) {
			var aop = this.anyObjectsPresent[j].data[i];
			if (aop)
				fn += "|(cellObjects" + i + "&" + aop + ")";
		}
		fn += ")";
	}
	fn += '\t)';
	return fn;
}

CellPattern.prototype.generateMatchFunction = function() {
	var i;
	var fn = '';
	var mul = STRIDE_OBJ === 1 ? '' : '*'+STRIDE_OBJ;	
	for (var i = 0; i < STRIDE_OBJ; ++i) {
		fn += '\tvar cellObjects' + i + ' = level.objects[i' + mul + (i ? '+'+i: '') + '];\n';
	}
	mul = STRIDE_MOV === 1 ? '' : '*'+STRIDE_MOV;
	for (var i = 0; i < STRIDE_MOV; ++i) {
		fn += '\tvar cellMovements' + i + ' = level.movements[i' + mul + (i ? '+'+i: '') + '];\n';
	}
	fn += "return " + this.generateMatchString()+';';
	if (fn in matchCache) {
		return matchCache[fn];
	}
	//console.log(fn.replace(/\s+/g, ' '));
	return matchCache[fn] = new Function("i",fn);
}

CellPattern.prototype.toJSON = function() {
	return [
		this.movementMask, this.cellMask, this.nonExistenceMask,
		this.moveNonExistenceMask, this.moveStationaryMask, this.randomDirOrEntityMask,
		this.movementsToRemove
	];
};

var _o1,_o2,_o2_5,_o3,_o4,_o5,_o6,_o7,_o8,_o9,_o10,_o11,_o12;
var _m1,_m2,_m3;

CellPattern.prototype.replace = function(rule, currentIndex) {
	var replace = this.replacement;

	if (replace === null) {
		return false;
	}

	var replace_RandomEntityMask = replace.randomEntityMask;
	var replace_RandomDirMask = replace.randomDirMask;

	var objectsSet = replace.objectsSet.cloneInto(_o1);
	var objectsClear = replace.objectsClear.cloneInto(_o2);

	var movementsSet = replace.movementsSet.cloneInto(_m1);
	var movementsClear = replace.movementsClear.cloneInto(_m2);
	movementsClear.ior(replace.movementsLayerMask);

	if (!replace_RandomEntityMask.iszero()) {
		var choices=[];
		for (var i=0;i<32*STRIDE_OBJ;i++) {
			if (replace_RandomEntityMask.get(i)) {
				choices.push(i);
			}
		}
		var rand = choices[Math.floor(RandomGen.uniform() * choices.length)];
		var n = state.idDict[rand];
		var o = state.objects[n];
		objectsSet.ibitset(rand);
		objectsClear.ior(state.layerMasks[o.layer]);
		movementsClear.ishiftor(0x1f, 5 * o.layer);
	}
	if (!replace_RandomDirMask.iszero()) {
		for (var layerIndex=0;layerIndex<level.layerCount;layerIndex++){
			if (replace_RandomDirMask.get(5*layerIndex)) {
				var randomDir = Math.floor(RandomGen.uniform()*4);
				movementsSet.ibitset(randomDir + 5 * layerIndex);
			}
		}
	}
	
	var curCellMask = level.getCellInto(currentIndex,_o2_5);
	var curMovementMask = level.getMovements(currentIndex);

	var oldCellMask = curCellMask.cloneInto(_o3);
	var oldMovementMask = curMovementMask.cloneInto(_m3);

	curCellMask.iclear(objectsClear);
	curCellMask.ior(objectsSet);

	curMovementMask.iclear(movementsClear);
	curMovementMask.ior(movementsSet);

	var rigidchange=false;
	var curRigidGroupIndexMask =0;
	var curRigidMovementAppliedMask =0;
	if (rule.isRigid) {
		var rigidGroupIndex = state.groupNumber_to_RigidGroupIndex[rule.groupNumber];
		rigidGroupIndex++;//don't forget to -- it when decoding :O
		var rigidMask = new BitVec(STRIDE_MOV);
		for (var layer = 0; layer < level.layerCount; layer++) {
			rigidMask.ishiftor(rigidGroupIndex, layer * 5);
		}
		rigidMask.iand(replace.movementsLayerMask);
		curRigidGroupIndexMask = level.rigidGroupIndexMask[currentIndex] || new BitVec(STRIDE_MOV);
		curRigidMovementAppliedMask = level.rigidMovementAppliedMask[currentIndex] || new BitVec(STRIDE_MOV);

		if (!rigidMask.bitsSetInArray(curRigidGroupIndexMask.data) || 
			!replace.movementsLayerMask.bitsSetInArray(curRigidMovementAppliedMask.data) ) {
			curRigidGroupIndexMask.ior(rigidMask);
			curRigidMovementAppliedMask.ior(replace.movementsLayerMask);
			rigidchange=true;

		}
	}

	var result = false;

	//check if it's changed
	if (!oldCellMask.equals(curCellMask) || !oldMovementMask.equals(curMovementMask) || rigidchange) { 
		result=true;
		if (rigidchange) {
			level.rigidGroupIndexMask[currentIndex] = curRigidGroupIndexMask;
			level.rigidMovementAppliedMask[currentIndex] = curRigidMovementAppliedMask;
		}

		var created = curCellMask.cloneInto(_o4);
		created.iclear(oldCellMask);
		sfxCreateMask.ior(created);
		var destroyed = oldCellMask.cloneInto(_o5);
		destroyed.iclear(curCellMask);
		sfxDestroyMask.ior(destroyed);

		level.setCell(currentIndex, curCellMask);
		level.setMovements(currentIndex, curMovementMask);

		var colIndex=(currentIndex/level.height)|0;
		var rowIndex=(currentIndex%level.height);
		level.colCellContents[colIndex].ior(curCellMask);
		level.rowCellContents[rowIndex].ior(curCellMask);
		level.mapCellContents.ior(curCellMask);
	}

	return result;
}


//say cellRow has length 5, with a split in the middle
/*
function cellRowMatchesWildcardFunctionGenerate(direction,cellRow,i, maxk, mink) {

	var result = [];
	var matchfirsthalf = cellRow[0].matches(i)&&cellRow[1].matches((i+d)%level.n_tiles);
	if (matchfirsthalf) {
		for (var k=mink,kmaxk;k++) {
			if (cellRow[2].matches((i+d*(k+0))%level.n_tiles)&&cellRow[2].matches((i+d*(k+1))%level.n_tiles)) {
				result.push([i,k]);
			}
		}
	}
	return result;
}
*/

function DoesCellRowMatchWildCard(direction,cellRow,i,maxk,mink) {
	if (mink === undefined) {
		mink = 0;
	}

	var cellPattern = cellRow[0];

    //var result=[];

    if (cellPattern.matches(i)){
    	var delta = dirMasksDelta[direction];
    	var d0 = delta[0]*level.height;
		var d1 = delta[1];
        var targetIndex = i;

        for (var j=1;j<cellRow.length;j+=1) {
            targetIndex = (targetIndex+d1+d0)%level.n_tiles;

            var cellPattern = cellRow[j]
            if (cellPattern === ellipsisPattern) {
            	//BAM inner loop time
            	for (var k=mink;k<maxk;k++) {
            		var targetIndex2=targetIndex;
                    targetIndex2 = (targetIndex2+(d1+d0)*(k)+level.n_tiles)%level.n_tiles;
            		for (var j2=j+1;j2<cellRow.length;j2++) {
		                cellPattern = cellRow[j2];
					    if (!cellPattern.matches(targetIndex2)) {
					    	break;
					    }
                        targetIndex2 = (targetIndex2+d1+d0)%level.n_tiles;
            		}

		            if (j2>=cellRow.length) {
		            	return true;
		                //result.push([i,k]);
		            }
            	}
            	break;
            } else if (!cellPattern.matches(targetIndex)) {
				break;
            }
        }               
    }  
    return false;
}

//say cellRow has length 3
/*
CellRow Matches can be specialized to look something like:
function cellRowMatchesFunctionGenerate(direction,cellRow,i) {
	var delta = dirMasksDelta[direction];
	var d = delta[1]+delta[0]*level.height;
	return cellRow[0].matches(i)&&cellRow[1].matches((i+d)%level.n_tiles)&&cellRow[2].matches((i+2*d)%level.n_tiles);
}
*/

function DoesCellRowMatch(direction,cellRow,i,k) {
	var cellPattern = cellRow[0];
    if (cellPattern.matches(i)) {

	    var delta = dirMasksDelta[direction];
	    var d0 = delta[0]*level.height;
		var d1 = delta[1];
		var cr_l = cellRow.length;

        var targetIndex = i;
        for (var j=1;j<cr_l;j++) {
            targetIndex = (targetIndex+d1+d0)%level.n_tiles;
            cellPattern = cellRow[j];
			if (cellPattern === ellipsisPattern) {
					//only for once off verifications
            	targetIndex = (targetIndex+(d1+d0)*k)%level.n_tiles; 					
            }
		    if (!cellPattern.matches(targetIndex)) {
                break;
            }
        }   
        
        if (j>=cellRow.length) {
            return true;
        }

    }  
    return false;
}

function matchCellRow(direction, cellRowMatch, cellRow, cellRowMask) {	
	var result=[];
	
	if ((!cellRowMask.bitsSetInArray(level.mapCellContents.data))) {
		return result;
	}

	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

    var len=cellRow.length;

    switch(direction) {
    	case 1://up
    	{
    		ymin+=(len-1);
    		break;
    	}
    	case 2: //down 
    	{
			ymax-=(len-1);
			break;
    	}
    	case 4: //left
    	{
    		xmin+=(len-1);
    		break;
    	}
    	case 8: //right
		{
			xmax-=(len-1);	
			break;
		}
    	default:
    	{
    		window.console.log("EEEP "+direction);
    	}
    }

    var horizontal=direction>2;
    if (horizontal) {
		for (var y=ymin;y<ymax;y++) {
			if (!cellRowMask.bitsSetInArray(level.rowCellContents[y].data)) {
				continue;
			}

			for (var x=xmin;x<xmax;x++) {
				var i = x*level.height+y;
				if (cellRowMatch(cellRow,i))
				{
					result.push(i);
				}
			}
		}
	} else {
		for (var x=xmin;x<xmax;x++) {
			if (!cellRowMask.bitsSetInArray(level.colCellContents[x].data)) {
				continue;
			}

			for (var y=ymin;y<ymax;y++) {
				var i = x*level.height+y;
				if (cellRowMatch(	cellRow,i))
				{
					result.push(i);
				}
			}
		}		
	}

	return result;
}


function matchCellRowWildCard(direction, cellRowMatch, cellRow,cellRowMask) {
	var result=[];
	if ((!cellRowMask.bitsSetInArray(level.mapCellContents.data))) {
		return result;
	}
	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

	var len=cellRow.length-1;//remove one to deal with wildcard
    switch(direction) {
    	case 1://up
    	{
    		ymin+=(len-1);
    		break;
    	}
    	case 2: //down 
    	{
			ymax-=(len-1);
			break;
    	}
    	case 4: //left
    	{
    		xmin+=(len-1);
    		break;
    	}
    	case 8: //right
		{
			xmax-=(len-1);	
			break;
		}
    	default:
    	{
    		window.console.log("EEEP2 "+direction);
    	}
    }



    var horizontal=direction>2;
    if (horizontal) {
		for (var y=ymin;y<ymax;y++) {
			if (!cellRowMask.bitsSetInArray(level.rowCellContents[y].data)) {
				continue;
			}

			for (var x=xmin;x<xmax;x++) {
				var i = x*level.height+y;
				var kmax;

				if (direction === 4) { //left
					kmax=x-len+2;
				} else if (direction === 8) { //right
					kmax=level.width-(x+len)+1;	
				} else {
					window.console.log("EEEP2 "+direction);					
				}

				result.push.apply(result, cellRowMatch(cellRow,i,kmax,0));
			}
		}
	} else {
		for (var x=xmin;x<xmax;x++) {
			if (!cellRowMask.bitsSetInArray(level.colCellContents[x].data)) {
				continue;
			}

			for (var y=ymin;y<ymax;y++) {
				var i = x*level.height+y;
				var kmax;

				if (direction === 2) { // down
					kmax=level.height-(y+len)+1;
				} else if (direction === 1) { // up
					kmax=y-len+2;					
				} else {
					window.console.log("EEEP2 "+direction);
				}
				result.push.apply(result, cellRowMatch(cellRow,i,kmax,0));
			}
		}		
	}

	return result;
}

function generateTuples(lists) {
    var tuples=[[]];

    for (var i=0;i<lists.length;i++)
    {
        var row = lists[i];
        var newtuples=[];
        for (var j=0;j<row.length;j++) {
            var valtoappend = row[j];
            for (var k=0;k<tuples.length;k++) {
                var tuple=tuples[k];
                var newtuple = tuple.concat([valtoappend]);
                newtuples.push(newtuple);
            }
        }
        tuples=newtuples;
    }
    return tuples;
}

var rigidBackups=[]

function commitPreservationState(ruleGroupIndex) {
	var propagationState = {
		ruleGroupIndex:ruleGroupIndex,
		//don't need to know the tuple index
		objects:new Int32Array(level.objects),
		movements:new Int32Array(level.movements),
		rigidGroupIndexMask:level.rigidGroupIndexMask.concat([]),
		rigidMovementAppliedMask:level.rigidMovementAppliedMask.concat([]),
		bannedGroup:level.bannedGroup.concat([])
	};
	rigidBackups[ruleGroupIndex]=propagationState;
	return propagationState;
}

function restorePreservationState(preservationState) {
//don't need to concat or anythign here, once something is restored it won't be used again.
	level.objects = new Int32Array(preservationState.objects);
	level.movements = new Int32Array(preservationState.movements);
	level.rigidGroupIndexMask = preservationState.rigidGroupIndexMask.concat([]);
    level.rigidMovementAppliedMask = preservationState.rigidMovementAppliedMask.concat([]);
    sfxCreateMask=new BitVec(STRIDE_OBJ);
    sfxDestroyMask=new BitVec(STRIDE_OBJ);
//	rigidBackups = preservationState.rigidBackups;
}

Rule.prototype.findMatches = function() {
	var matches=[];
	var cellRowMasks=this.cellRowMasks;
    for (var cellRowIndex=0;cellRowIndex<this.patterns.length;cellRowIndex++) {
        var cellRow = this.patterns[cellRowIndex];
        var matchFunction = this.cellRowMatches[cellRowIndex];
        if (this.isEllipsis[cellRowIndex]) {//if ellipsis     
        	var match = matchCellRowWildCard(this.direction,matchFunction,cellRow,cellRowMasks[cellRowIndex]);  
        } else {
        	var match = matchCellRow(this.direction,matchFunction,cellRow,cellRowMasks[cellRowIndex]);               	
        }
        if (match.length===0) {
            return [];
        } else {
            matches.push(match);
        }
    }
    return matches;
};

Rule.prototype.applyAt = function(delta,tuple,check) {
	var rule = this;
	//have to double check they apply
	//Q: why?
    if (check) {
        var ruleMatches=true;                
        for (var cellRowIndex=0;cellRowIndex<rule.patterns.length;cellRowIndex++) {
        	if (rule.isEllipsis[cellRowIndex]) {//if ellipsis
            	if (DoesCellRowMatchWildCard(rule.direction,rule.patterns[cellRowIndex],tuple[cellRowIndex][0],
            		tuple[cellRowIndex][1]+1, tuple[cellRowIndex][1])===false) { /* pass mink to specify */
                    ruleMatches=false;
                    break;
                }
        	} else {
            	if (DoesCellRowMatch(rule.direction,rule.patterns[cellRowIndex],tuple[cellRowIndex])===false) {
                    ruleMatches=false;
                    break;
                }
        	}
        }
        if (ruleMatches === false ) {
            return false;
        }
    }
    var result=false;
    
    //APPLY THE RULE
    var d0 = delta[0]*level.height;
    var d1 = delta[1];
    for (var cellRowIndex=0;cellRowIndex<rule.patterns.length;cellRowIndex++) {
        var preRow = rule.patterns[cellRowIndex];
        
        var currentIndex = rule.isEllipsis[cellRowIndex] ? tuple[cellRowIndex][0] : tuple[cellRowIndex];
        for (var cellIndex=0;cellIndex<preRow.length;cellIndex++) {
            var preCell = preRow[cellIndex];

            if (preCell === ellipsisPattern) {
            	var k = tuple[cellRowIndex][1];
            	currentIndex = (currentIndex+(d1+d0)*k)%level.n_tiles;
            	continue;
            }

            result = preCell.replace(rule, currentIndex) || result;

            currentIndex = (currentIndex+d1+d0)%level.n_tiles;
        }
    }

	if (verbose_logging && result){
		var ruleDirection = dirMaskName[rule.direction];
		var logString = '<font color="green">Rule <a onclick="jumpToLine(' + rule.lineNumber + ');"  href="javascript:void(0);">' + rule.lineNumber + '</a> ' + 
			ruleDirection + ' applied.</font>';
		consolePrint(logString);
	}

    return result;
};

Rule.prototype.tryApply = function() {
	var delta = dirMasksDelta[this.direction];

    //get all cellrow matches
    var matches = this.findMatches();
    if (matches.length===0) {
    	return false;
    }

    var result=false;	
	if (this.hasReplacements) {
	    var tuples = generateTuples(matches);
	    for (var tupleIndex=0;tupleIndex<tuples.length;tupleIndex++) {
	        var tuple = tuples[tupleIndex];
	        var shouldCheck=tupleIndex>0;
	        var success = this.applyAt(delta,tuple,shouldCheck);
	        result = success || result;
	    }
	}

    if (matches.length>0) {
    	this.queueCommands();
    }
    return result;
};

Rule.prototype.queueCommands = function() {
	var commands = this.commands;
	for(var i=0;i<commands.length;i++) {
		var command=commands[i];
		var already=false;
		if (level.commandQueue.indexOf(command[0])>=0) {
			continue;
		}
		level.commandQueue.push(command[0]);

		if (verbose_logging){
			var lineNumber = this.lineNumber;
			var ruleDirection = dirMaskName[this.direction];
			var logString = '<font color="green">Rule <a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);">' + lineNumber.toString() + '</a> triggers command '+command[0]+'.</font>';
			consolePrint(logString);
		}

		if (command[0]==='message') {			
			messagetext=command[1];
		}		
	}
};

function showTempMessage() {
	keybuffer=[];
	textMode=true;
	titleScreen=false;
	quittingMessageScreen=false;
	messageselected=false;
	tryPlayShowMessageSound();
	drawMessageScreen();
	canvasResize();
}

function applyRandomRuleGroup(ruleGroup) {
	var propagated=false;

	var matches=[];
	for (var ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
		var rule=ruleGroup[ruleIndex];
		var ruleMatches = rule.findMatches();
		if (ruleMatches.length>0) {
	    	var tuples  = generateTuples(ruleMatches);
	    	for (var j=0;j<tuples.length;j++) {
	    		var tuple=tuples[j];
				matches.push([ruleIndex,tuple]);
	    	}
		}		
	}

	if (matches.length===0)
	{
		return false;
	} 

	var match = matches[Math.floor(RandomGen.uniform()*matches.length)];
	var ruleIndex=match[0];
	var rule=ruleGroup[ruleIndex];
	var delta = dirMasksDelta[rule.direction];
	var tuple=match[1];
	var check=false;
	var modified = rule.applyAt(delta,tuple,check);

   	rule.queueCommands();

	return modified;
}

function applyRuleGroup(ruleGroup) {
	if (ruleGroup[0].isRandom) {
		return applyRandomRuleGroup(ruleGroup);
	}

	var loopPropagated=false;
    var propagated=true;
    var loopcount=0;
    while(propagated) {
    	loopcount++;
    	if (loopcount>200) 
    	{
    		logErrorCacheable("Got caught looping lots in a rule group :O",ruleGroup[0].lineNumber,true);
    		break;
    	}
        propagated=false;
        for (var ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
            var rule = ruleGroup[ruleIndex];            
            propagated = rule.tryApply() || propagated;
        }
        if (propagated) {
        	loopPropagated=true;
        }
    }

    return loopPropagated;
}

function applyRules(rules, loopPoint, startRuleGroupindex, bannedGroup){
    //for each rule
    //try to match it

    //when we're going back in, let's loop, to be sure to be sure
    var loopPropagated = startRuleGroupindex>0;
    var loopCount = 0;
    for (var ruleGroupIndex=startRuleGroupindex;ruleGroupIndex<rules.length;) {
    	if (bannedGroup && bannedGroup[ruleGroupIndex]) {
    		//do nothing
    	} else {
    		var ruleGroup=rules[ruleGroupIndex];
			loopPropagated = applyRuleGroup(ruleGroup) || loopPropagated;
	    }
        if (loopPropagated && loopPoint[ruleGroupIndex]!==undefined) {
        	ruleGroupIndex = loopPoint[ruleGroupIndex];
        	loopPropagated=false;
        	loopCount++;
			if (loopCount > 200) {
    			var ruleGroup=rules[ruleGroupIndex];
			   	logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0].lineNumber,true);
			   	break;
			}
        } else {
        	ruleGroupIndex++;
        	if (ruleGroupIndex===rules.length) {
        		if (loopPropagated && loopPoint[ruleGroupIndex]!==undefined) {
		        	ruleGroupIndex = loopPoint[ruleGroupIndex];
		        	loopPropagated=false;
		        	loopCount++;
					if (loopCount > 200) {
		    			var ruleGroup=rules[ruleGroupIndex];
					   	logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0].lineNumber,true);
					   	break;
					}
		        } 
        	}
        }
    }
}


//if this returns!=null, need to go back and reprocess
function resolveMovements(dir){
    var moved=true;
    while(moved){
        moved=false;
        for (var i=0;i<level.n_tiles;i++) {
        	moved = repositionEntitiesAtCell(i) || moved;
        }
    }
    var doUndo=false;

	for (var i=0;i<level.n_tiles;i++) {
		var cellMask = level.getCellInto(i,_o6);
		var movementMask = level.getMovements(i);
		if (!movementMask.iszero()) {
			var rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];
			if (rigidMovementAppliedMask !== 0) {
				movementMask.iand(rigidMovementAppliedMask);
				if (!movementMask.iszero()) {
					//find what layer was restricted
					for (var j=0;j<level.layerCount;j++) {
						var layerSection = movementMask.getshiftor(0x1f, 5*j);
						if (layerSection!==0) {
							//this is our layer!
							var rigidGroupIndexMask = level.rigidGroupIndexMask[i];
							var rigidGroupIndex = rigidGroupIndexMask.getshiftor(0x1f, 5*j);
							rigidGroupIndex--;//group indices start at zero, but are incremented for storing in the bitfield
							var groupIndex = state.rigidGroupIndex_to_GroupIndex[rigidGroupIndex];
							level.bannedGroup[groupIndex]=true;
							//backtrackTarget = rigidBackups[rigidGroupIndex];
							doUndo=true;
							break;
						}
					}
				}
			}
			for (var j=0;j<state.sfx_MovementFailureMasks.length;j++) {
				var o = state.sfx_MovementFailureMasks[j];
				var objectMask = o.objectMask;
				if (objectMask.anyBitsInCommon(cellMask)) {
					var directionMask = o.directionMask;
					if (movementMask.anyBitsInCommon(directionMask) && seedsToPlay_CantMove.indexOf(o.seed)===-1) {
						seedsToPlay_CantMove.push(o.seed);
					}
				}
			}
    	}

    	for (var j=0;j<STRIDE_MOV;j++) {
    		level.movements[j+i*STRIDE_MOV]=0;
    	}
	    level.rigidGroupIndexMask[i]=0;
	    level.rigidMovementAppliedMask[i]=0;
    }
    return doUndo;
}

var sfxCreateMask=0;
var sfxDestroyMask=0;

function calculateRowColMasks() {
	for(var i=0;i<level.mapCellContents.length;i++) {
		level.mapCellContents[i]=0;
	}

	for (var i=0;i<level.width;i++) {
		var ccc = level.colCellContents[i];
		ccc.setZero();
	}

	for (var i=0;i<level.height;i++) {
		var rcc = level.rowCellContents[i];
		rcc.setZero();
	}

	for (var i=0;i<level.width;i++) {
		for (var j=0;j<level.height;j++) {
			var index = j+i*level.height;
			var cellContents=level.getCellInto(index,_o9);
			level.mapCellContents.ior(cellContents);
			level.rowCellContents[j].ior(cellContents);
			level.colCellContents[i].ior(cellContents);
		}
	}
}

/* returns a bool indicating if anything changed */
function processInput(dir,dontCheckWin,dontModify) {
	againing = false;

	if (verbose_logging) { 
	 	if (dir===-1) {
	 		consolePrint('Turn starts with no input.')
	 	} else {
	 		consolePrint('=======================');
			consolePrint('Turn starts with input of ' + ['up','left','down','right','action'][dir]+'.');
	 	}
	}

	var bak = backupLevel();

	var playerPositions=[];
    if (dir<=4) {
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

        var i=0;
        level.bannedGroup = [];
        rigidBackups = [];
        level.commandQueue=[];
        var startRuleGroupIndex=0;
        var rigidloop=false;
        var startState = commitPreservationState();
	    sfxCreateMask=new BitVec(STRIDE_OBJ);
	    sfxDestroyMask=new BitVec(STRIDE_OBJ);

		seedsToPlay_CanMove=[];
		seedsToPlay_CantMove=[];

		calculateRowColMasks();

        do {
        //not particularly elegant, but it'll do for now - should copy the world state and check
        //after each iteration
        	rigidloop=false;
        	i++;
        	
        	if (verbose_logging){consolePrint('applying rules');}

        	applyRules(state.rules, state.loopPoint, startRuleGroupIndex, level.bannedGroup);
        	var shouldUndo = resolveMovements();

        	if (shouldUndo) {
        		rigidloop=true;
        		restorePreservationState(startState);
        		startRuleGroupIndex=0;//rigidGroupUndoDat.ruleGroupIndex+1;
        	} else {
        		if (verbose_logging){consolePrint('applying late rules');}
        		applyRules(state.lateRules, state.lateLoopPoint, 0);
        		startRuleGroupIndex=0;
        	}
        } while (i < 50 && rigidloop);

        if (i>=50) {
            consolePrint("looped through 50 times, gave up.  too many loops!");
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
        		backups.push(bak);
        		DoUndo(true);
        		return false;
        	}
        	//play player cantmove sounds here
        }

	    if (level.commandQueue.indexOf('cancel')>=0) {	
	    	if (verbose_logging) { 
	    		consolePrint('CANCEL command executed, cancelling turn.');
	    		consoleCacheDump();
			}
    		backups.push(bak);
    		DoUndo(true);
    		return false;
	    } 

	    if (level.commandQueue.indexOf('restart')>=0) {
	    	if (verbose_logging) { 
	    		consolePrint('RESTART command executed, reverting to restart state.');
	    		consoleCacheDump();
			}
    		backups.push(bak);
	    	DoRestart(true);
    		return true;
	    } 

	    if (dontModify && level.commandQueue.indexOf('win')>=0) {
	    	return true;
	    }
	    
        var modified=false;
	    for (var i=0;i<level.objects.length;i++) {
	    	if (level.objects[i]!==bak.dat[i]) {
				if (dontModify) {
	        		if (verbose_logging) {
	        			consoleCacheDump();
	        		}
	        		backups.push(bak);
	        		DoUndo(true);
					return true;
				} else {
					if (dir!==-1) {
	    				backups.push(bak);
	    			}
	    			modified=true;
	    		}
	    		break;
	    	}
	    }

		if (dontModify) {		
    		if (verbose_logging) {
    			consoleCacheDump();
    		}
			return false;
		}

        for (var i=0;i<seedsToPlay_CantMove.length;i++) {
	        	playSound(seedsToPlay_CantMove[i]);
        }

        for (var i=0;i<seedsToPlay_CanMove.length;i++) {
	        	playSound(seedsToPlay_CanMove[i]);
        }

        for (var i=0;i<state.sfx_CreationMasks.length;i++) {
        	var entry = state.sfx_CreationMasks[i];
        	if (sfxCreateMask.anyBitsInCommon(entry.objectMask)) {
	        	playSound(entry.seed);
        	}
        }

        for (var i=0;i<state.sfx_DestructionMasks.length;i++) {
        	var entry = state.sfx_DestructionMasks[i];
        	if (sfxDestroyMask.anyBitsInCommon(entry.objectMask)) {
	        	playSound(entry.seed);
        	}
        }

	    for (var i=0;i<level.commandQueue.length;i++) {
	 		var command = level.commandQueue[i];
	 		if (command.charAt(1)==='f')  {//identifies sfxN
	 			tryPlaySimpleSound(command);
	 		}  	
			if (unitTesting===false) {
				if (command==='message') {
					showTempMessage();
				}
			}
	    }

	    if (textMode===false && (dontCheckWin===undefined ||dontCheckWin===false)) {
	    	if (verbose_logging) { 
	    		consolePrint('Checking win condition.');
			}
	    	checkWin();
	    }

	    if (!winning) {
			if (level.commandQueue.indexOf('checkpoint')>=0) {
		    	if (verbose_logging) { 
		    		consolePrint('CHECKPOINT command executed, saving current state to the restart state.');
				}
				restartTarget=backupLevel();
			}	 

		    if (level.commandQueue.indexOf('again')>=0 && modified) {
		    	//first have to verify that something's changed
		    	var old_verbose_logging=verbose_logging;
		    	var oldmessagetext = messagetext;
		    	verbose_logging=false;
		    	if (processInput(-1,true,true)) {
			    	verbose_logging=old_verbose_logging;

			    	if (verbose_logging) { 
			    		consolePrint('AGAIN command executed, with changes detected - will execute another turn.');
					}

			    	againing=true;
			    	timer=0;
			    } else {		    	
			    	verbose_logging=old_verbose_logging;
					if (verbose_logging) { 
						consolePrint('AGAIN command not executed, it wouldn\'t make any changes.');
					}
			    }
			    verbose_logging=old_verbose_logging;
			    messagetext = oldmessagetext;
		    }   
		}
		    

	    level.commandQueue=[];

    }

	if (verbose_logging) {
		consoleCacheDump();
	}

	if (winning) {
		againing=false;
	}

	return modified;
}

function checkWin() {

	if (levelEditorOpened) {
		return;
	}

	if (level.commandQueue.indexOf('win')>=0) {
		consolePrint("Win Condition Satisfied");
		DoWin();
		return;
	}

	var won= false;
	if (state.winconditions.length>0)  {
		var passed=true;
		for (var wcIndex=0;wcIndex<state.winconditions.length;wcIndex++) {
			var wincondition = state.winconditions[wcIndex];
			var filter1 = wincondition[1];
			var filter2 = wincondition[2];
			var rulePassed=true;
			switch(wincondition[0]) {
				case -1://NO
				{
					for (var i=0;i<level.n_tiles;i++) {
						var cell = level.getCellInto(i,_o10);
						if ( (!filter1.bitsClearInArray(cell.data)) &&  
							 (!filter2.bitsClearInArray(cell.data)) ) {
							rulePassed=false;
							break;
						}
					}

					break;
				}
				case 0://SOME
				{
					var passedTest=false;
					for (var i=0;i<level.n_tiles;i++) {
						var cell = level.getCellInto(i,_o10);
						if ( (!filter1.bitsClearInArray(cell.data)) &&  
							 (!filter2.bitsClearInArray(cell.data)) ) {
							passedTest=true;
							break;
						}
					}
					if (passedTest===false) {
						rulePassed=false;
					}
					break;
				}
				case 1://ALL
				{
					for (var i=0;i<level.n_tiles;i++) {
						var cell = level.getCellInto(i,_o10);
						if ( (!filter1.bitsClearInArray(cell.data)) &&  
							 (filter2.bitsClearInArray(cell.data)) ) {
							rulePassed=false;
							break;
						}
					}
					break;
				}
			}
			if (rulePassed===false) {
				passed=false;
			}
		}
		won=passed;
	}

	if (won) {
		consolePrint("Win Condition Satisfied");
		DoWin();
	}
}

function DoWin() {
	if (winning) {
		return;
	}
	againing=false;
	tryPlayEndLevelSound();
	if (unitTesting) {
		nextLevel();
		return;
	}

	winning=true;
	timer=0;
}

/*
//this function isn't valid after refactoring, but also isn't used.
function anyMovements() {	
    for (var i=0;i<level.movementMask.length;i++) {
        if (level.movementMask[i]!==0) {
        	return true;
        }
    }
    return false;
}*/


function nextLevel() {
	keybuffer=[];
    againing=false;
	messagetext="";
	if (titleScreen) {
		if (titleSelection===0) {
			//new game
			curlevel=0;
		} 			
		loadLevelFromState(state,curlevel);
	} else {
		if (curlevel<(state.levels.length-1))
		{			
			curlevel++;
			textMode=false;
			titleScreen=false;
			quittingMessageScreen=false;
			messageselected=false;
			loadLevelFromState(state,curlevel);
		} else {
			curlevel=0;
			goToTitleScreen();
			tryPlayEndGameSound();
		}		
		//continue existing game
	}
	try {
		if (!!window.localStorage) {
			localStorage[document.URL]=curlevel;
		}
	} catch (ex) {

	}

	canvasResize();	
	clearInputHistory();
}

function goToTitleScreen(){
    againing=false;
	messagetext="";
	titleScreen=true;
	textMode=true;
	titleSelection=curlevel>0?1:0;
	generateTitleScreen();
}


