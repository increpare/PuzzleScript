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

intro_template = [
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

messagecontainer_template = [
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

titletemplate_firstgo = [
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

titletemplate_select0 = [
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

titletemplate_select1 = [
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


titletemplate_firstgo_selected = [
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

titletemplate_select0_selected = [
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

titletemplate_select1_selected = [
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
	level =  {
	    width: 5,
	    height: 5,
	    layerCount: 2,
	    dat: [
	    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
	    2, 1, 2, 2, 3, 3, 1, 1, 2, 2,
	    3, 2, 1, 3, 2, 1, 3, 2, 1, 3,
	    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
	    2, 1, 2, 2, 3, 3, 1, 1, 2, 2
	    ],
	    movementMask:[
	    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
	    2, 1, 2, 2, 3, 3, 1, 1, 2, 2,
	    3, 2, 1, 3, 2, 1, 3, 2, 1, 3,
	    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
	    2, 1, 2, 2, 3, 3, 1, 1, 2, 2
	    ],
	    rigidGroupIndexMask:[],//[indexgroupNumber, masked by layer arrays]
	    rigidMovementAppliedMask:[],//[indexgroupNumber, masked by layer arrays]
	    bannedGroup:[]
	};
	generateTitleScreen();
	canvasResize();
	redraw();
}

function generateTitleScreen()
{
	titleMode=curlevel>0?1:0;
	
	if (state.levels.length==0) {
		titleImage=intro_template;
		return;
	}

	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title.toUpperCase();
	}

	if (titleMode==0) {
		if (titleSelected) {
			titleImage = deepClone(titletemplate_firstgo_selected);		
		} else {
			titleImage = deepClone(titletemplate_firstgo);					
		}
	} else {
		if (titleSelection==0) {
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
		attributionsplit = wordwrap(attribution,titleImage[0].length);
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


function loadLevelFromState(state,levelindex) {	
	forceRegenImages=true;
	titleScreen=false;
	titleMode=curlevel>0?1:0;
	titleSelected=false;
	titleSelection=0;
	curlevel=levelindex;
    againing=false;
    var leveldat = state.levels[levelindex];
    if (leveldat===null) {
    	consolePrint("Trying to access a level that doesn't exist.");
    	return;
    }
    if (leveldat.message===undefined) {
    	titleMode=0;
    	textMode=false;
	    level = {
	        width: leveldat.w,
	        height: leveldat.h,
	        layerCount: leveldat.layerCount,
	        dat: leveldat.dat.concat([]),
	        movementMask: leveldat.dat.concat([]),
            rigidMovementAppliedMask: leveldat.dat.concat([]),
	        rigidGroupIndexMask: leveldat.dat.concat([]),//group index
	        rowCellContents:[],
	        colCellContents:[],
	        mapCellContents:0
	    };

	    for (var i=0;i<level.height;i++) {
	    	level.rowCellContents[i]=0;	    	
	    }
	    for (var i=0;i<level.width;i++) {
	    	level.colCellContents[i]=0;	    	
	    }

	    for (var i=0;i<level.movementMask.length;i++)
	    {
	        level.movementMask[i]=0;
	        level.rigidMovementAppliedMask[i]=0;
	        level.rigidGroupIndexMask[i]=0;
	    }

	    backups=[]
	    restartTarget=backupLevel();

	    if ('run_rules_on_level_start' in state.metadata) {
			processInput(-1,true);
	    }

	    if (levelindex=== 0){ 
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();			
		}

	} else {
		tryPlayShowMessageSound();
		drawMessageScreen();
    	canvasResize();
	}

	clearInputs();
}

function autoTickGame() {
  pushInput("wait");
	processInput(-1);
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

//setInterval(tick, 100);

//setTimeout(redraw,100);

function tick() {
redraw();
}


function tryPlaySimpleSound(soundname) {
	if (state.sfx_Events[soundname]!==undefined) {
		var seed = state.sfx_Events[soundname];
		playSeed(seed);
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
	return level.dat.concat([]);
}

function setGameState(_state, command) {
	oldflickscreendat=[];
	timer=0;
	autotick=0;
	winning=false;
	againing=false;
    messageselected=false;

	if (command===undefined) {
		command=["restart"];
	}
	if (state.levels.length==0 && command.length>0 && command[0]==="rebuild")  {
		command=["restart"];
	}

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
    	logBetaMessage("realtime_interval is a beta feature, its behaviour may change before it ends up in launch.  I would advise against circulating this game for wider distribution before then.",true);
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


    switch(command[0]){
    	case "restart":
    	{
		    winning=false;
		    timer=0;
		    titleScreen=true;
		    tryPlayTitleSound();
		    textMode=true;
		    titleSelection=0;
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
		    titleSelection=0;
		    titleSelected=false;
		    quittingMessageScreen=false;
		    quittingTitleScreen=false;
		    messageselected=false;
		    titleMode = 0;
			loadLevelFromState(state,targetLevel);
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
				    titleSelection=0;
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
	
	clearInputs();
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

var messagetext="";
function restoreLevel(lev) {
	oldflickscreendat=[];
	level.dat=lev.concat([]);

	//width/height don't change, neither does layercount
	for (var i=0;i<level.dat.length;i++) {
		level.movementMask[i]=0;
		level.rigidMovementAppliedMask[i]=0;
		level.rigidGroupIndexMask[i]=0;
	}	

    for (var i=0;i<level.height;i++) {
    	level.rowCellContents[i]=0;	    	
    }
    for (var i=0;i<level.width;i++) {
    	level.colCellContents[i]=0;	    	
    }


    againing=false;
    messagetext="";
    level.commandQueue=[];
	redraw();
}

var zoomscreen=false;
var flickscreen=false;
var screenwidth=0;
var screenheight=0;


function DoRestart(force) {
	if (force===false && ('norestart' in state.metadata)) {
		return;
	}
	if (force===false) {
		backups.push(backupLevel());
	}
	restoreLevel(restartTarget);
	tryPlayRestartSound();

	if ('run_rules_on_level_start' in state.metadata) {
    	processInput(-1,true);
	}
	
	level.commandQueue=[];
}

function DoUndo(force) {
	if ('noundo' in state.metadata && force!==true) {
		return;
	}
	if (backups.length>0) {
		var tobackup = backups[backups.length-1];
		restoreLevel(tobackup);
		backups = backups.splice(0,backups.length-1);
		tryPlayUndoSound();
	}
}
function getPlayerPositions() {
    var result=[];
    var playerMask = state.playerMask;
    for (i=0;i<level.dat.length;i++) {
        var cellMask = level.dat[i];
        if ((playerMask&cellMask)!=0){
            result.push(i);
        }
    }
    return result;
}

function getLayersOfMask(cellMask) {
    var layers=[];
    for (var i=0;i<state.objectCount;i++) {
        if ( (cellMask&(1<<i))!=0 ){
            var n = state.idDict[i];
            var o = state.objects[n];
            layers.push(o.layer)
        }
    }
    return layers;
}

function getLayerMask(cellMask) {
    var layerMask=0;
    for (var i=0;i<state.objectCount;i++) {
        if ( (cellMask&(1<<i))!=0 ){
            var n = state.idDict[i];
            var o = state.objects[n];
            layerMask = (layerMask | (parseInt("11111",2)<<(5*o.layer)));
        }
    }
    return layerMask;
}

function moveEntitiesAtIndex(positionIndex, entityMask, dirMask) {
    var cellMask = level.dat[positionIndex];
    var overlap = entityMask&cellMask;
    var layers = getLayersOfMask(overlap);

    var movementMask = level.movementMask[positionIndex];
    for (var i=0;i<layers.length;i++) {
        var layer = layers[i];
        movementMask = (movementMask | (dirMask<< (layer*5)));
    }
    level.movementMask[positionIndex]=movementMask;
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
    var ty = ((positionIndex%level.height)|0);
    var maxx = level.width-1;
    var maxy = level.height-1;

    if ( (tx===0&&dx<0) || (tx===maxx&&dx>0) || (ty===0&&dy<0) || (ty===maxy&&dy>0)) {
    	return false;
    }

    var targetIndex = (positionIndex+delta[1]+delta[0]*level.height)%level.dat.length;

    var layerMask = state.layerMasks[layer];
    var targetMask = level.dat[targetIndex];
    var collision = targetMask&layerMask;
    var sourceMask = level.dat[positionIndex];

    if ((collision!=0) && (dirMask!=16)) {
        return false;
    }
    var movingEntities = sourceMask&layerMask;
    level.dat[positionIndex] = sourceMask&(~layerMask);
    level.dat[targetIndex] = targetMask | movingEntities;


    var colIndex=(targetIndex/level.height)|0;
	var rowIndex=(targetIndex%level.height);
    level.colCellContents[colIndex]=(level.colCellContents[colIndex]|movingEntities);
    level.rowCellContents[rowIndex]=(level.rowCellContents[rowIndex]|movingEntities);
    level.mapCellContents = (level.mapCellContents|movingEntities);

	for (var i=0;i<state.sfx_MovementMasks.length;i++) {
		var o = state.sfx_MovementMasks[i];
		var objectMask = o.objectMask;
		if ((objectMask&sourceMask)!==0) {
			var movementMask = level.movementMask[positionIndex];
			var directionMask = o.directionMask;
			if ((movementMask&directionMask)!==0 && seedsToPlay_CanMove.indexOf(o.seed)===-1) {
				seedsToPlay_CanMove.push(o.seed);
			}
		}
	}
    return true;
}

var dirMask_random = [parseInt('00001', 2), parseInt('00010', 2), parseInt('00100', 2), parseInt('01000', 2)];
function randomDir() {
  return dirMask_random[Math.floor(Math.random() * dirMask_random.length)];
}

var randomDirMask = parseInt('00101', 2);

function repositionEntitiesAtCell(positionIndex) {
    var movementMask = level.movementMask[positionIndex];
    //assumes not zero
    //for each layer
    var moved=false;
    for (var layer=0;layer<6;layer++) {    	
        var layerMovement = parseInt('11111', 2) & (movementMask>>(5*layer));
        if (layerMovement!=0) {
//        	if (randomDirMask===layerMovement) {
//        		layerMovement = randomDir();
//        	}
            var thismoved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement);
            if (thismoved) {
            	movementMask = movementMask & (~(layerMovement<<(5*layer)));
            }
            moved = thismoved || moved;
        }
    }

   	level.movementMask[positionIndex] = movementMask;            

    return moved;
}

function ruleMovementMaskAgrees(ruleMovementMask,cellMovementMask){
    if (ruleMovementMask===0 ) {
        return true;
    } else {
        return (ruleMovementMask&cellMovementMask)!==0;
    }
}

var ellipsisDirection = 1<<31;
var randomEntityMask = parseInt('00101', 2);


function cellRowMatchesWildCard_ParticularK(direction,cellRow,i,k) {
    var initMovementMask= cellRow[0];
    var initCellMask = cellRow[1];
    var initNonExistenceMask = cellRow[2];
    var initStationaryMask = cellRow[4];
    var delta = dirMasksDelta[direction];
    var movementMask = level.movementMask[i];
    var cellMask = level.dat[i];

    var allowed;
    var result=[];

    if (



			((initCellMask&cellMask) == initCellMask) &&
			((initNonExistenceMask&cellMask)==0)&&
			((initMovementMask===0?true:((initMovementMask&movementMask)!==0))) &&
			((initStationaryMask&movementMask)==0)

    	//checkThing(initCellMask,initMovementMask,initNonExistenceMask,initStationaryMask,movementMask,cellMask)
    	) {
            var targetIndex = i;
            for (var j=6;j<cellRow.length;j+=6) {
                targetIndex = (targetIndex+delta[1]+delta[0]*level.height)%level.dat.length;
                var movementMask = level.movementMask[targetIndex];
                var ruleMovementMask= cellRow[j+0];

                var cellMask = level.dat[targetIndex];
                var ruleCellMask = cellRow[j+1];
                var ruleNonExistenceMask = cellRow[j+2];
                var ruleStationaryMask = cellRow[j+4];
                if (ruleMovementMask === ellipsisDirection) {
                	//BAM inner loop time
                	//for (var k=0;k<maxk;k++) 
                	{//k defined 
                		var targetIndex2=targetIndex;
                		targetIndex2 = (targetIndex2+delta[1]*(k)+delta[0]*(k)*level.height+level.dat.length)%level.dat.length;
                		for (var j2=j+6;j2<cellRow.length;j2+=6) {
                			movementMask = level.movementMask[targetIndex2];
			                cellMask = level.dat[targetIndex2];

			                ruleMovementMask= cellRow[j2+0];
			                ruleCellMask = cellRow[j2+1];
			                ruleNonExistenceMask = cellRow[j2+2];
			                ruleStationaryMask = cellRow[j2+4];

						    if (

								((ruleCellMask&cellMask) == ruleCellMask) &&
								((ruleNonExistenceMask&cellMask)==0)&&
								((ruleMovementMask===0?true:((ruleMovementMask&movementMask)!==0))) &&
								((ruleStationaryMask&movementMask)==0)

						    	//checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)
						    	) {
						    	//good
						    } else {
						    	break;
						    }
                			targetIndex2 = (targetIndex2+delta[1]+delta[0]*level.height)%level.dat.length;
                		}

			            if (j2>=cellRow.length) {
			                result.push([i,k]);
			            }
                	}
                	break;
                }


			    if (

								((ruleCellMask&cellMask) == ruleCellMask) &&
								((ruleNonExistenceMask&cellMask)==0)&&
								((ruleMovementMask===0?true:((ruleMovementMask&movementMask)!==0))) &&
								((ruleStationaryMask&movementMask)==0)
								//checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)
								) {
                    //GOOD
                } else {
                    break;
                }
            }   
            

    }  
    return result.length>0;
}

function cellRowMatchesWildCard(direction,cellRow,i,maxk) {
    var initMovementMask= cellRow[0];
    var initCellMask = cellRow[1];
    var initNonExistenceMask = cellRow[2];
    var initStationaryMask = cellRow[4];
    var delta = dirMasksDelta[direction];
    var movementMask = level.movementMask[i];
    var cellMask = level.dat[i];

    var allowed;
    var result=[];

    if (

			((initCellMask&cellMask) == initCellMask) &&
			((initNonExistenceMask&cellMask)==0)&&
			((initMovementMask===0?true:((initMovementMask&movementMask)!==0))) &&
			((initStationaryMask&movementMask)==0)
    	//checkThing(initCellMask,initMovementMask,initNonExistenceMask,initStationaryMask,movementMask,cellMask)
    	) {
            var targetIndex = i;
            for (var j=6;j<cellRow.length;j+=6) {
                targetIndex = (targetIndex+delta[1]+delta[0]*level.height)%level.dat.length;
                var movementMask = level.movementMask[targetIndex];
                var ruleMovementMask= cellRow[j+0];

                var cellMask = level.dat[targetIndex];
                var ruleCellMask = cellRow[j+1];
                var ruleNonExistenceMask = cellRow[j+2];
                var ruleStationaryMask = cellRow[j+4];
                if (ruleMovementMask === ellipsisDirection) {
                	//BAM inner loop time
                	for (var k=0;k<maxk;k++) {
                		var targetIndex2=targetIndex;
                		targetIndex2 = (targetIndex2+delta[1]*(k)+delta[0]*(k)*level.height+level.dat.length)%level.dat.length;
                		for (var j2=j+6;j2<cellRow.length;j2+=6) {
                			movementMask = level.movementMask[targetIndex2];
			                cellMask = level.dat[targetIndex2];

			                ruleMovementMask= cellRow[j2+0];
			                ruleCellMask = cellRow[j2+1];
			                ruleNonExistenceMask = cellRow[j2+2];
			                ruleStationaryMask = cellRow[j2+4];

						    if (

								((ruleCellMask&cellMask) == ruleCellMask) &&
								((ruleNonExistenceMask&cellMask)==0)&&
								((ruleMovementMask===0?true:((ruleMovementMask&movementMask)!==0))) &&
								((ruleStationaryMask&movementMask)==0)
						    	//checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)

						    	) {
						    	//good
						    } else {
						    	break;
						    }
                			targetIndex2 = (targetIndex2+delta[1]+delta[0]*level.height)%level.dat.length;
                		}

			            if (j2>=cellRow.length) {
			                result.push([i,k]);
			            }
                	}
                	break;
                }


			    if (
						((ruleCellMask&cellMask) == ruleCellMask) &&
						((ruleNonExistenceMask&cellMask)==0)&&
						((ruleMovementMask===0?true:((ruleMovementMask&movementMask)!==0))) &&
						((ruleStationaryMask&movementMask)==0)
			    	//checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)
			    	) {
                    //GOOD
                } else {
                    break;
                }
            }   
            

    }  
    return result;
}

function checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask) {
	return ((ruleCellMask&cellMask) == ruleCellMask) &&
			((ruleNonExistenceMask&cellMask)==0)&&
			(ruleMovementMaskAgrees(ruleMovementMask,movementMask)) &&
			((ruleStationaryMask&movementMask)==0);
}

function cellRowMatches(direction,cellRow,i,k) {
    var initMovementMask= cellRow[0];
    var initCellMask = cellRow[1];
    var initNonExistenceMask = cellRow[2];
    var initStationaryMask = cellRow[4];
    var delta = dirMasksDelta[direction];
    var movementMask = level.movementMask[i];
    var cellMask = level.dat[i];

    var allowed;

    if (
			((initCellMask&cellMask) == initCellMask) &&
			((initNonExistenceMask&cellMask)==0)&&
			((initMovementMask===0?true:((initMovementMask&movementMask)!==0))) &&
			((initStationaryMask&movementMask)==0)

//    	checkThing(initCellMask,initMovementMask,initNonExistenceMask,initStationaryMask,movementMask,cellMask)

    	) {
            var targetIndex = i;
            for (var j=6;j<cellRow.length;j+=6) {
                targetIndex = (targetIndex+delta[1]+delta[0]*level.height)%level.dat.length;
                var movementMask = level.movementMask[targetIndex];
                var ruleMovementMask= cellRow[j+0];
 				if (ruleMovementMask === ellipsisDirection) {
 					//only for once off verifications
                	targetIndex = (targetIndex+delta[1]*k+delta[0]*k*level.height)%level.dat.length; 					
                }
                var cellMask = level.dat[targetIndex];

                var ruleCellMask = cellRow[j+1];
                var ruleNonExistenceMask = cellRow[j+2];
                var ruleStationaryMask = cellRow[j+4];
			    if (

			((ruleCellMask&cellMask) == ruleCellMask) &&
			((ruleNonExistenceMask&cellMask)==0)&&
			((ruleMovementMask===0?true:((ruleMovementMask&movementMask)!==0))) &&
			((ruleStationaryMask&movementMask)==0)

			    	//checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)
			    	) {
                    //GOOD
                } else {
                    break;
                }
            }   
            
            if (j>=cellRow.length) {
                return true;
            }

    }  
        return false;
}

function matchCellRow(direction, cellRow, cellRowMask) {	
	var result=[];
	if ((cellRowMask&level.mapCellContents)===0) {
		return result;
	}

	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

    var len=((cellRow.length/6)|0);

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
			if ((level.rowCellContents[y]&cellRowMask)===0) {
				continue;
			}

			for (var x=xmin;x<xmax;x++) {
				var i = x*level.height+y;
				if (cellRowMatches(direction,cellRow,i))
				{
					result.push(i);
				}
			}
		}
	} else {
		for (var x=xmin;x<xmax;x++) {
			if ((level.colCellContents[x]&cellRowMask)===0) {
				continue;
			}

			for (var y=ymin;y<ymax;y++) {
				var i = x*level.height+y;
				if (cellRowMatches(direction,cellRow,i))
				{
					result.push(i);
				}
			}
		}		
	}

	return result;
}


function matchCellRowWildCard(direction, cellRow,cellRowMask) {
	var result=[];
	if ((cellRowMask&level.mapCellContents)===0) {
		return result;
	}
	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

	var len=((cellRow.length/6)|0)-1;//remove one to deal with wildcard
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
			if ((level.rowCellContents[y]&cellRowMask)===0) {
				continue;
			}

			for (var x=xmin;x<xmax;x++) {
				var i = x*level.height+y;
				var kmax;

				switch(direction) {
			    	case 1://up
			    	{
			    		kmax=y-len+2;
			    		break;
			    	}
			    	case 2: //down 
			    	{
						kmax=level.height-(y+len)+1;
						break;
			    	}
			    	case 4: //left
			    	{
			    		kmax=x-len+2;
			    		break;
			    	}
			    	case 8: //right
					{
						kmax=level.width-(x+len)+1;	
						break;
					}
			    	default:
			    	{
			    		window.console.log("EEEP2 "+direction);
			    	}
			    }
				result.push.apply(result, cellRowMatchesWildCard(direction,cellRow,i,kmax));
			}
		}
	} else {
		for (var x=xmin;x<xmax;x++) {
			if ((level.colCellContents[x]&cellRowMask)===0) {
				continue;
			}

			for (var y=ymin;y<ymax;y++) {
				var i = x*level.height+y;
				var kmax;

				switch(direction) {
			    	case 1://up
			    	{
			    		kmax=y-len+2;
			    		break;
			    	}
			    	case 2: //down 
			    	{
						kmax=level.height-(y+len)+1;
						break;
			    	}
			    	case 4: //left
			    	{
			    		kmax=x-len+2;
			    		break;
			    	}
			    	case 8: //right
					{
						kmax=level.width-(x+len)+1;	
						break;
					}
			    	default:
			    	{
			    		window.console.log("EEEP2 "+direction);
			    	}
			    }
				result.push.apply(result, cellRowMatchesWildCard(direction,cellRow,i,kmax));
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


propagationState = {
	ruleGroupIndex:0,
	levelDat:[],
	levelMovementMask:[],
	rigidGroupIndexMask:[],//[indexgroupNumber, masked by layer arrays]
    rigidMovementAppliedMask:[],
	bannedGroup:[]
}

var rigidBackups=[]

function commitPreservationState(ruleGroupIndex) {
	var propagationState = {
		ruleGroupIndex:ruleGroupIndex,
		//don't need to know the tuple index
		dat:level.dat.concat([]),
		levelMovementMask:level.movementMask.concat([]),
		rigidGroupIndexMask:level.rigidGroupIndexMask.concat([]),//[[mask,groupNumber]
        rigidMovementAppliedMask:level.rigidMovementAppliedMask.concat([]),
//		rigidBackups:rigidBackups.concat([]),
		bannedGroup:level.bannedGroup.concat([])
	};
	rigidBackups[ruleGroupIndex]=propagationState;
	return propagationState;
}

function restorePreservationState(preservationState) {
//don't need to concat or anythign here, once something is restored it won't be used again.
	level.dat = preservationState.dat.concat([]);
	level.movementMask = preservationState.levelMovementMask.concat([]);
	level.rigidGroupIndexMask = preservationState.rigidGroupIndexMask.concat([]);
    level.rigidMovementAppliedMask = preservationState.rigidMovementAppliedMask.concat([]);
    sfxCreateMask=0;
    sfxDestroyMask=0;
//	rigidBackups = preservationState.rigidBackups;
}

function findRuleMatches(rule) {
	var matches=[];
	var cellRowMasks=rule[11];
    for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
        var cellRow = rule[1][cellRowIndex];
        if (rule[5][cellRowIndex]) {//if ellipsis     
        	var match = matchCellRowWildCard(rule[0],cellRow,cellRowMasks[cellRowIndex]);  
        } else {
        	var match = matchCellRow(rule[0],cellRow,cellRowMasks[cellRowIndex]);               	
        }
        if (match.length==0) {
            return [];
        } else {
            matches.push(match);
        }
    }
    return matches;
}

function applyRuleAt(rule,delta,tuple,check) {
	//have to double check they apply
    if (check) {
        var ruleMatches=true;                
        for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
        	if (rule[5][cellRowIndex]) {//if ellipsis
            	if (cellRowMatchesWildCard_ParticularK(rule[0],rule[1][cellRowIndex],tuple[cellRowIndex][0],tuple[cellRowIndex][1])===false) {
                    ruleMatches=false;
                    break;
                }
        	} else {
            	if (cellRowMatches(rule[0],rule[1][cellRowIndex],tuple[cellRowIndex])===false) {
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
    var rigidCommitted=false;
    for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
        var preRow = rule[1][cellRowIndex];
        var postRow = rule[2][cellRowIndex];
        
        var currentIndex = rule[5][cellRowIndex] ? tuple[cellRowIndex][0] : tuple[cellRowIndex];
        for (var cellIndex=0;cellIndex<preRow.length;cellIndex+=6) {
            var preCell_Movement = preRow[cellIndex+0];
            if (preCell_Movement === ellipsisDirection) {
            	var k = tuple[cellRowIndex][1];
            	currentIndex = (currentIndex+delta[1]*k+delta[0]*k*level.height)%level.dat.length;
            	continue;
            }
            var preCell_Objects = preRow[cellIndex+1];
            var preCell_NonExistence = preRow[cellIndex+2];
            var preCell_MoveNonExistence = preRow[cellIndex+3];
            var preCell_StationaryMask = preRow[cellIndex+4];

            
            var postCell_Movements = postRow[cellIndex+0];
			var postCell_Objects = postRow[cellIndex+1];
            var postCell_NonExistence = postRow[cellIndex+2];
            var postCell_MovementsLayerMask = postRow[cellIndex+3];
            var postCell_StationaryMask = postRow[cellIndex+4];
            var postCell_RandomEntityMask = postRow[cellIndex+5];
            var postCell_RandomDirMask = preRow[cellIndex+5];

            if (postCell_RandomEntityMask !== 0) {
            	var choices=[];
            	for (var i=0;i<32;i++) {
            		if  ((postCell_RandomEntityMask&(1<<i))!==0) {
            			choices.push(i);
            		}
            	}
              var idx = (randomEntIdxAvailable() ? 
                popRandomEntIdx() : 
                Math.floor(Math.random() * choices.length));
            	var rand = choices[idx];
              pushInput("randomEntIdx:"+idx);
            	var n = state.idDict[rand];
            	var o = state.objects[n];
            	var objectMask = state.layerMasks[o.layer];
            	var movementLayerMask = (1+2+4+8+16)<<(5*o.layer);
            	postCell_Objects = postCell_Objects | (1<<rand);
            	postCell_NonExistence = postCell_NonExistence | state.layerMasks[o.layer];
            	postCell_StationaryMask = postCell_StationaryMask | movementLayerMask;
            }
            if (postCell_RandomDirMask !== 0 ) {
            	for (var layerIndex=0;layerIndex<6;layerIndex++){
            		var layerSection = parseInt("11111",2)&(postCell_RandomDirMask>>(5*layerIndex));
            		if (layerSection!==0) {
            			var r = randomDirAvailable() ? popRandomDir() : randomDir();
                  pushInput("randomDir:"+r);
            			postCell_Movements = postCell_Movements | (r<<(5*layerIndex));
            		}
            	}
            }
            
            var curCellMask = level.dat[currentIndex];
            var curMovementMask = level.movementMask[currentIndex];

            var oldCellMask = curCellMask;
            var oldMovementMask = curMovementMask;

            //1 remove old
            curCellMask = curCellMask&(~preCell_Objects);
            curMovementMask = curMovementMask&(~preCell_Movement);
            
            //2 make way for new
            curCellMask = curCellMask&(~postCell_NonExistence);
            curMovementMask = curMovementMask&(~preCell_MoveNonExistence);

            //3 mask out old movements before adding new
            if (postCell_Movements!==0) {
            	curMovementMask = curMovementMask&(~postCell_MovementsLayerMask);
            }

            //4 add new
            curCellMask = curCellMask | postCell_Objects;
            curMovementMask = curMovementMask & (~postCell_StationaryMask);
            curMovementMask = curMovementMask | postCell_Movements;

            var rigidchange=false;
            var curRigidGroupIndexMask =0;
            var curRigidMovementAppliedMask =0;
			if (rule[7]) {
        		rigidCommitted=true;
        		var groupNumber = rule[6];
        		var rigidGroupIndex = state.groupNumber_to_RigidGroupIndex[groupNumber];  
        		rigidGroupIndex++;//don't forget to -- it when decoding :O              	
        		var rigidMask = 
        					(rigidGroupIndex) +
        					((rigidGroupIndex<< ( 1 * 5 ))) +
        					((rigidGroupIndex<< ( 2 * 5 ))) +
        					((rigidGroupIndex<< ( 3 * 5 ))) +
        					((rigidGroupIndex<< ( 4 * 5 ))) +
        					((rigidGroupIndex<< ( 5 * 5 )));
        		rigidMask = rigidMask & postCell_MovementsLayerMask;
        		curRigidGroupIndexMask = level.rigidGroupIndexMask[currentIndex];
        		curRigidMovementAppliedMask = level.rigidMovementAppliedMask[currentIndex];

        		var oldrigidGroupIndexMask = curRigidGroupIndexMask;
        		var oldRigidMovementAppliedMask = curRigidMovementAppliedMask;

        		curRigidGroupIndexMask = curRigidGroupIndexMask | rigidMask;
        		curRigidMovementAppliedMask = curRigidMovementAppliedMask | postCell_MovementsLayerMask;

        		if (oldrigidGroupIndexMask!==curRigidGroupIndexMask ||
        			oldRigidMovementAppliedMask !== curRigidMovementAppliedMask) {
        			rigidchange=true;
        		}

        	}

            //check if it's changed
            if (oldCellMask!==curCellMask || oldMovementMask!=curMovementMask || rigidchange) { 
                result=true;
                if (rigidchange) {
        			level.rigidGroupIndexMask[currentIndex] = curRigidGroupIndexMask;
        			level.rigidMovementAppliedMask[currentIndex] = curRigidMovementAppliedMask;
        		}

        		var thingsCreated = curCellMask & (~oldCellMask);
        		var thingsDestroyed = oldCellMask & (~curCellMask);
        		sfxCreateMask = sfxCreateMask | thingsCreated;
        		sfxDestroyMask = sfxDestroyMask | thingsDestroyed;

                level.dat[currentIndex]=curCellMask;
                level.movementMask[currentIndex]=curMovementMask;

                var colIndex=(currentIndex/level.height)|0;
				var rowIndex=(currentIndex%level.height);
                level.colCellContents[colIndex]=(level.colCellContents[colIndex]|curCellMask);
                level.rowCellContents[rowIndex]=(level.rowCellContents[rowIndex]|curCellMask);
                level.mapCellContents = (level.mapCellContents|curCellMask);

            } else {

            }

            currentIndex = (currentIndex+delta[1]+delta[0]*level.height)%level.dat.length;
        }
    }

	if (verbose_logging && result){
		var lineNumber = rule[3];
		var ruleDirection = dirMaskName[rule[0]];
		var logString = '<font color="green">Rule <a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);">' + lineNumber.toString() + '</a> applied.</font>';
		consolePrint(logString);
	}

    return result;
}

function tryApplyRule(rule,ruleGroupIndex,ruleIndex){
	var delta = dirMasksDelta[rule[0]];
    //get all cellrow matches
    var matches=findRuleMatches(rule);
    if (matches.length===0) {
    	return false;
    }

    var result=false;	
	if (rule[9]===false) {//if the rule has a rhs
	    var tuples  = generateTuples(matches);
	    for (var tupleIndex=0;tupleIndex<tuples.length;tupleIndex++) {
	        var tuple = tuples[tupleIndex];
	        var shouldCheck=tupleIndex>0;
	        result = applyRuleAt(rule,delta,tuple,shouldCheck) || result;
	    }
	}

    if (matches.length>0) {
    	queueCommands(rule);
    }
    return result;
}


function queueCommands(rule) {
	var commands = rule[8];
	for(var i=0;i<commands.length;i++) {
		var command=commands[i];
		var already=false;
		if (level.commandQueue.indexOf(command[0])>=0) {
			continue;
		}
		level.commandQueue.push(command[0]);
		if (command[0]==='message') {			
			messagetext=command[1];
		}		
	}
}

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
		var ruleMatches = findRuleMatches(rule);
		if (ruleMatches.length>0) {
	    	var tuples  = generateTuples(ruleMatches);
	    	for (var j=0;j<tuples.length;j++) {
	    		var tuple=tuples[j];
				matches.push([ruleIndex,tuple]);
	    	}
		}		
	}

	if (matches.length==0)
	{
		return false;
	} 

  var idx = randomRuleIdxAvailable() ? 
    popRandomRuleIdx() : 
    Math.floor(Math.random()*matches.length);
	var match = matches[idx];
  pushInput("randomRuleIdx:"+idx);
	var ruleIndex=match[0];
	var rule=ruleGroup[ruleIndex];
	var delta = dirMasksDelta[rule[0]];
	var tuple=match[1];
	var check=false;
	var modified = applyRuleAt(rule,delta,tuple,check);

   	queueCommands(rule);

	return modified;
}

function applyRuleGroup(ruleGroup) {

	var randomGroup=ruleGroup[0][10];
	if (randomGroup) {
		return applyRandomRuleGroup(ruleGroup);
	}

	var loopPropagated=false;
    var propagated=true;
    var loopcount=0;
    while(propagated) {
    	loopcount++;
    	if (loopcount>200) 
    	{
    		logErrorNoLine("got caught looping lots in a rule group :O",true);
    		break;
    	}
        propagated=false
        for (var ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
            var rule = ruleGroup[ruleIndex];            
            propagated = tryApplyRule(rule) || propagated;
        }
        if (propagated) {
        	loopPropagated=true;
        }
    }

    return loopPropagated;
}

function propagateMovements(startRuleGroupindex){
        //for each rule
            //try to match it

    //when we're going back in, let's loop, to be sure to be sure
    var loopPropagated = startRuleGroupindex>0;
    var loopCount = 0;
    for (var ruleGroupIndex=startRuleGroupindex;ruleGroupIndex<state.rules.length;) {
    	if (level.bannedGroup[ruleGroupIndex]) {
    		//do nothing
    	} else {
    		var ruleGroup=state.rules[ruleGroupIndex];
			loopPropagated = applyRuleGroup(ruleGroup) || loopPropagated;	        	        
	    }
        if (loopPropagated && state.loopPoint[ruleGroupIndex]!==undefined) {
        	ruleGroupIndex = state.loopPoint[ruleGroupIndex];
        	loopPropagated=false;
        	loopCount++;
			if (loopCount > 200) {
    			var ruleGroup=state.rules[ruleGroupIndex];
			   	logError("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0][3],true);
			   	break;
			}
        } else {
        	ruleGroupIndex++;
        	if (ruleGroupIndex===state.rules.length) {
        		if (loopPropagated && state.loopPoint[ruleGroupIndex]!==undefined) {
		        	ruleGroupIndex = state.loopPoint[ruleGroupIndex];
		        	loopPropagated=false;		        
		        	loopCount++;
					if (loopCount > 200) {
		    			var ruleGroup=state.rules[ruleGroupIndex];
					   	logError("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0][3],true);
					   	break;
					}
		        } 
        	}
        }
    }
}   


function propagateLateMovements(){
    var loopPropagated = true;
    var loopCount = 0;
    for (var ruleGroupIndex=0;ruleGroupIndex<state.lateRules.length;) {
    	if (level.bannedGroup[ruleGroupIndex]) {
    		//do nothing
    	} else {
    		var ruleGroup=state.lateRules[ruleGroupIndex];
    		var modified = applyRuleGroup(ruleGroup);

			loopPropagated = modified || loopPropagated;	        	        
	    }
        if (loopPropagated && state.lateLoopPoint[ruleGroupIndex]!==undefined) {
        	ruleGroupIndex = state.lateLoopPoint[ruleGroupIndex];
        	loopPropagated=false;
        	loopCount++;
			if (loopCount > 200) {
    			var ruleGroup=state.lateRules[ruleGroupIndex];
			   	logError("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0][3],true);
			   	break;
			}
        } else {
        	ruleGroupIndex++;
        	if (ruleGroupIndex===state.lateRules.length) {
        		if (loopPropagated && state.lateLoopPoint[ruleGroupIndex]!==undefined) {
		        	ruleGroupIndex = state.lateLoopPoint[ruleGroupIndex];
		        	loopPropagated=false;
		        	loopCount++;
					if (loopCount > 00) {
		    			var ruleGroup=state.lateRules[ruleGroupIndex];
					   	logError("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0][3],true);
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
        for (var i=0;i<level.dat.length;i++) {
            var movementMask = level.movementMask[i];
             if (movementMask!=0)
             {
                 moved = repositionEntitiesAtCell(i) || moved;
            }
        }
    }
    var doUndo=false;

    for (var i=0;i<level.movementMask.length;i++) {

    	var cellMask = level.dat[i];
    	var movementMask = level.movementMask[i];
    	if (movementMask!==0) {
    		var rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];
    		var movementMask_restricted = rigidMovementAppliedMask&movementMask;    			
    		if (movementMask_restricted!==0) {
    			//find what layer was restricted
    			for (var j=0;j<6;j++) {
    				var layerSection = parseInt("11111",2)&(movementMask_restricted>>(5*j));
    				if (layerSection!==0) {
    					//this is our layer!
    					var rigidGroupIndexMask = level.rigidGroupIndexMask[i];
    					var rigidGroupIndex = parseInt("11111",2)&(rigidGroupIndexMask>>(5*j));
    					rigidGroupIndex--;//group indices start at zero, but are incremented for storing in the bitfield
    					var groupIndex = state.rigidGroupIndex_to_GroupIndex[rigidGroupIndex];
    					level.bannedGroup[groupIndex]=true;
    					//backtrackTarget = rigidBackups[rigidGroupIndex];
    					doUndo=true;
    					break;
    				}
    			}
    		}

			for (var j=0;j<state.sfx_MovementFailureMasks.length;j++) {
				var o = state.sfx_MovementFailureMasks[j];
				var objectMask = o.objectMask;
				if ((objectMask&cellMask)!==0) {
					var directionMask = o.directionMask;
					if ((movementMask&directionMask)!==0 && seedsToPlay_CantMove.indexOf(o.seed)===-1) {
						seedsToPlay_CantMove.push(o.seed);
					}
				}
			}
    	}

        level.movementMask[i]=0;
        level.rigidGroupIndexMask[i]=0;
        level.rigidMovementAppliedMask[i]=0;
    }
    return doUndo;
}

/*
    		    			var rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];
	    		var overlap = rigidGroupIndexMask&movementMask;
    			var rigidGroupIndexMask = level.rigidGroupIndexMask[i];
    			//need to find out what the group number is;
*/

var sfxCreateMask=0;
var sfxDestroyMask=0;

function calculateRowColMasks() {
	level.mapCellContents=0;
	for (var i=0;i<level.width;i++) {
		level.colCellContents[i]=0;
	}

	for (var j=0;j<level.height;j++) {
		level.rowCellContents[i]=0;
	}

	for (var i=0;i<level.width;i++) {
		for (var j=0;j<level.height;j++) {
			var index = j+i*level.height;
			var cellContents=level.dat[index];
			level.mapCellContents = level.mapCellContents|cellContents;
			level.rowCellContents[j] = level.rowCellContents[j]|cellContents;
			level.colCellContents[i] = level.colCellContents[i]|cellContents;
		}
	}
}

function processInput(dir,dontCheckWin,dontModify) {

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
        var first=true;
        level.bannedGroup = [];
        rigidBackups = [];
        level.commandQueue=[];
        var startRuleGroupIndex=0;
        var rigidloop=false;
        var startState = commitPreservationState();
	    messagetext="";
	    sfxCreateMask=0;
	    sfxDestroyMask=0;

		seedsToPlay_CanMove=[];
		seedsToPlay_CantMove=[];

		calculateRowColMasks();

        while (first || rigidloop/*||(anyMovements()&& (i<50))*/) {
        //not particularly elegant, but it'll do for now - should copy the world state and check
        //after each iteration
        	first=false;
        	rigidloop=false;
        	i++;
        	
        	if (verbose_logging){consolePrint('applying rules');}

        	propagateMovements(startRuleGroupIndex);	
        	var shouldUndo = resolveMovements();

        	if (shouldUndo) {
        		rigidloop=true;
        		restorePreservationState(startState);
        		startRuleGroupIndex=0;//rigidGroupUndoDat.ruleGroupIndex+1;
        	} else {
        		if (verbose_logging){consolePrint('applying late rules');}
        		propagateLateMovements();
        		startRuleGroupIndex=0;
        	}
        }

        if (i>=50) {
        	window.console.log("looped through 50 times, gave up.  too many loops!");
        }

        for (var i=0;i<seedsToPlay_CantMove.length;i++) {
	        	playSeed(seedsToPlay_CantMove[i]);
        }


        if (playerPositions.length>0 && state.metadata.require_player_movement!==undefined) {
        	var somemoved=false;
        	for (var i=0;i<playerPositions.length;i++) {
        		var pos = playerPositions[i];
        		var val = level.dat[pos];
        		if ((val&state.playerMask)===0) {
        			somemoved=true;
        			break;
        		}
        	}
        	if (somemoved===false) {
	    		consolePrint('require_player_movement set, but no player movement detected, so cancelling turn.');
        		backups.push(bak);
        		DoUndo(true);
        		seedsToPlay_CanMove=[];
        		seedsToPlay_CantMove=[];
        		return;
        	}
        	//play player cantmove sounds here
        }

	    if (level.commandQueue.indexOf('cancel')>=0) {	
	    	if (verbose_logging) { 
	    		consolePrint('CANCEL command executed, cancelling turn.');
			}
    		backups.push(bak);
    		DoUndo(true);
    		seedsToPlay_CanMove=[];
    		seedsToPlay_CantMove=[];
    		redraw();
    		return;
	    } 

	    if (level.commandQueue.indexOf('restart')>=0) {
	    	if (verbose_logging) { 
	    		consolePrint('RESTART command executed, reverting to restart state.');
			}
    		backups.push(bak);
	    	DoRestart(true);	
    		seedsToPlay_CanMove=[];
    		seedsToPlay_CantMove=[];
    		redraw();   
    		return true; 	
	    } 

        for (var i=0;i<seedsToPlay_CanMove.length;i++) {
	        	playSeed(seedsToPlay_CanMove[i]);
        }

        for (var i=0;i<state.sfx_CreationMasks.length;i++) {
        	var entry = state.sfx_CreationMasks[i];
        	if ((sfxCreateMask&entry.objectMask)!==0) {
	        	playSeed(entry.seed);
        	}
        }

        for (var i=0;i<state.sfx_DestructionMasks.length;i++) {
        	var entry = state.sfx_DestructionMasks[i];
        	if ((sfxDestroyMask&entry.objectMask)!==0) {
	        	playSeed(entry.seed);
        	}
        }

	    for (var i=0;i<level.movementMask.length;i++) {
        	level.movementMask[i]=0;
        	level.rigidGroupIndexMask[i]=0;
        	level.rigidMovementAppliedMask[i]=0;
        }

        var modified=false;
	    for (var i=0;i<level.dat.length;i++) {
	    	if (level.dat[i]!==bak[i]) {

				if (dontModify) {
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
			return false;
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

	    if (level.commandQueue.indexOf('again')>=0 && modified) {

	    	var old_verbose_logging=verbose_logging;
	    	//verbose_logging=false;
	    	//first have to verify that something's changed
	    	if (processInput(-1,true,true)) {
		    	if (verbose_logging) { 
		    		consolePrint('AGAIN command executed, with changes detected: will execute another turn.');
				}

		    	againing=true;
		    	timer=0;
		    }
		    verbose_logging=old_verbose_logging;
	    }
		if (level.commandQueue.indexOf('checkpoint')>=0) {
	    	if (verbose_logging) { 
	    		consolePrint('CHECKPOINT command executed, saving current state to the restart state.');
			}
			restartTarget=backupLevel();
		}	    
	    
	    if (textMode===false && (dontCheckWin===undefined ||dontCheckWin===false)) {
	    	if (verbose_logging) { 
	    		consolePrint('Checking win condition.');
			}
	    	checkWin();
	    }

	    level.commandQueue=[];

    }

    redraw();
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
					for (var i=0;i<level.dat.length;i++) {
						var val = level.dat[i];
						if ( ((filter1&val)!==0) &&  ((filter2&val)!==0) ) {
							rulePassed=false;
							break;
						}
					}

					break;
				}
				case 0://SOME
				{
					var passedTest=false;
					for (var i=0;i<level.dat.length;i++) {
						var val = level.dat[i];
						if ( ((filter2&val)!==0) &&  ((filter1&val)!==0) ) {
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
					for (var i=0;i<level.dat.length;i++) {
						var val = level.dat[i];
						if ( ((filter1&val)!==0) &&  ((filter2&val)===0) ) {
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

function anyMovements() {	
    for (var i=0;i<level.movementMask.length;i++) {
        if (level.movementMask[i]!=0) {
        	return true;
        }
    }
    return false;
}


function nextLevel() {
	keybuffer=[];
    againing=false;
	messagetext="";
	if (titleScreen) {
		if (titleSelection==0) {
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
	localStorage[document.URL]=curlevel;
	canvasResize();	
	clearInputs();
}

function goToTitleScreen(){
    againing=false;
	messagetext="";
	titleScreen=true;
	textMode=true;
	titleSelection=0;
	generateTitleScreen();
}


