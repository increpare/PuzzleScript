/*
credits: 
codemirror for being an awesome web text editor
arne for the main colour palette
dock for help making others
terry for poking me to add some silly features
lots of other people
*/

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
	"..............v 0.1...............",
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
var titleInputMode=0;//1 means there are options
var titleSelection=0;
var titleSelected=false;
function generateTitleScreen()
{

	if (state.levels.length==0) {
		titleImage=intro_template;
		return;
	}

	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title.toUpperCase();
	}

	continueText = "continue";
	leftnote1 = "arrow keys to move";
	if (titleInputMode==0) {
		if (titleSelected) {
			titleImage = deepClone(titletemplate_firstgo_selected);		
		} else {
			titleImage = deepClone(titletemplate_firstgo);					
		}
	} else {
		if (titleSelection==0) {
			if (titleSelected) {
				titleImage = deepClone(titletemplate_select0);		
			} else {
				titleImage = deepClone(titletemplate_select0_selected);					
			}			
		} else {
			if (titleSelected) {
				titleImage = deepClone(titletemplate_select0);		
			} else {
				titleImage = deepClone(titletemplate_select0_selected);					
			}						
		}
	}

	for (var i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	var width = titleImage[0].length;
	var titleLength=title.length;
	var lmargin = ((width-titleLength)/2)|0;
	var rmargin = width-titleLength-lmargin;
	var row = titleImage[1];
	titleImage[1]=row.slice(0,lmargin)+title+row.slice(lmargin+title.length);
	var row = titleImage[3];
	if (state.metadata.author!==undefined) {
		var attribution="by "+state.metadata.author;
		titleImage[3]=row.slice(0,width-attribution.length-1)+attribution+row[row.length-1];			
	}

}

var state = {
	title: "2D Whale World",
	attribution: "increpare",
   	objectCount: 2,
   	metadata:[],
   	levels:[]
}

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

function drawMessageScreen() {
    var leveldat = state.levels[curlevel];
	titleMode=false;
	textMode=true;
	titleImage = deepClone(messagecontainer_template);

	for (var i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	var width = titleImage[0].length;
	var message = leveldat.message.trim();
	var messageLength=message.length;
	var lmargin = ((width-messageLength)/2)|0;
	var rmargin = width-messageLength-lmargin;
	var row = titleImage[5];
	titleImage[5]=row.slice(0,lmargin)+message+row.slice(lmargin+message.length);		
	if (quittingMessageScreen) {
		titleImage[10]=titleImage[9];
	}
}

function loadLevelFromState(state,levelindex) {	
	titleScreen=false;
	titleMode=curlevel>0?1:0;
	titleSelected=false;
	titleSelection=0;
	curlevel=levelindex;
    var leveldat = state.levels[levelindex];
    if (leveldat.message===undefined) {
    	titleMode=false;
    	textMode=false;
	    level = {
	        width: leveldat.w,
	        height: leveldat.h,
	        layerCount: leveldat.layerCount,
	        dat: leveldat.dat.concat([]),
	        movementMask: leveldat.dat.concat([]),
            rigidMovementAppliedMask: leveldat.dat.concat([]),
	        rigidGroupIndexMask: leveldat.dat.concat([])//group index
	    };
	    for (var i=0;i<level.movementMask.length;i++)
	    {
	        level.movementMask[i]=0;
	        level.rigidMovementAppliedMask[i]=0;
	        level.rigidGroupIndexMask[i]=0;
	    }

	    if ('run_rules_on_level_start' in state.metadata) {
			processInput(-1);
	    }

	    backups=[]
	    restartTarget=backupLevel();
	} else {
		drawMessageScreen();
    	canvasResize();
	}


}

function autoTickGame() {
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

/** @expose */
function mouseMove(e) {
    x = e.clientX;
    y = e.clientY;
    //window.console.log("showcoord ("+ canvas.width+","+canvas.height+") ("+x+","+y+")");
    redraw();
}

/** @expose */
function mouseOut() {
//  window.console.log("clear");
}


var backups=[];
var restartTarget;

function backupLevel() {
	return deepClone(level);
}

function setGameState(_state, command) {
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
    } else {
    	autotick=0;
    	autotickinterval=0;
    }

    if (state.metadata.key_repeat_interval!==undefined) {
		repeatinterval=state.metadata.key_repeat_interval*1000;
    } else {
    	repeatinterval=150;
    }

    switch(command[0]){
    	case "restart":
    	{
		    winning=false;
		    timer=0;
		    titleScreen=true;
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
	
    canvasResize();
}

function restoreLevel(lev) {
	level = deepClone(lev);

    for (var i=0;i<level.movementMask.length;i++)
    {
        level.movementMask[i]=0;
        level.rigidGroupIndexMask[i]=0;
        level.rigidMovementAppliedMask[i]=0;
    }
	redraw();
}

var zoomscreen=false;
var flickscreen=false;
var screenwidth=0;
var screenheight=0;


function DoRestart() {
	backups.push(backupLevel());
	restoreLevel(restartTarget);
}

function DoUndo() {
	if (backups.length>0) {
		var tobackup = backups[backups.length-1];
		restoreLevel(tobackup);
		backups = backups.splice(0,backups.length-1);		
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
    if (collision!=0) {
        return false;
    }
    var sourceMask = level.dat[positionIndex];
    var movingEntities = sourceMask&layerMask;
    level.dat[positionIndex] = sourceMask&(~layerMask);
    level.dat[targetIndex] = targetMask | movingEntities;
    return true;
}

var dirMask_random = [parseInt('00001', 2), parseInt('00010', 2), parseInt('00100', 2), parseInt('01000', 2)];
function randomDir() {
   return dirMask_random[Math.floor(Math.random() * dirMask_random.length)];
}

function repositionEntiteisAtCell(positionIndex) {
    var movementMask = level.movementMask[positionIndex];
    //assumes not zero
    //for each layer
    var moved=false;
    for (var layer=0;layer<5;layer++) {
        var layerMovement = parseInt('11111', 2) & (movementMask>>(5*layer));
        if (layerMovement!=0) {
        	var randomMask = parseInt('00101', 2);
        	if (randomMask==layerMovement) {
        		layerMovement = randomDir();
        	}
            moved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement) || moved;
        }
    }

    if (moved){
        level.movementMask[positionIndex]=0;
    } 
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

    if (checkThing(initCellMask,initMovementMask,initNonExistenceMask,initStationaryMask,movementMask,cellMask)) {
            var targetIndex = i;
            for (var j=5;j<cellRow.length;j+=5) {
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
                		for (var j2=j+5;j2<cellRow.length;j2+=5) {
                			movementMask = level.movementMask[targetIndex2];
			                cellMask = level.dat[targetIndex2];

			                ruleMovementMask= cellRow[j2+0];
			                ruleCellMask = cellRow[j2+1];
			                ruleNonExistenceMask = cellRow[j2+2];
			                ruleStationaryMask = cellRow[j2+4];

						    if (checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)) {
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


			    if (checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)) {
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

    if (checkThing(initCellMask,initMovementMask,initNonExistenceMask,initStationaryMask,movementMask,cellMask)) {
            var targetIndex = i;
            for (var j=5;j<cellRow.length;j+=5) {
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
			    if (checkThing(ruleCellMask,ruleMovementMask,ruleNonExistenceMask,ruleStationaryMask,movementMask,cellMask)) {
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

function matchCellRow(direction, cellRow) {
	var result=[];
	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

    var len=((cellRow.length/5)|0);
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
	for (var x=xmin;x<xmax;x++) {
		for (var y=ymin;y<ymax;y++) {
			var i = x*level.height+y;
			if (cellRowMatches(direction,cellRow,i))
			{
				result.push(i);
			}
		}
	}
	return result;
}


function matchCellRowWildCard(direction, cellRow) {
	var result=[];
	var xmin=0;
	var xmax=level.width;
	var ymin=0;
	var ymax=level.height;

	var len=((cellRow.length/5)|0)-1;//remove one to deal with wildcard
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
	for (var x=xmin;x<xmax;x++) {
		for (var y=ymin;y<ymax;y++) {
			var i = x*level.height+y;
			var kmax;

			switch(direction) {
		    	case 1://up
		    	{
		    		kmax=y;
		    		break;
		    	}
		    	case 2: //down 
		    	{
					kmax=ymax-y;
					break;
		    	}
		    	case 4: //left
		    	{
		    		kmax=x;
		    		break;
		    	}
		    	case 8: //right
				{
					kmax=xmax-x;	
					break;
				}
		    	default:
		    	{
		    		window.console.log("EEEP2 "+direction);
		    	}
		    }
			result = result.concat(cellRowMatchesWildCard(direction,cellRow,i,kmax))
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
//	rigidBackups:[],
//	bannedGroup:[]
}

var rigidBackups = [];//doesn't need to be backed up
var bannedGroup = [];

function commitPreservationState(ruleGroupIndex) {
	var propagationState = {
		ruleGroupIndex:ruleGroupIndex,
		//don't need to know the tuple index
		dat:level.dat.concat([]),
		levelMovementMask:level.movementMask.concat([]),
		rigidGroupIndexMask:level.rigidGroupIndexMask.concat([]),//[[mask,groupNumber]
        rigidMovementAppliedMask:level.rigidMovementAppliedMask.concat([]),
//		rigidBackups:rigidBackups.concat([]),
		bannedGroup:bannedGroup.concat([])
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
//	rigidBackups = preservationState.rigidBackups;
}

function tryApplyRule(rule,ruleGroupIndex,ruleIndex){
	var result=false;	
	var delta = dirMasksDelta[rule[0]];
    //get all cellrow matches
    var matches=[];
    for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
        var cellRow = rule[1][cellRowIndex];
        if (rule[5][cellRowIndex]) {//if ellipsis     
        	var match = matchCellRowWildCard(rule[0],cellRow);  
        } else {
        	var match = matchCellRow(rule[0],cellRow);               	
        }
        if (match.length==0) {
            return false;
        } else {
            matches.push(match);
        }
    }

    var tuples  = generateTuples(matches);
    for (var tupleIndex=0;tupleIndex<tuples.length;tupleIndex++) {
        var tuple = tuples[tupleIndex];
        //have to double check they apply
        if (tupleIndex>0) {
            var matches=true;                
            for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
            	if (rule[5][cellRowIndex]) {//if ellipsis
	            	if (cellRowMatchesWildCard(rule[0],rule[1][cellRowIndex],tuple[cellRowIndex][0],tuple[cellRowIndex][1])==false) {
	                    matches=false;
	                    break;
	                }
            	} else {
	            	if (cellRowMatches(rule[0],rule[1][cellRowIndex],tuple[cellRowIndex])==false) {
	                    matches=false;
	                    break;
	                }
            	}
            }
            if (matches ==false ) {
                continue;
            }
        }
        //APPLY THE RULE
        var rigidCommitted=false;
        for (var cellRowIndex=0;cellRowIndex<rule[1].length;cellRowIndex++) {
            var preRow = rule[1][cellRowIndex];
            var postRow = rule[2][cellRowIndex];
            
            var currentIndex = rule[5][cellRowIndex] ? tuple[cellRowIndex][0] : tuple[cellRowIndex];
            for (var cellIndex=0;cellIndex<preRow.length;cellIndex+=5) {
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

                if (postCell_Movements === randomEntityMask) {
                	var choices=[];
                	for (var i=0;i<32;i++) {
                		if  ((postCell_Objects&(1<<i))!==0) {
                			choices.push(i);
                		}
                	}
                	var rand = choices[Math.floor(Math.random() * choices.length)];
                	var n = state.idDict[rand];
                	var o = state.objects[n];
                	var layerMask = state.layerMasks[o.layer];
                	postCell_Movements = 0;
                	postCell_Objects = (1<<rand);
                	postCell_NonExistence = layerMask;
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
                curMovementMask = curMovementMask | postCell_Movements;
                curMovementMask = curMovementMask & (~postCell_StationaryMask);

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
                    level.dat[currentIndex]=curCellMask;
                    level.movementMask[currentIndex]=curMovementMask;
                }

                currentIndex = (currentIndex+delta[1]+delta[0]*level.height)%level.dat.length;
            }
        }
    }
    return result;
}


function propagateMovements(startRuleGroupindex){
        //for each rule
            //try to match it

    //when we're going back in, let's loop, to be sure to be sure
    var loopPropagated = startRuleGroupindex>0;
    for (var ruleGroupIndex=startRuleGroupindex;ruleGroupIndex<state.rules.length;) {
    	if (bannedGroup[ruleGroupIndex]) {
    		//do nothing
    	} else {
/*	    	if (state.rigidGroups[ruleGroupIndex]) {
	    		var rigid_Group_Index = state.groupNumber_to_RigidGroupIndex;
	    		if (rigidBackups[rigid_Group_Index]===undefined) {
	    			rigidBackups[rigid_Group_Index]=commitPreservationState(ruleGroupIndex);
	    		}
	    	}
*/
	        var ruleGroup=state.rules[ruleGroupIndex];

	        var propagated=true;
	        var loopcount=0;
	        while(propagated) {
	        	loopcount++;
	        	if (loopcount>50) 
	        	{
	        		window.console.log("got caught looping in a rule group :O");
	        		break;
	        	}
	            propagated=false
	            for (var ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
	                var rule = ruleGroup[ruleIndex];            
	                propagated = tryApplyRule(rule) || propagated;
	            }
	            loopPropagated = propagated || loopPropagated;
	        }
	    }
        ruleGroupIndex++;
        if (loopPropagated && state.loopPoint[ruleGroupIndex]!==undefined) {
        	ruleGroupIndex = state.loopPoint[ruleGroupIndex];
        	loopPropagated=false;
        }
    }
}

function propagateLateMovements(){
        //for each rule
            //try to match it
    for (var ruleGroupIndex=0;ruleGroupIndex<state.lateRules.length;ruleGroupIndex++) {
        var ruleGroup=state.lateRules[ruleGroupIndex];

        var propagated=true;
        while(propagated) {
            propagated=false
            for (var ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
                var rule = ruleGroup[ruleIndex];            
                propagated = tryApplyRule(rule) || propagated;
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
                 moved = repositionEntiteisAtCell(i) || moved;
            }
        }
    }
    var doUndo=false;

    for (var i=0;i<level.movementMask.length;i++) {

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
    					bannedGroup[groupIndex]=true;
    					//backtrackTarget = rigidBackups[rigidGroupIndex];
    					doUndo=true;
    					break;
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

function processInput(dir) {
	bak = backupLevel();

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
        bannedGroup = [];
        rigidBackups = [];
        var startRuleGroupIndex=0;
        var rigidloop=false;
        var startState = commitPreservationState();

        while (first || rigidloop||(anyMovements()&& (i<50))) {
        //not particularly elegant, but it'll do for now - should copy the world state and check
        //after each iteration
        	first=false;
        	rigidloop=false;
        	i++;
        	propagateMovements(startRuleGroupIndex);	
        	var shouldUndo = resolveMovements();

        	if (shouldUndo) {
        		rigidloop=true;
        		restorePreservationState(startState);
        		startRuleGroupIndex=0;//rigidGroupUndoDat.ruleGroupIndex+1;
        	} else {
        		propagateLateMovements();
        		startRuleGroupIndex=0;
        	}
        }

        if (i>=50) {
        	window.console.log("looped through 50 times, gave up.  too many loops!");
        }

	    for (var i=0;i<level.movementMask.length;i++) {
        	level.movementMask[i]=0;
        	level.rigidGroupIndexMask[i]=0;
        	level.rigidMovementAppliedMask[i]=0;
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
        		backups.push(bak);
        		DoUndo();
        		return;
        	}
        }

	    for (var i=0;i<level.dat.length;i++) {
	    	if (level.dat[i]!=bak.dat[i]) {
	    		backups.push(bak);
	    		break;
	    	}
	    }

	    checkWin();
    }

    redraw();
}

function checkWin() {
	var wincondition = state.wincondition;
	if (wincondition.length==0) {
		return;
	}
	var filter1 = wincondition[1];
	var filter2 = wincondition[2];

	var won= false;
	switch(wincondition[0]) {
		case -1://NO
		{
			won=true;
			for (var i=0;i<level.dat.length;i++) {
				var val = level.dat[i];
				if ( ((filter1&val)!==0) &&  ((filter2&val)!==0) ) {
					won=false;
					break;
				}
			}

			break;
		}
		case 0://SOME
		{
			won=false;
			for (var i=0;i<level.dat.length;i++) {
				var val = level.dat[i];
				if ( ((filter2&val)!==0) &&  ((filter1&val)!==0) ) {
					won=true;
					break;
				}
			}
			break;
		}
		case 1://ALL
		{
			won=true;
			for (var i=0;i<level.dat.length;i++) {
				var val = level.dat[i];
				if ( ((filter1&val)!==0) &&  ((filter2&val)===0) ) {
					won=false;
					break;
				}
			}
			break;
		}
	}

	if (won) {
		DoWin();
	}
}

function DoWin() {
	if (winning) {
		return;
	}

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

function saveProgress() {

}

function restoreProgress() {

}

function nextLevel() {
	if (titleScreen) {
		if (titleSelection==0) {
			//new game
			curlevel=0;
			saveProgress();
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
			saveProgress();
			loadLevelFromState(state,curlevel);
		} else {
			curlevel=0;
			saveProgress();
			goToTitleScreen();
		}
		//continue existing game
	}
	canvasResize();
}

function goToTitleScreen(){
	titleScreen=true;
	textMode=true;
	titleSelection=0;
	generateTitleScreen();
}
