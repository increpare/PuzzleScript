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


let RandomGen = new RNG();

const intro_template = [
	"..................................",
	"..................................",
	"..................................",
	"......Puzzle Script Terminal......",
	"..............v 1.7...............",
	"..................................",
	"..................................",
	"..................................",
	".........insert cartridge.........",
	"..................................",
	"..................................",
	"..................................",
	".................................."
];

const messagecontainer_template = [
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

const titletemplate_firstgo = [
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

const titletemplate_select0 = [
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

const titletemplate_select1 = [
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


const titletemplate_firstgo_selected = [
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

const titletemplate_select0_selected = [
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

const titletemplate_select1_selected = [
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
	"................................."];

let titleImage=[];
const titleWidth=titletemplate_select1[0].length;
const titleHeight=titletemplate_select1.length;
let textMode=true;
let titleScreen=true;
let titleMode=0;//1 means there are options
let titleSelection=0;
let titleSelected=false;

function showContinueOptionOnTitleScreen(){
	return (curlevel>0||curlevelTarget!==null)&&(curlevel in state.levels);
}

function unloadGame() {
	levelEditorOpened=false;
	state=introstate;
	level = new Level(0, 5, 5, 2, null);
	level.objects = new Int32Array(0);
	generateTitleScreen();
	canvasResize();
	redraw();
}

function generateTitleScreen()
{
	titleMode=showContinueOptionOnTitleScreen()?1:0;

	if (state.levels.length===0) {
		titleImage=intro_template;
		return;
	}

	let title = "PuzzleScript Game";
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

	let noAction = 'noaction' in state.metadata;	
	let noUndo = 'noundo' in state.metadata;
	let noRestart = 'norestart' in state.metadata;
	if (noUndo && noRestart) {
		titleImage[11]="..............................................";
	} else if (noUndo) {
		titleImage[11]=".......R to restart...........................";
	} else if (noRestart) {
		titleImage[11]=".Z to undo.....................";
	}
	if (noAction) {
		titleImage[10]=".......X to select............................";
	}
	for (let i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	let width = titleImage[0].length;
	let titlelines=wordwrap(title,titleImage[0].length);
	if (state.metadata.author!==undefined){
		if ( titlelines.length>3){
			titlelines.splice(3);
			logWarning("Game title is too long to fit on screen, truncating to three lines.",state.metadata_lines.title,true);
		}
	} else {
		if ( titlelines.length>5){
			titlelines.splice(5);
			logWarning("Game title is too long to fit on screen, truncating to five lines.",state.metadata_lines.title,true);
		}

	}
	for (let i=0;i<titlelines.length;i++) {
		let titleline=titlelines[i];
		let titleLength=titleline.length;
		let lmargin = ((width-titleLength)/2)|0;
		let rmargin = width-titleLength-lmargin;
		let row = titleImage[1+i];
		titleImage[1+i]=row.slice(0,lmargin)+titleline+row.slice(lmargin+titleline.length);
	}
	if (state.metadata.author!==undefined) {
		let attribution="by "+state.metadata.author;
		let attributionsplit = wordwrap(attribution,titleImage[0].length);
		if (attributionsplit[0].length<titleImage[0].length){
			attributionsplit[0]=" "+attributionsplit[0];
		}
		if (attributionsplit.length>3){
			attributionsplit.splice(3);
			logWarning("Author list too long to fit on screen, truncating to three lines.",state.metadata_lines.author,true);
		}
		for (let i=0;i<attributionsplit.length;i++) {
			let line = attributionsplit[i]+" ";
			if (line.length>width){
				line=line.slice(0,width);
			}
			let row = titleImage[3+i];
			titleImage[3+i]=row.slice(0,width-line.length)+line;
		}
	}

}

const introstate = {
	title: "EMPTY GAME",
	attribution: "increpare",
   	objectCount: 2,
   	metadata:[],
   	levels:[],
   	bgcolor:"#000000",
   	fgcolor:"#FFFFFF"
};

let state = introstate;

function deepClone(item) {
    if (!item) { return item; } // null, undefined values check

    let types = [ Number, String, Boolean ], 
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
                let result = item.cloneNode( true );    
            } else if (!item.prototype) { // check that this is a literal
                if (item instanceof Date) {
                    result = new Date(item);
                } else {
                    // it is an object literal
                    result = {};
                    for (let i in item) {
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
    let cut = true;
 
    if (!str) { return str; }
 
    let regex = '.{1,' +width+ '}(\\s|$)' + (cut ? '|.{' +width+ '}|.+$' : '|\\S+?(\\s|$)');
 
    return str.match( RegExp(regex, 'g') );
 
}

let splitMessage=[];

function drawMessageScreen() {
	titleMode=0;
	textMode=true;
	titleImage = deepClone(messagecontainer_template);

	for (let i=0;i<titleImage.length;i++)
	{
		titleImage[i]=titleImage[i].replace(/\./g, ' ');
	}

	let emptyLineStr = titleImage[9];
	let xToContinueStr = titleImage[10];

	titleImage[10]=emptyLineStr;

	let width = titleImage[0].length;

	let message;
	if (messagetext==="") {
		let leveldat = state.levels[curlevel];
		message = leveldat.message.trim();
	} else {
		message = messagetext;
	}
	
	splitMessage = wordwrap(message,titleImage[0].length);


	let offset = 5-((splitMessage.length/2)|0);
	if (offset<0){
		offset=0;
	}

	let count = Math.min(splitMessage.length,12);
	for (let i=0;i<count;i++) {
		let m = splitMessage[i];
		let row = offset+i;	
		let messageLength=m.length;
		let lmargin = ((width-messageLength)/2)|0;
		let rmargin = width-messageLength-lmargin;
		let rowtext = titleImage[row];
		titleImage[row]=rowtext.slice(0,lmargin)+m+rowtext.slice(lmargin+m.length);		
	}

	let endPos = 10;
	if (count>=10) {
		if (count<12){
			endPos = count + 1;
		} else {
			endPos = 12;
		}
        }
	if (quittingMessageScreen) {
		titleImage[endPos]=emptyLineStr;
	} else {
		titleImage[endPos]=xToContinueStr;
	}
	
	canvasResize();
}

let loadedLevelSeed=0;

function loadLevelFromLevelDat(state,leveldat,randomseed,clearinputhistory) {	
	if (randomseed==null) {
		randomseed = (Math.random() + Date.now()).toString();
	}
	loadedLevelSeed = randomseed;
	RandomGen = new RNG(loadedLevelSeed);
	forceRegenImages=true;
	ignoreNotJustPressedAction=true;
	titleScreen=false;
	titleMode=showContinueOptionOnTitleScreen()?1:0;
	titleSelection=showContinueOptionOnTitleScreen()?1:0;
	titleSelected=false;
    againing=false;
    if (leveldat===undefined) {
    	consolePrint("Trying to access a level that doesn't exist.",true);
	goToTitleScreen();
    	return;
    }
    if (leveldat.message===undefined) {
    	titleMode=0;
    	textMode=false;
		level = leveldat.clone();
		RebuildLevelArrays();


        if (state!==undefined) {
	        if (state.metadata.flickscreen!==undefined){
	            oldflickscreendat=[
	            	0,
	            	0,
	            	Math.min(state.metadata.flickscreen[0],level.width),
	            	Math.min(state.metadata.flickscreen[1],level.height)
	            ];
	        } else if (state.metadata.zoomscreen!==undefined){
	            oldflickscreendat=[
	            	0,
	            	0,
	            	Math.min(state.metadata.zoomscreen[0],level.width),
	            	Math.min(state.metadata.zoomscreen[1],level.height)
	            ];
	        }
        }

	    backups=[]
	    restartTarget=backupLevel();
		keybuffer=[];

	    if ('run_rules_on_level_start' in state.metadata) {
			runrulesonlevelstart_phase=true;
			processInput(-1,true);
			runrulesonlevelstart_phase=false;
	    }
	} else {
		ignoreNotJustPressedAction=true;
		tryPlayShowMessageSound();
		drawMessageScreen();
    	canvasResize();
	}

	if (clearinputhistory===true){
		clearInputHistory();
	}
}

function loadLevelFromStateTarget(state,levelindex,target,randomseed) {	
    let leveldat = target;    
	curlevel=levelindex;
	curlevelTarget=target;
    if (leveldat.message===undefined) {
	    if (levelindex=== 0){ 
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();			
		}
    }
    loadLevelFromLevelDat(state,state.levels[levelindex],randomseed);
    restoreLevel(target);
    restartTarget=target;
}

function loadLevelFromState(state,levelindex,randomseed) {	
    let leveldat = state.levels[levelindex];    
	curlevel=levelindex;
	curlevelTarget=null;
    if (leveldat!==undefined && leveldat.message===undefined) {
	    if (levelindex=== 0){ 
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();			
		}
    }
    loadLevelFromLevelDat(state,leveldat,randomseed);
}

let sprites = [
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
if (titleMode>0){
	titleSelection=1;
}

canvasResize();

function tryPlaySimpleSound(soundname) {
	if (state.sfx_Events[soundname]!==undefined) {
		let seed = state.sfx_Events[soundname];
		playSound(seed,true);
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

function tryPlayCancelSound() {
	tryPlaySimpleSound("cancel");
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

let backups=[];
let restartTarget;

function backupLevel() {
	let ret = {
		dat : new Int32Array(level.objects),
		width : level.width,
		height : level.height,
		oldflickscreendat: oldflickscreendat.concat([])
	};
	return ret;
}

function level4Serialization() {
	let ret = {
		dat : Array.from(level.objects),
		width : level.width,
		height : level.height,
		oldflickscreendat: oldflickscreendat.concat([])
	};
	return ret;
}



function setGameState(_state, command, randomseed) {
	
	if (_state===undefined) {
		_state=introstate;
		return;
	}
	oldflickscreendat=[];
	timer=0;
	autotick=0;
	winning=false;
	againing=false;
    messageselected=false;
    STRIDE_MOV=_state.STRIDE_MOV;
    STRIDE_OBJ=_state.STRIDE_OBJ;
    
    sfxCreateMask=new BitVec(STRIDE_OBJ);
    sfxDestroyMask=new BitVec(STRIDE_OBJ);

	if (command===undefined) {
		command=["restart"];
	}
	if ((state.levels.length===0 || _state.levels.length===0) && command.length>0 && command[0]==="rebuild")  {
		command=["restart"];
	}
	if (randomseed===undefined) {
		randomseed=null;
	}
	RandomGen = new RNG(randomseed);

	state = _state;

    if (command[0]!=="rebuild"){
    	backups=[];
    }
    //set sprites
    sprites = [];
    for (let n in state.objects) {
        if (state.objects.hasOwnProperty(n)) {
            let object = state.objects[n];
            let sprite = {
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
		    titleSelection=showContinueOptionOnTitleScreen()?1:0;
		    titleSelected=false;
		    quittingMessageScreen=false;
		    quittingTitleScreen=false;
		    messageselected=false;
		    titleMode = 0;
		    if (showContinueOptionOnTitleScreen()) {
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
		case "loadFirstNonMessageLevel":{
			for (let i=0;i<state.levels.length;i++){
				if (state.levels[i].hasOwnProperty("message")){
					continue;
				}
				let targetLevel = i;
				curlevel=targetLevel;
				curlevelTarget=null;
			    winning=false;
			    timer=0;
			    titleScreen=false;
			    textMode=false;
			    titleSelection=showContinueOptionOnTitleScreen()?1:0;
			    titleSelected=false;
			    quittingMessageScreen=false;
			    quittingTitleScreen=false;
			    messageselected=false;
			    titleMode = 0;
				loadLevelFromState(state,targetLevel,randomseed);
				break;
			}
			break;	
		}
		case "loadLevel":
		{
			let targetLevel = command[1];
			curlevel=targetLevel;
			curlevelTarget=null;
		    winning=false;
		    timer=0;
		    titleScreen=false;
		    textMode=false;
		    titleSelection=showContinueOptionOnTitleScreen()?1:0;
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
			let targetLine = command[1];
			for (let i=state.levels.length-1;i>=0;i--) {
				let level= state.levels[i];
				if(level.lineNumber<=targetLine+1) {
					curlevel=i;
					curlevelTarget=null;
				    winning=false;
				    timer=0;
				    titleScreen=false;
				    textMode=false;
				    titleSelection=showContinueOptionOnTitleScreen()?1:0;
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


	if (state.sounds.length==0){
		killAudioButton();
	} else {
		showAudioButton();
	}
	
}

function RebuildLevelArrays() {
	level.movements = new Int32Array(level.n_tiles * STRIDE_MOV);

    level.rigidMovementAppliedMask = [];
    level.rigidGroupIndexMask = [];
	level.rowCellContents = [];
	level.rowCellContents_Movements = [];
	level.colCellContents = [];
	level.colCellContents_Movements = [];
	level.mapCellContents = new BitVec(STRIDE_OBJ);
	level.mapCellContents_Movements = new BitVec(STRIDE_MOV);

	//I have these to avoid dynamic allocation - I generate 3 because why not, 
	//but according to my tests I never seem to call this while a previous copy is still in scope
	_movementVecs = [new BitVec(STRIDE_MOV),new BitVec(STRIDE_MOV),new BitVec(STRIDE_MOV)];
	_rigidVecs = [new BitVec(STRIDE_MOV),new BitVec(STRIDE_MOV),new BitVec(STRIDE_MOV)];

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
	

    for (let i=0;i<level.height;i++) {
    	level.rowCellContents[i]=new BitVec(STRIDE_OBJ);	    	
    }
    for (let i=0;i<level.width;i++) {
    	level.colCellContents[i]=new BitVec(STRIDE_OBJ);	    	
    }

    for (let i=0;i<level.height;i++) {
    	level.rowCellContents_Movements[i]=new BitVec(STRIDE_MOV);	    	
    }
    for (let i=0;i<level.width;i++) {
    	level.colCellContents_Movements[i]=new BitVec(STRIDE_MOV);	    	
    }

    for (let i=0;i<level.n_tiles;i++)
    {
        level.rigidMovementAppliedMask[i]=new BitVec(STRIDE_MOV);
        level.rigidGroupIndexMask[i]=new BitVec(STRIDE_MOV);
    }
}

let messagetext="";

function applyDiff(diff, level_objects) {

	let index=0;
	
	while (index<diff.dat.length){
		let start_index = diff.dat[index];
		let copy_length = diff.dat[index+1];
		if (copy_length===0){
			break;//tail of buffer is all 0s
		}
		for (let j=0;j<copy_length;j++){
			level_objects[start_index+j]=diff.dat[index+2+j];
		}
		index += 2 + copy_length;
	}
}

function unconsolidateDiff(before,after) {

	// If before is not a diff, return it, otherwise generate a complete 'before' 
	// state from the 'after' state and the 'diff' (remember, the diffs are all 
	// backwards...).
	if (!before.hasOwnProperty("diff")) {
		return before;
	}

	let after_objects = new Int32Array(after.dat);
	applyDiff(before, after_objects);

	return {
		dat: after_objects,
		width: before.width,
		height: before.height,
		oldflickscreendat: before.oldflickscreendat
	}
}

function restoreLevel(lev) {
	let diffing = lev.hasOwnProperty("diff");

	oldflickscreendat=lev.oldflickscreendat.concat([]);

	if (diffing){
		applyDiff(lev, level.objects);
	} else {	
		level.objects = new Int32Array(lev.dat);
	}

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

		for (let i=0;i<level.n_tiles;i++) {
			level.movements[i]=0;
			level.rigidMovementAppliedMask[i].setZero();
			level.rigidGroupIndexMask[i].setZero();
		}	

	    for (let i=0;i<level.height;i++) {
	    	let rcc = level.rowCellContents[i];
	    	rcc.setZero();
	    }
	    for (let i=0;i<level.width;i++) {
	    	let ccc = level.colCellContents[i];
	    	ccc.setZero();
	    }
	}

    againing=false;
    level.commandQueue=[];
    level.commandQueueSourceRules=[];
}

var zoomscreen=false;
var flickscreen=false;
var screenwidth=0;
var screenheight=0;

//compresses 'before' into diff
function consolidateDiff(before,after){
	if (before.width !== after.width || before.height!==after.height || before.dat.length!==after.dat.length){
		return before;
	}
	if (before.hasOwnProperty("diff")||after.hasOwnProperty("diff")){
		return before;
	}
	//only generate diffs if level size is bigger than this
	if (before.dat.length<1024){
		return before;
	}
	//diff structure: repeating ( start,length, [ data ] )
	let result = new Int32Array(128);
	let position=0;
	let chain=false;
	let chain_start_idx_in_diff=-1;
	let before_dat = before.dat;
	let after_dat = after.dat;
	for (let i=0;i<before_dat.length;i++){
		if (chain===false){
			if (before_dat[i]!==after_dat[i]){
				chain=true;
				chain_start_idx_in_diff = position;

				if (result.length<position+4){
					let doubled = new Int32Array(2*result.length);
					doubled.set(result);
					result = doubled;
				}

				result[position+0]=i;
				result[position+1]=1;
				result[position+2]=before_dat[i];
				position+=3;
			}
		} else {
			if (before_dat[i]!==after_dat[i]){
				
				if (position+1>=result.length){
					if (result.length<position+4){
						let doubled = new Int32Array(2*result.length);
						doubled.set(result);
						result = doubled;
					}	
				}
				result[chain_start_idx_in_diff+1]++;
				result[position]=before_dat[i];
				position++;
			} else {
				chain=false;
			}
		}
	}
	return {		
		diff : true,
		dat : result,
		width : before.width,
		height : before.height,
		oldflickscreendat: before.oldflickscreendat
	}
}

function addUndoState(state){
	backups.push(state);
	if(backups.length>2 && !backups[backups.length-1].hasOwnProperty("diff")){
		backups[backups.length-3]=consolidateDiff(backups[backups.length-3],backups[backups.length-2]);
	}
}

function DoRestart(force) {
	if (restarting===true){
		return;
	}
	if (force!==true && ('norestart' in state.metadata)) {
		return;
	}
	
	if (againing){
		DoUndo(force,true);
	}
	restarting=true;
	if (force!==true) {
		addUndoState(backupLevel());
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
	level.commandQueueSourceRules=[];
	restarting=false;
}

function backupDiffers(){
	if (backups.length==0){
		return true;
	}
	let bak = backups[backups.length-1];

	if (bak.hasOwnProperty("diff")){
		return bak.dat.length!==0 && bak.dat[1]!==0;//if it's empty or if it's all 0s
	} else {
		for (let i=0;i<level.objects.length;i++) {
			if (level.objects[i]!==bak.dat[i]) {
				return true;
			}
		}
		return false;
	}
}

function DoUndo(force,ignoreDuplicates) {
	if ((!levelEditorOpened)&&('noundo' in state.metadata && force!==true)) {
		return;
	}
	if (verbose_logging) {
		consolePrint("--- undoing ---",true);
	}

	if (ignoreDuplicates){
		while (backupDiffers()==false){
			backups.pop();
		}
	}

	if (backups.length>0) {
		let torestore = backups[backups.length-1];
		restoreLevel(torestore);
		backups = backups.splice(0,backups.length-1);
		if (! force) {
			tryPlayUndoSound();
		}
	}
}

function getPlayerPositions() {
    let result=[];
    let playerMask = state.playerMask;
    for (let i=0;i<level.n_tiles;i++) {
        level.getCellInto(i,_o11);
        if (playerMask.anyBitsInCommon(_o11)) {
            result.push(i);
        }
    }
    return result;
}

function getLayersOfMask(cellMask) {
    let layers=[];
    for (let i=0;i<state.objectCount;i++) {
        if (cellMask.get(i)) {
            let n = state.idDict[i];
            let o = state.objects[n];
            layers.push(o.layer)
        }
    }
    return layers;
}

// this function is used to unroll loops in parallel from bitvec - it returns a string
// representation of the javascript unrolled code
function UNROLL(command, array_size){
	var toks = command.split(" ");
	var result=""
	for (let i=0;i<array_size;i++) {
		result+=`${toks[0]}.data[${i}] ${toks[1]} ${toks[2]}.data[${i}];\n`;
	}
	return result;
}

function generate_moveEntitiesAtIndex(OBJECT_SIZE,MOVEMENT_SIZE){	
	var fn = `
    let cellMask = level.getCell(positionIndex);
	${UNROLL("cellMask &= entityMask",OBJECT_SIZE)}
    let layers = getLayersOfMask(cellMask);

    let movementMask = level.getMovements(positionIndex);
    for (let i=0;i<layers.length;i++) {
    	movementMask.ishiftor(dirMask, 5 * layers[i]);
    }
    level.setMovements(positionIndex, movementMask);

	const colIndex=(positionIndex/level.height)|0;
	const rowIndex=(positionIndex%level.height);
	${UNROLL("level.colCellContents_Movements[colIndex] |= movementMask",MOVEMENT_SIZE)}
	${UNROLL("level.rowCellContents_Movements[rowIndex] |= movementMask",MOVEMENT_SIZE)}
	${UNROLL("level.mapCellContents_Movements |= movementMask",MOVEMENT_SIZE)}
	`
	var func = new Function("level","positionIndex", "entityMask", "dirMask", fn);
	return func;
}


function startMovement(dir) {
	let movedany=false;
    let playerPositions = getPlayerPositions();
    for (let i=0;i<playerPositions.length;i++) {
        let playerPosIndex = playerPositions[i];
        state.moveEntitiesAtIndex(level,playerPosIndex,state.playerMask,dir);
    }
    return playerPositions;
}

let dirMasksDelta = {
     1:[0,-1],//up
     2:[0,1],//'down'  : 
     4:[-1,0],//'left'  : 
     8:[1,0],//'right' : 
     15:[0,0],//'?' : 
     16:[0,0],//'action' : 
     3:[0,0]//'no'
};

let dirMaskName = {
     1:'up',
     2:'down'  ,
     4:'left'  , 
     8:'right',  
     15:'?' ,
     16:'action',
     3:'no'
};

let seedsToPlay_CanMove=[];
let seedsToPlay_CantMove=[];

function repositionEntitiesOnLayer(positionIndex,layer,dirMask) 
{
    let delta = dirMasksDelta[dirMask];

    let dx = delta[0];
    let dy = delta[1];
    let tx = ((positionIndex/level.height)|0);
    let ty = ((positionIndex%level.height));
    let maxx = level.width-1;
    let maxy = level.height-1;

    if ( (tx===0&&dx<0) || (tx===maxx&&dx>0) || (ty===0&&dy<0) || (ty===maxy&&dy>0)) {
    	return false;
    }

    let targetIndex = (positionIndex+delta[1]+delta[0]*level.height);

    let layerMask = state.layerMasks[layer];
    let targetMask = level.getCellInto(targetIndex,_o7);
    let sourceMask = level.getCellInto(positionIndex,_o8);

    if (layerMask.anyBitsInCommon(targetMask) && (dirMask!=16)) {
        return false;
    }

	for (let i=0;i<state.sfx_MovementMasks[layer].length;i++) {
		let o = state.sfx_MovementMasks[layer][i];
		let objectMask = o.objectMask;
		if (objectMask.anyBitsInCommon(sourceMask)) {
			let movementMask = level.getMovements(positionIndex);
			let directionMask = o.directionMask;
			if (movementMask.anyBitsInCommon(directionMask) && seedsToPlay_CanMove.indexOf(o.seed)===-1) {
				seedsToPlay_CanMove.push(o.seed);
			}
		}
	}

    let movingEntities = sourceMask.clone();
    sourceMask.iclear(layerMask);
    movingEntities.iand(layerMask);
    targetMask.ior(movingEntities);

    level.setCell(positionIndex, sourceMask);
    level.setCell(targetIndex, targetMask);
	
    let colIndex=(targetIndex/level.height)|0;
	let rowIndex=(targetIndex%level.height);
	
    level.colCellContents[colIndex].ior(movingEntities);
    level.rowCellContents[rowIndex].ior(movingEntities);
	//corresponding movement stuff in setmovements
    return true;
}

function repositionEntitiesAtCell(positionIndex) {
    let movementMask = level.getMovements(positionIndex);
    if (movementMask.iszero())
        return false;

    let moved=false;
    for (let layer=0;layer<level.layerCount;layer++) {
        let layerMovement = movementMask.getshiftor(0x1f, 5*layer);
        if (layerMovement!==0) {
            let thismoved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement);
            if (thismoved) {
                movementMask.ishiftclear(layerMovement, 5*layer);
                moved = true;
            }
        }
    }

   	level.setMovements(positionIndex, movementMask);

    return moved;
}

let ellipsisPattern = ['ellipsis'];


function Rule(rule) {
	this.direction = rule[0]; 		/* direction rule scans in */
	this.patterns = rule[1];		/* lists of CellPatterns to match */
	this.hasReplacements = rule[2];
	this.lineNumber = rule[3];		/* rule source for debugging */
	this.ellipsisCount = rule[4];		/* number of ellipses present */
	this.groupNumber = rule[5];		/* execution group number of rule */
	this.isRigid = rule[6];
	this.commands = rule[7];		/* cancel, restart, sfx, etc */
	this.isRandom = rule[8];
	this.cellRowMasks = rule[9];
	this.cellRowMasks_Movements = rule[10];
	this.ruleMask = new BitVec(STRIDE_OBJ);
	for (let m of this.cellRowMasks) {
		this.ruleMask.ior(m);
	}

	/*I tried out doing a ruleMask_movements as well along the lines of the above,
	but it didn't help at all - I guess because almost every tick there are movements 
	somewhere on the board - move filtering works well at a row/col level, but is pretty 
	useless (or worse than useless) on a boardwide level*/

	this.cellRowMatches = [];
	for (let i=0;i<this.patterns.length;i++) {
		this.cellRowMatches.push(this.generateCellRowMatchesFunction(this.patterns[i],this.ellipsisCount[i]));
	}
	/* TODO: eliminate isRigid, groupNumber, isRandom
	from this class by moving them up into a RuleGroup class */

	this.findMatches = this.generateFindMatchesFunction();
}


Rule.prototype.generateCellRowMatchesFunction = function(cellRow,ellipsisCount)  {
	if (ellipsisCount===0) {
		let cr_l = cellRow.length;

		/*
		hard substitute in the first one - if I substitute in all of them, firefox chokes.
		*/
		let fn = "";
		let mul = STRIDE_OBJ === 1 ? '' : '*'+STRIDE_OBJ;	
		for (let i = 0; i < STRIDE_OBJ; ++i) {
			fn += 'let cellObjects' + i + ' = objects[i' + mul + (i ? '+'+i: '') + '];\n';
		}
		mul = STRIDE_MOV === 1 ? '' : '*'+STRIDE_MOV;
		for (let i = 0; i < STRIDE_MOV; ++i) {
			fn += 'let cellMovements' + i + ' = movements[i' + mul + (i ? '+'+i: '') + '];\n';
		}
		fn += "return "+cellRow[0].generateMatchString('0_');// cellRow[0].matches(i)";
		for (let cellIndex=1;cellIndex<cr_l;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches(i+"+cellIndex+"*d, objects, movements)";
		}
		fn+=";";

		if (fn in matchCache) {
			return matchCache[fn];
		}
		//console.log(fn.replace(/\s+/g, ' '));
		return matchCache[fn] = new Function("cellRow","i", 'd', 'objects', 'movements',fn);
	} else if (ellipsisCount===1){
		let cr_l = cellRow.length;

		let fn = `let result = [];
if(cellRow[0].matches(i, objects, movements)`;
		let cellIndex=1;
		for (;cellRow[cellIndex]!==ellipsisPattern;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches(i+"+cellIndex+"*d, objects, movements)";
		}
		cellIndex++;
		fn+=`) {
	for (let k=kmin;k<kmax;k++) {
		if(cellRow[`+cellIndex+`].matches((i+d*(k+`+(cellIndex-1)+`)), objects, movements)`;
		cellIndex++;
		for (;cellIndex<cr_l;cellIndex++) {
			fn+="&&cellRow["+cellIndex+"].matches((i+d*(k+"+(cellIndex-1)+")), objects, movements)";			
		}
		fn+=`){
			result.push([i,k]);
		}
	}
}
`;		
		fn+="return result;"


		if (fn in matchCache) {
			return matchCache[fn];
		}
		//console.log(fn.replace(/\s+/g, ' '));
		return matchCache[fn] = new Function("cellRow","i","kmax","kmin", 'd', "objects", "movements",fn);
	} else { //ellipsisCount===2
		let cr_l = cellRow.length;

		let ellipsis_index_1=-1;
		let ellipsis_index_2=-1;
		for (let cellIndex=0;cellIndex<cr_l;cellIndex++) {
			if (cellRow[cellIndex]===ellipsisPattern) {
				if (ellipsis_index_1===-1) {
					ellipsis_index_1=cellIndex;
				} else {
					ellipsis_index_2=cellIndex;
					break;
				}
			}
		}

		let fn = `let result = [];
if(cellRow[0].matches(i, objects, movements)`;

		for (let idx=1;idx<ellipsis_index_1;idx++) {
			fn+="&&cellRow["+idx+"].matches(i+"+idx+"*d, objects, movements)";
		}
		fn+=") {\n";

		//try match middle part
		fn+=`
	for (let k1=k1min;k1<k1max;k1++) {
		if(cellRow[`+(ellipsis_index_1+1)+`].matches((i+d*(k1+`+(ellipsis_index_1+1-1)+`)), objects, movements)`;
		for (let idx=ellipsis_index_1+2;idx<ellipsis_index_2;idx++) {
			fn+="&&cellRow["+idx+"].matches((i+d*(k1+"+(idx-1)+")), objects, movements)";			
		}
		fn+="		){\n";
		//try match right part

		fn+=`
			for (let k2=k2min;k1+k2<kmax && k2<k2max;k2++) {
				if(cellRow[`+(ellipsis_index_2+1)+`].matches((i+d*(k1+k2+`+(ellipsis_index_2+1-2)+`)), objects, movements)`;
		for (let idx=ellipsis_index_2+2;idx<cr_l;idx++) {
			fn+="&&cellRow["+idx+"].matches((i+d*(k1+k2+"+(idx-2)+")), objects, movements)";			
		}
		fn+=`
				){
					result.push([i,k1,k2]);
				}
			}
		}
	}			
}	
return result;`;


		if (fn in matchCache) {
			return matchCache[fn];
		}
		//console.log(fn.replace(/\s+/g, ' '));
		return matchCache[fn] = new Function("cellRow","i","kmax","kmin", "k1max","k1min","k2max","k2min", 'd', "objects", "movements",fn);

	}
}


let STRIDE_OBJ = 1;
let STRIDE_MOV = 1;

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
	let fn = "(true";
	for (let i = 0; i < Math.max(STRIDE_OBJ, STRIDE_MOV); ++i) {
		const co = 'cellObjects' + i;
		const cm = 'cellMovements' + i;
		const op = this.objectsPresent.data[i];
		const om = this.objectsMissing.data[i];
		const mp = this.movementsPresent.data[i];
		const mm = this.movementsMissing.data[i];
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
	for (let j = 0; j < this.anyObjectsPresent.length; j++) {
		fn += "\t\t&& (0";
		for (let i = 0; i < STRIDE_OBJ; ++i) {
			const aop = this.anyObjectsPresent[j].data[i];
			if (aop)
				fn += "|(cellObjects" + i + "&" + aop + ")";
		}
		fn += ")";
	}
	fn += '\t)';
	return fn;
}

CellPattern.prototype.generateMatchFunction = function() {
	let fn = '';
	let mul = STRIDE_OBJ === 1 ? '' : '*'+STRIDE_OBJ;	
	for (let i = 0; i < STRIDE_OBJ; ++i) {
		fn += '\tconst cellObjects' + i + ' = objects[i' + mul + (i ? '+'+i: '') + '];\n';
	}
	mul = STRIDE_MOV === 1 ? '' : '*'+STRIDE_MOV;
	for (let i = 0; i < STRIDE_MOV; ++i) {
		fn += '\tconst cellMovements' + i + ' = movements[i' + mul + (i ? '+'+i: '') + '];\n';
	}
	fn += "return " + this.generateMatchString()+';';
	if (fn in matchCache) {
		return matchCache[fn];
	}
	//console.log(fn.replace(/\s+/g, ' '));
	return matchCache[fn] = new Function("i", "objects", "movements", fn);
}

let _o1,_o2,_o2_5,_o3,_o4,_o5,_o6,_o7,_o8,_o9,_o10,_o11,_o12;
let _m1,_m2,_m3;

CellPattern.prototype.replace = function(rule, currentIndex) {
    let replace = this.replacement;
    if (replace === null) {
        return false;
    }

    let replace_RandomEntityMask = replace.randomEntityMask;
    let replace_RandomDirMask = replace.randomDirMask;

	const OBJECT_SIZE = replace.objectsSet.data.length;
	const MOVEMENT_SIZE = replace.movementsSet.data.length;


    // Inline the BitVec cloning operations directly on the Int32Arrays
    for (let i = 0; i < OBJECT_SIZE; i++) {
        _o1.data[i] = replace.objectsSet.data[i];
        _o2.data[i] = replace.objectsClear.data[i];
	}
	for (let i = 0; i < MOVEMENT_SIZE; i++) {
        _m1.data[i] = replace.movementsSet.data[i];
        _m2.data[i] = replace.movementsClear.data[i] | replace.movementsLayerMask.data[i];
    }

    if (!replace_RandomEntityMask.iszero()) {
        let choices = [];
        for (let i = 0; i < 32 * STRIDE_OBJ; i++) {
            if (replace_RandomEntityMask.get(i)) {
                choices.push(i);
            }
        }
        let rand = choices[Math.floor(RandomGen.uniform() * choices.length)];
        let n = state.idDict[rand];
        let o = state.objects[n];
        
        // Inline ibitset(rand)
        _o1.data[rand >>> 5] |= 1 << (rand & 31);
        
        // Inline ior(state.layerMasks[o.layer])
        const layerMask = state.layerMasks[o.layer].data;
        for (let i = 0; i < OBJECT_SIZE; i++) {
            _o2.data[i] |= layerMask[i];
        }
        
        // Inline ishiftor(0x1f, 5 * o.layer)
        const shift = 5 * o.layer;
        const wordIndex = shift >>> 5;
        const bitOffset = shift & 31;
        if (bitOffset === 0) {
            _m2.data[wordIndex] |= 0x1f;
        } else {
            _m2.data[wordIndex] |= (0x1f << bitOffset);
        }
    }
    if (!replace_RandomDirMask.iszero()) {
        for (let layerIndex = 0; layerIndex < level.layerCount; layerIndex++) {
            if (replace_RandomDirMask.get(5 * layerIndex)) {
                let randomDir = Math.floor(RandomGen.uniform() * 4);
                _m1.ibitset(randomDir + 5 * layerIndex);
            }
        }
    }

    let curCellMask = level.getCellInto(currentIndex, _o2_5);
    let curMovementMask = level.getMovements(currentIndex);

    // Inline cloning for oldCellMask and oldMovementMask
    for (let i = 0; i < OBJECT_SIZE; i++) {
        _o3.data[i] = curCellMask.data[i];
	}
	for (let i = 0; i < MOVEMENT_SIZE; i++) {
        _m3.data[i] = curMovementMask.data[i];
    }

    curCellMask.iclear(_o2);
    curCellMask.ior(_o1);

    curMovementMask.iclear(_m2);
    curMovementMask.ior(_m1);

    let rigidchange=false;
    let curRigidGroupIndexMask =0;
    let curRigidMovementAppliedMask =0;
    if (rule.isRigid) {
        let rigidGroupIndex = state.groupNumber_to_RigidGroupIndex[rule.groupNumber];
        rigidGroupIndex++;//don't forget to -- it when decoding :O
        let rigidMask = new BitVec(STRIDE_MOV);
        for (let layer = 0; layer < level.layerCount; layer++) {
            rigidMask.ishiftor(rigidGroupIndex, layer * 5);
        }
        rigidMask.iand(replace.movementsLayerMask);
        curRigidGroupIndexMask = level.rigidGroupIndexMask[currentIndex] || new BitVec(STRIDE_MOV);
        curRigidMovementAppliedMask = level.rigidMovementAppliedMask[currentIndex] || new BitVec(STRIDE_MOV);

        if (!rigidMask.bitsSetInArray(curRigidGroupIndexMask.data) &&
            !replace.movementsLayerMask.bitsSetInArray(curRigidMovementAppliedMask.data) ) {
            curRigidGroupIndexMask.ior(rigidMask);
            curRigidMovementAppliedMask.ior(replace.movementsLayerMask);
            rigidchange=true;

        }
    }

    let result = false;

    //check if it's changed
    if (!_o3.equals(curCellMask) || !_m3.equals(curMovementMask) || rigidchange) { 
        result=true;
        if (rigidchange) {
            level.rigidGroupIndexMask[currentIndex] = curRigidGroupIndexMask;
            level.rigidMovementAppliedMask[currentIndex] = curRigidMovementAppliedMask;
        }

        let created = curCellMask.cloneInto(_o4);
        created.iclear(_o3);
        sfxCreateMask.ior(created);
        let destroyed = _o3.cloneInto(_o5);
        destroyed.iclear(curCellMask);
        sfxDestroyMask.ior(destroyed);

        level.setCell(currentIndex, curCellMask);
        level.setMovements(currentIndex, curMovementMask);

        let colIndex=(currentIndex/level.height)|0;
        let rowIndex=(currentIndex%level.height);
        level.colCellContents[colIndex].ior(curCellMask);
        level.rowCellContents[rowIndex].ior(curCellMask);
        level.mapCellContents.ior(curCellMask);

    }

    return result;
}


function matchCellRow(direction, cellRowMatch, cellRow, cellRowMask,cellRowMask_Movements,d) {	
	let result=[];
	
	if ((!cellRowMask.bitsSetInArray(level.mapCellContents.data))||
	(!cellRowMask_Movements.bitsSetInArray(level.mapCellContents_Movements.data))) {
		return result;
	}

	let xmin=0;
	let xmax=level.width;
	let ymin=0;
	let ymax=level.height;

    let len=cellRow.length;

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

    const horizontal=direction>2;
    if (horizontal) {
		for (let y=ymin;y<ymax;y++) {
			if (!cellRowMask.bitsSetInArray(level.rowCellContents[y].data) 
			|| !cellRowMask_Movements.bitsSetInArray(level.rowCellContents_Movements[y].data)) {
				continue;
			}

			for (let x=xmin;x<xmax;x++) {
				const i = x*level.height+y;
				if (cellRowMatch(cellRow,i,d, level.objects, level.movements))
				{
					result.push(i);
				}
			}
		}
	} else {
		for (let x=xmin;x<xmax;x++) {
			if (!cellRowMask.bitsSetInArray(level.colCellContents[x].data)
			|| !cellRowMask_Movements.bitsSetInArray(level.colCellContents_Movements[x].data)) {
				continue;
			}

			for (let y=ymin;y<ymax;y++) {
				const i = x*level.height+y;
				if (cellRowMatch(	cellRow,i, d, level.objects, level.movements))
				{
					result.push(i);
				}
			}
		}		
	}

	return result;
}


function matchCellRowWildCard(direction, cellRowMatch, cellRow,cellRowMask,cellRowMask_Movements,d,wildcardCount) {
	let result=[];
	if ((!cellRowMask.bitsSetInArray(level.mapCellContents.data))
	|| (!cellRowMask_Movements.bitsSetInArray(level.mapCellContents_Movements.data))) {
		return result;
	}
	
	let xmin=0;
	let xmax=level.width;
	let ymin=0;
	let ymax=level.height;

	let len=cellRow.length-wildcardCount;//remove one to deal with wildcard
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

    const horizontal=direction>2;
    if (horizontal) {
		for (let y=ymin;y<ymax;y++) {
			if (!cellRowMask.bitsSetInArray(level.rowCellContents[y].data)
			|| !cellRowMask_Movements.bitsSetInArray(level.rowCellContents_Movements[y].data) ) {
				continue;
			}

			for (let x=xmin;x<xmax;x++) {
				const i = x*level.height+y;
				let kmax;

				if (direction === 4) { //left
					kmax=x-len+2;
				} else if (direction === 8) { //right
					kmax=level.width-(x+len)+1;	
				} else {
					window.console.log("EEEP2 "+direction);					
				}

				if (wildcardCount===1) {
					result.push.apply(result, cellRowMatch(cellRow,i,kmax,0, d, level.objects, level.movements));
				} else {
					result.push.apply(result, cellRowMatch(cellRow,i,kmax,0,kmax,0,kmax,0, d, level.objects, level.movements));
				}
			}
		}
	} else {
		for (let x=xmin;x<xmax;x++) {
			if (!cellRowMask.bitsSetInArray(level.colCellContents[x].data)
			|| !cellRowMask_Movements.bitsSetInArray(level.colCellContents_Movements[x].data)) {
				continue;
			}

			for (let y=ymin;y<ymax;y++) {
				const i = x*level.height+y;
				let kmax;

				if (direction === 2) { // down
					kmax=level.height-(y+len)+1;
				} else if (direction === 1) { // up
					kmax=y-len+2;					
				} else {
					window.console.log("EEEP2 "+direction);
				}
				if (wildcardCount===1) {
					result.push.apply(result, cellRowMatch(cellRow,i,kmax,0, d, level.objects, level.movements));
				} else {
					result.push.apply(result, cellRowMatch(cellRow,i,kmax,0, kmax,0, kmax,0, d, level.objects, level.movements));
				}
			}
		}		
	}

	return result;
}

function generateTuples(lists) {
    let tuples=[[]];

    for (let i=0;i<lists.length;i++)
    {
        const row = lists[i];
        const newtuples=[];
        for (let j=0;j<row.length;j++) {
            let valtoappend = row[j];
            for (let k=0;k<tuples.length;k++) {
                const tuple=tuples[k];
                const newtuple = tuple.concat([valtoappend]);
                newtuples.push(newtuple);
            }
        }
        tuples=newtuples;
    }
    return tuples;
}



Rule.prototype.findMatches = function() {	
	if ( ! this.ruleMask.bitsSetInArray(level.mapCellContents.data) )
		return [];

	const d = level.delta_index(this.direction)

	let matches=[];
	const cellRowMasks=this.cellRowMasks;
	const cellRowMasks_Movements=this.cellRowMasks_Movements;
    for (let cellRowIndex=0;cellRowIndex<this.patterns.length;cellRowIndex++) {
        const cellRow = this.patterns[cellRowIndex];
        const matchFunction = this.cellRowMatches[cellRowIndex];
		let match;
        if (this.ellipsisCount[cellRowIndex]===1) {//if ellipsis     
        	match = matchCellRowWildCard(this.direction,matchFunction,cellRow,cellRowMasks[cellRowIndex],cellRowMasks_Movements[cellRowIndex],d,this.ellipsisCount[cellRowIndex]);  
        } else  if (this.ellipsisCount[cellRowIndex]===0) {
        	match = matchCellRow(this.direction,matchFunction,cellRow,cellRowMasks[cellRowIndex],cellRowMasks_Movements[cellRowIndex],d);               	
        } else { // ellipsiscount===2
        	match = matchCellRowWildCard(this.direction,matchFunction,cellRow,cellRowMasks[cellRowIndex],cellRowMasks_Movements[cellRowIndex],d,this.ellipsisCount[cellRowIndex]);  
		}
        if (match.length===0) {
            return [];
        } else {
            matches.push(match);
        }
    }
    return matches;
};

Rule.prototype.directional = function(){
	//Check if other rules in its rulegroup with the same line number.
	for (let i=0;i<state.rules.length;i++){
		const rg = state.rules[i];
		let copyCount=0;
		for (let j=0;j<rg.length;j++){
			if (this.lineNumber===rg[j].lineNumber){
				copyCount++;
			}
			if (copyCount>1){
				return true;
			}
		}
	}

    return false;
}

Rule.prototype.applyAt = function(level,tuple,check,delta) {
	//have to double check they apply 
	//(cf test ellipsis bug: rule matches two candidates, first replacement invalidates second)
	if (check)
	{
		for (let cellRowIndex=0; cellRowIndex<this.patterns.length; cellRowIndex++)
		{
			if (this.ellipsisCount[cellRowIndex]===1)
			{
				if ( this.cellRowMatches[cellRowIndex](
						this.patterns[cellRowIndex], 
						tuple[cellRowIndex][0], 
						tuple[cellRowIndex][1]+1, 
							tuple[cellRowIndex][1], 
						delta, level.objects, level.movements
					).length == 0 )
					return false
			} else if (this.ellipsisCount[cellRowIndex]===2){
				if ( this.cellRowMatches[cellRowIndex](
					this.patterns[cellRowIndex], 
						tuple[cellRowIndex][0],  
						tuple[cellRowIndex][1]+tuple[cellRowIndex][2]+1, 
							tuple[cellRowIndex][1]+tuple[cellRowIndex][2], 
						tuple[cellRowIndex][1]+1, 
							tuple[cellRowIndex][1],  
						tuple[cellRowIndex][2]+1, 
							tuple[cellRowIndex][2], 
							delta, level.objects, level.movements
						).length == 0 )
					return false
			} else {
				if ( ! this.cellRowMatches[cellRowIndex](
					this.patterns[cellRowIndex], 
						tuple[cellRowIndex], 
						delta, level.objects, level.movements
						) )
				return false
			}
		}
	}


    let result=false;
	let anyellipses=false;

    //APPLY THE RULE
    for (let cellRowIndex=0;cellRowIndex<this.patterns.length;cellRowIndex++) {
        const preRow = this.patterns[cellRowIndex];
    	let ellipse_index=0;

        let currentIndex = this.ellipsisCount[cellRowIndex]>0 ? tuple[cellRowIndex][0] : tuple[cellRowIndex];
        for (let cellIndex=0;cellIndex<preRow.length;cellIndex++) {
            const preCell = preRow[cellIndex];

            if (preCell === ellipsisPattern) {
            	const k = tuple[cellRowIndex][1+ellipse_index];
				ellipse_index++;
				anyellipses=true;
            	currentIndex += delta*k;
            	continue;
            }

            result = preCell.replace(this, currentIndex) || result;

            currentIndex += delta;
        }
    }

	if (verbose_logging && result){
		let ruleDirection = dirMaskName[this.direction];
		if (!this.directional()){
			ruleDirection="";
		}

		let inspect_ID =  addToDebugTimeline(level,this.lineNumber);
		let gapMessage="";
		
		let logString = `<font color="green">Rule <a onclick="jumpToLine(${this.lineNumber});"  href="javascript:void(0);">${this.lineNumber}</a> ${ruleDirection} applied${gapMessage}.</font>`;
		consolePrint(logString,false,this.lineNumber,inspect_ID);
		
	}

    return result;
};

Rule.prototype.tryApply = function(level) {
	const delta = level.delta_index(this.direction);

    //get all cellrow matches
    let matches = this.findMatches(level);
    if (matches.length===0) {
    	return false;
    }

    let result=false;	
	if (this.hasReplacements) {
	    let tuples = generateTuples(matches);
	    for (let tupleIndex=0;tupleIndex<tuples.length;tupleIndex++) {
	        let tuple = tuples[tupleIndex];
	        let shouldCheck=tupleIndex>0;
	        let success = this.applyAt(level,tuple,shouldCheck,delta);
	        result = success || result;
	    }
	}

    if (matches.length>0) {
    	this.queueCommands();
    }
    return result;
};

Rule.prototype.queueCommands = function() {
	
	if (this.commands.length==0){
		return;
	}

	//commandQueue is an array of strings, message.commands is an array of array of strings (For messagetext parameter), so I search through them differently
	let preexisting_cancel=level.commandQueue.indexOf("cancel")>=0;
	let preexisting_restart=level.commandQueue.indexOf("restart")>=0;
	
	let currule_cancel = false;
	let currule_restart = false;
	for (let i=0;i<this.commands.length;i++){
		let cmd = this.commands[i][0];
		if (cmd==="cancel"){
			currule_cancel=true;
		} else if (cmd==="restart"){
			currule_restart=true;
		}
	}

	//priority cancel > restart > everything else
	//if cancel is the queue from other rules, ignore everything
	if (preexisting_cancel){
		return;
	}
	//if restart is in the queue from other rules, only apply if there's a cancel present here
	if (preexisting_restart && !currule_cancel){
		return;
	}

	//if you are writing a cancel or restart, clear the current queue
	if (currule_cancel || currule_restart){
		level.commandQueue=[];
        level.commandQueueSourceRules=[];
		messagetext="";
	}

	for(let i=0;i<this.commands.length;i++) {
		const command=this.commands[i];
		let already=false;
		if (level.commandQueue.indexOf(command[0])>=0) {
			continue;
		}
		level.commandQueue.push(command[0]);
		level.commandQueueSourceRules.push(this);

		if (verbose_logging){
			const lineNumber = this.lineNumber;
			const ruleDirection = dirMaskName[this.direction];
			const logString = '<font color="green">Rule <a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);">' + lineNumber.toString() + '</a> triggers command '+command[0]+'.</font>';
			consolePrint(logString,false,lineNumber,null);
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
	ignoreNotJustPressedAction=true;
	tryPlayShowMessageSound();
	drawMessageScreen();
	canvasResize();
}

function processOutputCommands(commands) {
	for (let i=0;i<commands.length;i++) {
		let command = commands[i];
		if (command.charAt(1)==='f')  {//identifies sfxN
			tryPlaySimpleSound(command);
		}
		if (unitTesting===false) {
			if (command==='message') {
				showTempMessage();
			}
		}
	}
}

function applyRandomRuleGroup(level,ruleGroup) {
	let propagated=false;

	let matches=[];
	for (let ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
		let rule=ruleGroup[ruleIndex];
		let ruleMatches = rule.findMatches(level);
		if (ruleMatches.length>0) {
	    	let tuples  = generateTuples(ruleMatches);
	    	for (let j=0;j<tuples.length;j++) {
	    		let tuple=tuples[j];
				matches.push([ruleIndex,tuple]);
	    	}
		}		
	}

	if (matches.length===0)
	{
		return false;
	} 

	let match = matches[Math.floor(RandomGen.uniform()*matches.length)];
	let ruleIndex=match[0];
	let rule=ruleGroup[ruleIndex];
	let tuple=match[1];
	let check=false;
	const delta = level.delta_index(rule.direction)
	let modified = rule.applyAt(level,tuple,check,delta);

   	rule.queueCommands();

	return modified;
}


function applyRuleGroup(ruleGroup) {
	if (ruleGroup[0].isRandom) {
		return applyRandomRuleGroup(level,ruleGroup);
	}

	let loopPropagated=false;
    let propagated=true;
    let loopcount=0;
	let nothing_happened_counter = -1;
    while(propagated) {
    	loopcount++;
    	if (loopcount>200) 
    	{
    		logErrorCacheable("Got caught looping lots in a rule group :O",ruleGroup[0].lineNumber,true);
    		break;
    	}
        propagated=false;

        for (let ruleIndex=0;ruleIndex<ruleGroup.length;ruleIndex++) {
            let rule = ruleGroup[ruleIndex];     
			if (rule.tryApply(level)){
				propagated=true;
				nothing_happened_counter=0;//why am I resetting to 1 rather than 0? because I've just verified that applications of the current rule are exhausted
			} else {
				nothing_happened_counter++;
			}
			if ( nothing_happened_counter === ruleGroup.length)
				break;
        }
        if (propagated) {
        	loopPropagated=true;
			
			if (verbose_logging){
				debugger_turnIndex++;
				addToDebugTimeline(level,-2);//pre-movement-applied debug state
			}
        }
    }

    return loopPropagated;
}

function applyRules(rules, loopPoint, startRuleGroupindex, bannedGroup){
    //for each rule
    //try to match it

    //when we're going back in, let's loop, to be sure to be sure
    let loopPropagated = startRuleGroupindex>0;
    let loopCount = 0;
    for (let ruleGroupIndex=startRuleGroupindex;ruleGroupIndex<rules.length;) {
    	if (bannedGroup && bannedGroup[ruleGroupIndex]) {
    		//do nothing
    	} else {
    		let ruleGroup=rules[ruleGroupIndex];
			loopPropagated = applyRuleGroup(ruleGroup) || loopPropagated;
	    }
        if (loopPropagated && loopPoint[ruleGroupIndex]!==undefined) {
        	ruleGroupIndex = loopPoint[ruleGroupIndex];
        	loopPropagated=false;
        	loopCount++;
			if (loopCount > 200) {
    			let ruleGroup=rules[ruleGroupIndex];
			   	logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0].lineNumber,true);
			   	break;
			}
			
			if (verbose_logging){
				debugger_turnIndex++;
				addToDebugTimeline(level,-2);//pre-movement-applied debug state
			}
        } else {
        	ruleGroupIndex++;
        	if (ruleGroupIndex===rules.length) {
        		if (loopPropagated && loopPoint[ruleGroupIndex]!==undefined) {
		        	ruleGroupIndex = loopPoint[ruleGroupIndex];
		        	loopPropagated=false;
		        	loopCount++;
					if (loopCount > 200) {
		    			let ruleGroup=rules[ruleGroupIndex];
					   	logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", ruleGroup[0].lineNumber,true);
					   	break;
					}
		        } 
        	}
			
			if (verbose_logging){
				debugger_turnIndex++;
				addToDebugTimeline(level,-2);//pre-movement-applied debug state
			}
        }
    }
}


//if this returns!=null, need to go back and reprocess
function resolveMovements(level, bannedGroup){
    let moved=true;
    while(moved){
        moved=false;
        for (let i=0;i<level.n_tiles;i++) {
        	moved = repositionEntitiesAtCell(i) || moved;
        }
    }
    let doUndo=false;

	//Search for any rigidly-caused movements remaining
	for (let i=0;i<level.n_tiles;i++) {
		let cellMask = level.getCellInto(i,_o6);
		let movementMask = level.getMovements(i);
		if (!movementMask.iszero()) {
			let rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];
			if (!rigidMovementAppliedMask.iszero()) {
				movementMask.iand(rigidMovementAppliedMask);
				if (!movementMask.iszero()) {
					//find what layer was restricted
					for (let j=0;j<level.layerCount;j++) {
						let layerSection = movementMask.getshiftor(0x1f, 5*j);
						if (layerSection!==0) {
							//this is our layer!
							let rigidGroupIndexMask = level.rigidGroupIndexMask[i];
							let rigidGroupIndex = rigidGroupIndexMask.getshiftor(0x1f, 5*j);
							rigidGroupIndex--;//group indices start at zero, but are incremented for storing in the bitfield
							let groupIndex = state.rigidGroupIndex_to_GroupIndex[rigidGroupIndex];
							if (bannedGroup[groupIndex]!==true){
								bannedGroup[groupIndex]=true
								//backtrackTarget = rigidBackups[rigidGroupIndex];
								doUndo=true;
							}
							break;
						}
					}
				}
			}
			for (let j=0;j<state.sfx_MovementFailureMasks.length;j++) {
				let o = state.sfx_MovementFailureMasks[j];
				let objectMask = o.objectMask;
				if (objectMask.anyBitsInCommon(cellMask)) {
					let directionMask = o.directionMask;
					if (movementMask.anyBitsInCommon(directionMask) && seedsToPlay_CantMove.indexOf(o.seed)===-1) {
						seedsToPlay_CantMove.push(o.seed);
					}
				}
			}
    	}

    	for (let j=0;j<STRIDE_MOV;j++) {
    		level.movements[j+i*STRIDE_MOV]=0;
    	}
	    level.rigidGroupIndexMask[i].setZero();
	    level.rigidMovementAppliedMask[i].setZero();
    }
    return doUndo;
}

let sfxCreateMask=null;
let sfxDestroyMask=null;

function calculateRowColMasks() {
	for(let i=0;i<level.mapCellContents.data.length;i++) {
		level.mapCellContents.data[i]=0;
		level.mapCellContents_Movements.data[i]=0;	
	}

	for (let i=0;i<level.width;i++) {
		let ccc = level.colCellContents[i];
		ccc.setZero();
		let ccc_Movements = level.colCellContents_Movements[i];
		ccc_Movements.setZero();
	}

	for (let i=0;i<level.height;i++) {
		let rcc = level.rowCellContents[i];
		rcc.setZero();
		let rcc_Movements = level.rowCellContents_Movements[i];
		rcc_Movements.setZero();
	}

	for (let i=0;i<level.width;i++) {
		for (let j=0;j<level.height;j++) {
			let index = j+i*level.height;
			let cellContents=level.getCellInto(index,_o9);
			level.mapCellContents.ior(cellContents);
			level.rowCellContents[j].ior(cellContents);
			level.colCellContents[i].ior(cellContents);

			
			let mapCellContents_Movements=level.getMovementsInto(index,_m1);
			level.mapCellContents_Movements.ior(mapCellContents_Movements);
			level.rowCellContents_Movements[j].ior(mapCellContents_Movements);
			level.colCellContents_Movements[i].ior(mapCellContents_Movements);
		}
	}
}
/* returns a bool indicating if anything changed */
function processInput(dir, dontDoWin, dontModify) {
	againing = false;

	let bak = backupLevel();
	let inputindex = dir;
	let playerPositions = [];

	if (verbose_logging) { 
		debugger_turnIndex++;
		addToDebugTimeline(level, -2); // pre-movement-applied debug state
	}

	if (dir >= 0) {
		switch (dir) {
			case 0: // up
				dir = parseInt('00001', 2);
				break;
			case 1: // left
				dir = parseInt('00100', 2);
				break;
			case 2: // down
				dir = parseInt('00010', 2);
				break;
			case 3: // right
				dir = parseInt('01000', 2);
				break;
			case 4: // action
				dir = parseInt('10000', 2);
				break;
		}
		playerPositions = startMovement(dir);
	}

	if (verbose_logging) {
		consolePrint('Applying rules');
		let inspect_ID = addToDebugTimeline(level, -1);
		if (dir === -1) {
			consolePrint(`Turn starts with no input.`, false, null, inspect_ID);
		} else {
			consolePrint(`Turn starts with input of ${['up', 'left', 'down', 'right', 'action'][inputindex]}.`, false, null, inspect_ID);
		}
	}

	let bannedGroup = [];
	level.commandQueue = [];
	level.commandQueueSourceRules = [];
	let startRuleGroupIndex = 0;
	let rigidloop = false;
	const startState = {
		objects: new Int32Array(level.objects),
		movements: new Int32Array(level.movements),
		rigidGroupIndexMask: level.rigidGroupIndexMask.concat([]),
		rigidMovementAppliedMask: level.rigidMovementAppliedMask.concat([]),
		commandQueue: [],
		commandQueueSourceRules: []
	};
	sfxCreateMask.setZero();
	sfxDestroyMask.setZero();
	seedsToPlay_CanMove = [];
	seedsToPlay_CantMove = [];
	calculateRowColMasks();
	let alreadyResolved = [];

	let i = 0;
	do {
		rigidloop = false;
		i++;

		//everything outside of these two lines in this loop is rigid-body nonsense
		applyRules(state.rules, state.loopPoint, startRuleGroupIndex, bannedGroup);
		let shouldUndo = resolveMovements(level, bannedGroup);

		if (shouldUndo) {
			rigidloop = true;

			// trackback
			if (IDE) {
				let newBannedGroups = [];
				for (let key in bannedGroup) {
					if (!alreadyResolved.includes(key)) {
						newBannedGroups.push(key);
						alreadyResolved.push(key);
					}
				}
				let bannedLineNumbers = newBannedGroups.map(rgi => state.rules[rgi][0].lineNumber);
				let ts = bannedLineNumbers.length > 1 ? "lines " : "line ";
				ts += bannedLineNumbers.map(ln => `<a onclick="jumpToLine(${ln});" href="javascript:void(0);">${ln}</a>`).join(", ");
				consolePrint(`Rigid movement application failed in rule-Group starting from ${ts}, and will be disabled in resimulation. Rolling back...`);
			}
			level.objects = new Int32Array(startState.objects);
			level.movements = new Int32Array(startState.movements);
			level.rigidGroupIndexMask = startState.rigidGroupIndexMask.concat([]);
			level.rigidMovementAppliedMask = startState.rigidMovementAppliedMask.concat([]);
			level.commandQueue = startState.commandQueue.concat([]);
			level.commandQueueSourceRules = startState.commandQueueSourceRules.concat([]);
			sfxCreateMask.setZero();
			sfxDestroyMask.setZero();

			if (verbose_logging && rigidloop && i > 0) {
				consolePrint('Relooping through rules because of rigid.');
				debugger_turnIndex++;
				addToDebugTimeline(level, -2); // pre-movement-applied debug state
			}

			startRuleGroupIndex = 0;
		} else {
			if (verbose_logging) {
				let eof_idx = debug_visualisation_array[debugger_turnIndex].length + 1;
				let inspect_ID = addToDebugTimeline(level, eof_idx);
				consolePrint(`Processed movements.`, false, null, inspect_ID);

				if (state.lateRules.length > 0) {
					debugger_turnIndex++;
					addToDebugTimeline(level, -2); // pre-movement-applied debug state
					consolePrint('Applying late rules');
				}
			}
			applyRules(state.lateRules, state.lateLoopPoint, 0);
			startRuleGroupIndex = 0;
		}
	} while (i < 50 && rigidloop);

	if (i >= 50) {
		consolePrint("Looped through 50 times, gave up.  too many loops!");
	}

	if (playerPositions.length > 0 && state.metadata.require_player_movement !== undefined) {
		let somemoved = false;
		for (let i = 0; i < playerPositions.length; i++) {
			let pos = playerPositions[i];
			let val = level.getCell(pos);
			if (state.playerMask.bitsClearInArray(val.data)) {
				somemoved = true;
				break;
			}
		}
		if (somemoved === false) {
			if (verbose_logging) {
				consolePrint('require_player_movement set, but no player movement detected, so cancelling turn.');
				consoleCacheDump();
			}
			addUndoState(bak);
			DoUndo(true, false);
			messagetext = "";
			textMode = false;
			return false;
		}
	}

	// Factorized command-queue processing
	let modified = processCommandQueue(bak, dontModify, dontDoWin, inputindex);

	if (verbose_logging) {
		consoleCacheDump();
	}
	if (winning) {
		againing = false;
	}
	
	return modified;
}

function playSounds(seedsToPlay_CantMove, seedsToPlay_CanMove, sfx_CreationMasks, sfx_DestructionMasks, sfxCreateMask, sfxDestroyMask) {
	for (let i = 0; i < seedsToPlay_CantMove.length; i++) {
		playSound(seedsToPlay_CantMove[i]);
	}
	for (let i = 0; i < seedsToPlay_CanMove.length; i++) {
		playSound(seedsToPlay_CanMove[i]);
	}
	for (let i = 0; i < sfx_CreationMasks.length; i++) {
		let entry = sfx_CreationMasks[i];
		if (sfxCreateMask.anyBitsInCommon(entry.objectMask)) {
			playSound(entry.seed);
		}
	}
	for (let i = 0; i < sfx_DestructionMasks.length; i++) {
		let entry = sfx_DestructionMasks[i];
		if (sfxDestroyMask.anyBitsInCommon(entry.objectMask)) {
			playSound(entry.seed);
		}
	}
}

function processCommandQueue(bak, dontModify, dontDoWin, inputDir) {
	// Process CANCEL command
	const cancelIndex = level.commandQueue.indexOf('cancel');
	if (cancelIndex >= 0) {
		if (verbose_logging) {
			consoleCacheDump();
			let cancelRule = level.commandQueueSourceRules[cancelIndex];
			consolePrintFromRule('CANCEL command executed, cancelling turn.', cancelRule, true);
		}
		if (!dontModify) {
			processOutputCommands(level.commandQueue);
		}
		let commandsLeft = level.commandQueue.length > 1;
		addUndoState(bak);
		DoUndo(true, false);
		tryPlayCancelSound();
		return commandsLeft;
	}

	// Process RESTART command
	const restartIndex = level.commandQueue.indexOf('restart');
	if (restartIndex >= 0) {
		if (verbose_logging && runrulesonlevelstart_phase) {
			let r = level.commandQueueSourceRules[restartIndex];
			logWarning(
				'A "restart" command is being triggered in the "run_rules_on_level_start" section of level creation, which would cause an infinite loop if it was actually triggered, but it\'s being ignored.',
				r.lineNumber,
				true
			);
		}
		if (verbose_logging) {
			let r = level.commandQueueSourceRules[restartIndex];
			consolePrintFromRule('RESTART command executed, reverting to restart state.', r);
			consoleCacheDump();
		}
		if (!dontModify) {
			processOutputCommands(level.commandQueue);
		}
		addUndoState(bak);
		if (!dontModify) {
			DoRestart(true);
		}
	}

	// Check for modifications comparing level.objects to backup
	let modified = false;
	for (let i = 0; i < level.objects.length; i++) {
		if (level.objects[i] !== bak.dat[i]) {
			if (dontModify) {
				if (verbose_logging) {
					consoleCacheDump();
				}
				addUndoState(bak);
				DoUndo(true, false);
				return true;
			} else {
				if (inputDir !== -1) {
					addUndoState(bak);
				} else if (backups.length > 0) {
					backups[backups.length - 1] = unconsolidateDiff(backups[backups.length - 1], bak);
				}
				modified = true;
			}
			break;
		}
	}

	// When dontModify is set, also check for win or restart commands.
	if (dontModify && (level.commandQueue.includes('win') || level.commandQueue.includes('restart'))) {
		return true;
	}

	if (!dontModify) {
		// Play failure/movement sounds as needed.
		playSounds(seedsToPlay_CantMove, seedsToPlay_CanMove, state.sfx_CreationMasks, state.sfx_DestructionMasks, sfxCreateMask, sfxDestroyMask);
		processOutputCommands(level.commandQueue);
	}

	// If not in text mode, check for win conditions.
	if (textMode === false) {
		if (dontDoWin === undefined) {
			dontDoWin = false;
		}
		checkWin(dontDoWin);
	}

	// If not winning, process checkpoints and AGAIN command.
	if (!winning) {
		let checkpointIndex = level.commandQueue.indexOf('checkpoint');
		if (checkpointIndex >= 0) {
			if (verbose_logging) {
				let r = level.commandQueueSourceRules[checkpointIndex];
				consolePrintFromRule('CHECKPOINT command executed, saving current state to the restart state.', r);
			}
			restartTarget = level4Serialization();
			hasUsedCheckpoint = true;
			let backupStr = JSON.stringify(restartTarget);
			storage_set(document.URL + '_checkpoint', backupStr);
			storage_set(document.URL, curlevel);
		}

		let againIndex = level.commandQueue.indexOf('again');
		if (againIndex >= 0 && modified) {
			let r = level.commandQueueSourceRules[againIndex];
			let oldVerboseLogging = verbose_logging;
			let oldMessageText = messagetext;
			verbose_logging = false;
			if (processInput(-1, true, true)) {
				verbose_logging = oldVerboseLogging;
				if (verbose_logging) {
					consolePrintFromRule('AGAIN command executed, with changes detected - will execute another turn.', r);
				}
				againing = true;
				timer = 0;
			} else {
				verbose_logging = oldVerboseLogging;
				if (verbose_logging) {
					consolePrintFromRule("AGAIN command not executed, it wouldn't make any changes.", r);
				}
			}
			verbose_logging = oldVerboseLogging;
			messagetext = oldMessageText;
		}
	}

	if (verbose_logging) {
		consolePrint('Turn complete');
	}

	level.commandQueue = [];
	level.commandQueueSourceRules = [];
	return modified;
}

function checkWin(dontDoWin) {

	if (levelEditorOpened) {
		dontDoWin=true;
	}

	if (level.commandQueue.indexOf('win')>=0) {
		if (runrulesonlevelstart_phase){
			consolePrint("Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)");		
		} else {
			consolePrint("Win Condition Satisfied");
		}
		if(!dontDoWin){
			DoWin();
		}
		return;
	}

	let won= false;
	if (state.winconditions.length>0)  {
		let passed=true;
		for (let wcIndex=0;wcIndex<state.winconditions.length;wcIndex++) {
			let wincondition = state.winconditions[wcIndex];
			let filter1 = wincondition[1];
			let filter2 = wincondition[2];
			let aggr1 = wincondition[4];
			let aggr2 = wincondition[5];

			let rulePassed=true;
			
			const f1 = aggr1 ? c=>filter1.bitsSetInArray(c) : c=>!filter1.bitsClearInArray(c);
			const f2 = aggr2 ? c=>filter2.bitsSetInArray(c) : c=>!filter2.bitsClearInArray(c);

			switch(wincondition[0]) {
				case -1://NO
				{
					for (let i=0;i<level.n_tiles;i++) {
						let cell = level.getCellInto(i,_o10);
						if ( (f1(cell.data)) &&  
							 (f2(cell.data)) ) {
							rulePassed=false;
							break;
						}
					}

					break;
				}
				case 0://SOME
				{
					let passedTest=false;
					for (let i=0;i<level.n_tiles;i++) {
						let cell = level.getCellInto(i,_o10);
						if ( (f1(cell.data)) &&  
							 (f2(cell.data)) ) {
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
					for (let i=0;i<level.n_tiles;i++) {
						let cell = level.getCellInto(i,_o10);
						if ( (f1(cell.data)) &&  
							 (!f2(cell.data)) ) {
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
		if (runrulesonlevelstart_phase){
			consolePrint("Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)");		
		} else {
			consolePrint("Win Condition Satisfied");
		}
		if (!dontDoWin){
			DoWin();
		}
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
    for (let i=0;i<level.movementMask.length;i++) {
        if (level.movementMask[i]!==0) {
        	return true;
        }
    }
    return false;
}*/


function nextLevel() {
    againing=false;
	messagetext="";
	if (state && state.levels && (curlevel>state.levels.length) ){
		curlevel=state.levels.length-1;
	}
	ignoreNotJustPressedAction=true;
	if (titleScreen) {
		if (titleSelection===0) {
			//new game
			curlevel=0;
			curlevelTarget=null;
		} 			
		if (curlevelTarget!==null){			
			loadLevelFromStateTarget(state,curlevel,curlevelTarget);
		} else {
			loadLevelFromState(state,curlevel);
		}
	} else {	
		if (hasUsedCheckpoint){
			curlevelTarget=null;
			hasUsedCheckpoint=false;
		}
		if (curlevel<(state.levels.length-1))
		{			
			curlevel++;
			curlevelTarget=null;
			textMode=false;
			titleScreen=false;
			quittingMessageScreen=false;
			messageselected=false;

			if (curlevelTarget!==null){			
				loadLevelFromStateTarget(state,curlevel,curlevelTarget);
			} else {
				loadLevelFromState(state,curlevel);
			}
		} else {
			try{
				storage_remove(document.URL);
				storage_remove(document.URL+'_checkpoint');				
			} catch(ex){
					
			}
			
			curlevel=0;
			curlevelTarget=null;
			goToTitleScreen();
			tryPlayEndGameSound();
		}		
		//continue existing game
	}
	try {
		
		storage_set(document.URL,curlevel);
		if (curlevelTarget!==null){
			restartTarget=level4Serialization();
			let backupStr = JSON.stringify(restartTarget);
			storage_set(document.URL+'_checkpoint',backupStr);
		} else {
			storage_remove(document.URL+"_checkpoint");
		}		
		
	} catch (ex) {

	}

	if (state!==undefined && state.metadata.flickscreen!==undefined){
		oldflickscreendat=[0,0,Math.min(state.metadata.flickscreen[0],level.width),Math.min(state.metadata.flickscreen[1],level.height)];
	}
	canvasResize();	
}

function goToTitleScreen(){
    againing=false;
	messagetext="";
	titleScreen=true;
	textMode=true;
	doSetupTitleScreenLevelContinue();
	titleSelection=showContinueOptionOnTitleScreen()?1:0;
	generateTitleScreen();
	if (canvas!==null){//otherwise triggers error in cat bastard test
		regenSpriteImages();
	}
}

Rule.prototype.generateFindMatchesFunction = function() {
    let fn = '';
    
    // Initial mask check
    fn += 'if (!this.ruleMask.bitsSetInArray(level.mapCellContents.data)) return [];\n';
    fn += 'const d = level.delta_index(this.direction);\n';
    fn += 'const matches = [];\n';
    
    // Unroll the pattern matching loop
    for (let i = 0; i < this.patterns.length; i++) {
        fn += `let match${i};\n`;
        
        // Generate specialized matching code based on ellipsis count
        if (this.ellipsisCount[i] === 0) {
            fn += `match${i} = matchCellRow(this.direction, this.cellRowMatches[${i}], ` +
                  `this.patterns[${i}], this.cellRowMasks[${i}], ` +
                  `this.cellRowMasks_Movements[${i}], d);\n`;
        } else if (this.ellipsisCount[i] === 1) {
            fn += `match${i} = matchCellRowWildCard(this.direction, this.cellRowMatches[${i}], ` +
                  `this.patterns[${i}], this.cellRowMasks[${i}], ` +
                  `this.cellRowMasks_Movements[${i}], d, 1);\n`;
        } else { // ellipsisCount === 2
            fn += `match${i} = matchCellRowWildCard(this.direction, this.cellRowMatches[${i}], ` +
                  `this.patterns[${i}], this.cellRowMasks[${i}], ` +
                  `this.cellRowMasks_Movements[${i}], d, 2);\n`;
        }
        
        // Early return if no matches
        fn += `if (match${i}.length === 0) return [];\n`;
        fn += `matches.push(match${i});\n`;
    }
    
    fn += 'return matches;';
    
    // Cache the generated function like you do with matchCache
    if (fn in matchCache) {
        return matchCache[fn];
    }
    return matchCache[fn] = new Function('level', fn);
}


