'use strict';

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
	"..............v 1.8...............",
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

const blank_row = "..................................";

const titletemplate_menu_startgame = [
	"............start game............",
	"..........#.start game.#..........",
	"-----------.start game.-----------"
]

const titletemplate_menu_newgame = [
	".............new game.............",
	"...........#.new game.#...........",
	"------------.new game.------------",

]

const titletemplate_menu_continue = [
	".............continue.............",
	"...........#.continue.#...........",
	"------------.continue.------------",
]

const titletemplate_controls = {
	arrows:
		".arrow keys to move...............",
	action:
		".X to action......................",
	undorestart:
		".Z to undo, R to restart..........",
	undo: 
		".Z to undo........................",
	restart: 
		".R to restart.....................",
};

const loading_line = "------------ loading  ------------";

let titleImage = [];
const TERMINAL_WIDTH = intro_template[0].length;
const TERMINAL_HEIGHT = intro_template.length;
let titleScreen = true;
let titleMode = 0;//1 means there are options
let titleSelection = 0;
let titleSelected = false;

function showContinueOptionOnTitleScreen() {
	return (curlevel > 0 || curlevelTarget !== null) && (curlevel in state.levels);
}

function unloadGame() {
	if (levelEditorOpened){
		printLevel();
	}
	levelEditorOpened = false;
	state = introstate;
	level = new Level(0, 5, 5, 2, null);
	level.objects = new Int32Array(0);
	generateTitleScreen();
	canvasResize();
	redraw();
}


function align_centre(str, width) {
	if (str.length>=width){
		return str;
	}
	let free_space = width - str.length;
	let left_space = Math.floor(free_space / 2);
	let right_space = free_space - left_space;
	return '.'.repeat(left_space) + str + '.'.repeat(right_space);
}

function align_right(str,width) {
	if (str.length>=width){
		return str;
	}
	//first, if possible, add single space to right
	if (str.length<width-1){
		str = str + '.';
	}
	return str.padStart(width,'.');
}

function generateTitleScreen() {	
	titleMode = showContinueOptionOnTitleScreen() ? 1 : 0;
	
	if (state.levels.length === 0) {
		//if body has light-theme, set fgcolor to black, bgcolor to white
		if (document.body.classList.contains('light-theme')) {
			state.fgcolor = '#000000';
			state.bgcolor = '#FFFFFF';
		} else {
			state.fgcolor = '#FFFFFF';
			state.bgcolor = '#000000';
		}
		titleImage = intro_template;
		return;
	}

	let title = "PuzzleScript Game";
	if (state.metadata.title !== undefined) {
		title = state.metadata.title;
	}

	let title_lines = wordwrap(expandCJKCharacters(title), TERMINAL_WIDTH);
	title_lines = title_lines.map(l => align_centre(l,TERMINAL_WIDTH));
	let author_lines = [];
	if (state.metadata.author !== undefined) {
		author_lines = wordwrap(expandCJKCharacters("by " + state.metadata.author), TERMINAL_WIDTH);
		author_lines = author_lines.map(l => align_right(l,TERMINAL_WIDTH));
	}
	

	let layout_rows = {
		header: null,
		menu_options: [],
		controls: [titletemplate_controls.arrows],
	}	

	if (titleMode===0){
		if (titleSelected){
			layout_rows.menu_options.push(blank_row);
			layout_rows.menu_options.push(titletemplate_menu_startgame[2]);
			layout_rows.menu_options.push(blank_row);
		} else {
			layout_rows.menu_options.push(blank_row);
			layout_rows.menu_options.push(titletemplate_menu_startgame[1]);
			layout_rows.menu_options.push(blank_row);
		}
	} else {
		if (titleSelection===0){
			layout_rows.menu_options.push(titletemplate_menu_newgame[titleSelected?2:1]);
			layout_rows.menu_options.push(blank_row);
			layout_rows.menu_options.push(titletemplate_menu_continue[0]);
		} else {
			layout_rows.menu_options.push(titletemplate_menu_newgame[0]);
			layout_rows.menu_options.push(blank_row);
			layout_rows.menu_options.push(titletemplate_menu_continue[titleSelected?2:1]);
		}
	}

	const has_action = !('noaction' in state.metadata);
	const has_undo = !('noundo' in state.metadata);
	const has_restart = !('norestart' in state.metadata);

	let extra_header_rows = 0;
	if (has_action){
		layout_rows.controls.push(titletemplate_controls.action);
	} else {
		extra_header_rows++;
	}
	if (has_undo && has_restart){
		layout_rows.controls.push(titletemplate_controls.undorestart);
	}
	else if (has_restart){
		layout_rows.controls.push(titletemplate_controls.restart);
	} else if (has_undo){
		layout_rows.controls.push(titletemplate_controls.undo);
	} else {
		extra_header_rows++;
	}

	if (extra_header_rows>1){
		extra_header_rows--;
		layout_rows.controls.push(blank_row);
	}
	let header_size = 5 + extra_header_rows;
	//I have five rows to allocate to title + author
	let bs=0;
	let t_len = title_lines.length;
	let bm=0;
	let a_len = author_lines.length;
	let be=0;

	if (bs+t_len+bm+a_len+be<header_size){
		be++;
	}
	if (bs+t_len+bm+a_len+be<header_size){
		bm++;
	}
	if (bs+t_len+bm+a_len+be<header_size){
		bs++;
	}

	// if we've removed input options, we
	// allocate space above and below
	if (bs+t_len+bm+a_len+be<header_size){
		bs++;
	}
	
	//if they're too long, need to trim
	if (bs+t_len+bm+a_len+be>header_size){
		let title_trimmed=false;
		let author_trimmed=false;
		while (bs+t_len+bm+a_len+be>header_size){
			if (author_lines.length>1){
				author_lines.pop();
				author_trimmed=true;
				a_len--;
			} else if (title_lines.length>1){
				title_lines.pop();
				title_trimmed=true;
				t_len--;
			}
		}
		if (title_trimmed){
			logWarning("Game title is too long to fit on screen; truncating to fit.", state.metadata_lines.title, true);
		}
		if (author_trimmed){
			logWarning("Author name is too long to fit on screen; truncating to fit.", state.metadata_lines.author, true);
		}
	}

	let header = [];
	for (let i=0;i<bs;i++){
		header.push(blank_row);
	}
	for (let i=0;i<t_len;i++){
		header.push(title_lines[i]);
	}
	for (let i=0;i<bm;i++){
		header.push(blank_row);
	}
	for (let i=0;i<a_len;i++){
		header.push(author_lines[i]);
	}
	for (let i=0;i<be;i++){
		header.push(blank_row);
	}
	
	layout_rows.header = header;

	let selection_row = 0;
	/*now to build up the screen */
	titleImage = [];
		

	for (let i=0;i<layout_rows.header.length;i++){
		titleImage.push(layout_rows.header[i]);
	}

	for (let i=0;i<layout_rows.menu_options.length;i++){

		let cur_row = layout_rows.menu_options[i];
		if (cur_row.indexOf('---') !== -1){
			selection_row = titleImage.length;
		}

		titleImage.push(cur_row);

	}

	titleImage.push(blank_row);

	for (let i=0;i<layout_rows.controls.length;i++){
		titleImage.push(layout_rows.controls[i]);
	}

	titleImage.push(blank_row);

	while (titleImage.length<TERMINAL_HEIGHT){
		titleImage.push(blank_row);
	}
	
	titleImage.push(blank_row);

	let regen_letters = false;
	if (titleSelected){		
		let frame = get_title_animation_frame()+2;
		// frame = frame % 12; // loops not pingpong
		//want it to pingpong, not loop
		const loading_text = frame > 12;
		if (loading_text) {

			titleImage[selection_row] = loading_line;
			if (frame === 13){
				//in case it has new characters...
				regen_letters=true;
			}
		}
		frame = frame % 23;
		if (frame > 11) {
			frame = (11-(frame % 12));
		}
		const left_index_to_replace = 11-frame;
		const right_index_to_replace = 22+frame;
		if (!loading_text){
			//generate a string with "left_index_to_replace" spaces
			const spaces = ' '.repeat(left_index_to_replace);
			titleImage[selection_row] = spaces + '#' + titleImage[selection_row].slice(left_index_to_replace + 1, right_index_to_replace) + '#' + spaces;
			
		} else {
			titleImage[selection_row] = 
				titleImage[selection_row].slice(0, left_index_to_replace) 
				+ '#' 
				+ titleImage[selection_row].slice(left_index_to_replace + 1, right_index_to_replace) 
				+ '#' 
				+ titleImage[selection_row].slice(right_index_to_replace + 1);
		}

		if (frame === 2){
			//when animation starts playing (on frame 2), 
			regen_letters=true;
		}
	}


	

	for (let i=0;i<titleImage.length;i++){
		titleImage[i] = titleImage[i].replace(/\./g, ' ');
	}

	if (regen_letters){
		regenSpriteImages();
	}
}

const introstate = {
	title: "EMPTY GAME",
	attribution: "increpare",
	objectCount: 2,
	metadata: [],
	levels: [],
	bgcolor: "#000000",
	fgcolor: "#FFFFFF"
};

let state = introstate;

function deepClone(item) {
	if (!item) { return item; } // null, undefined values check

	let types = [Number, String, Boolean],
		result;

	// normalizing primitives if someone did new String('aaa'), or new Number('444');
	types.forEach(function (type) {
		if (item instanceof type) {
			result = type(item);
		}
	});

	if (typeof result == "undefined") {
		if (Object.prototype.toString.call(item) === "[object Array]") {
			result = [];
			item.forEach(function (child, index, array) {
				result[index] = deepClone(child);
			});
		} else if (typeof item == "object") {
			// testing that this is DOM
			if (item.nodeType && typeof item.cloneNode == "function") {
				let result = item.cloneNode(true);
			} else if (!item.prototype) { // check that this is a literal
				if (item instanceof Date) {
					result = new Date(item);
				} else {
					// it is an object literal
					result = {};
					for (let i in item) {
						result[i] = deepClone(item[i]);
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

function wordwrap(str, width) {

	width = width || 75;
	let cut = true;

	if (!str) { return str; }

	let regex = '.{1,' + width + '}(\\s|$)' + (cut ? '|.{' + width + '}|.+$' : '|\\S+?(\\s|$)');

	return str.match(RegExp(regex, 'g'));

}

let splitMessage = [];

function expandCJKCharacters(message) {
	//CJK characters are 2 characters wide, so we need to insert a space after
	//each one to give them space
	let expandedMessage = "";
	for (let i = 0; i < message.length; i++) {
		let char = message[i];
		expandedMessage += char;
		if (!font.hasOwnProperty(char)) {
			expandedMessage +=  " ";
		}
	}
	return expandedMessage;
}
function drawMessageScreen() {
	titleMode = 0;
	textMode = true;
	titleImage = deepClone(messagecontainer_template);

	for (let i = 0; i < titleImage.length; i++) {
		titleImage[i] = titleImage[i].replace(/\./g, ' ');
	}

	let emptyLineStr = titleImage[9];
	let xToContinueStr = titleImage[10];

	titleImage[10] = emptyLineStr;

	let width = titleImage[0].length;

	let message;
	if (messagetext === "") {
		let leveldat = state.levels[curlevel];
		message = leveldat.message.trim();
	} else {
		message = messagetext;
	}

	splitMessage = wordwrap(expandCJKCharacters(message), titleImage[0].length);


	let offset = 5 - ((splitMessage.length / 2) | 0);
	if (offset < 0) {
		offset = 0;
	}

	let count = Math.min(splitMessage.length, 12);
	for (let i = 0; i < count; i++) {
		let m = splitMessage[i];
		let row = offset + i;
		let messageLength = m.length;
		let lmargin = ((width - messageLength) / 2) | 0;
		let rmargin = width - messageLength - lmargin;
		let rowtext = titleImage[row];
		titleImage[row] = rowtext.slice(0, lmargin) + m + rowtext.slice(lmargin + m.length);
	}

	let endPos = 10;
	if (count >= 10) {
		if (count < 12) {
			endPos = count + 1;
		} else {
			endPos = 12;
		}
	}
	if (quittingMessageScreen) {
		titleImage[endPos] = emptyLineStr;
	} else {
		titleImage[endPos] = xToContinueStr;
	}

	canvasResize();
}

let loadedLevelSeed = 0;

function loadLevelFromLevelDat(state, leveldat, randomseed, clearinputhistory) {
	if (!randomseed) {
		randomseed = (Math.random() + Date.now()).toString();
	}
	loadedLevelSeed = randomseed;
	RandomGen = new RNG(loadedLevelSeed);
	forceRegenImages = true;
	ignoreNotJustPressedAction = true;
	titleScreen = false;
	titleMode = showContinueOptionOnTitleScreen() ? 1 : 0;
	titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
	titleSelected = false;
	againing = false;
	if (leveldat === undefined) {
		consolePrint("Trying to access a level that doesn't exist.", true);
		goToTitleScreen();
		return;
	}
	if (leveldat.message === undefined) {
		titleMode = 0;
		textMode = false;
		tick_lazy_function_generation(false);
		level = leveldat.clone();
		RebuildLevelArrays();


		if (state !== undefined) {
			if (state.metadata.flickscreen !== undefined) {
				oldflickscreendat = [
					0,
					0,
					Math.min(state.metadata.flickscreen[0], level.width),
					Math.min(state.metadata.flickscreen[1], level.height)
				];
			} else if (state.metadata.zoomscreen !== undefined) {
				oldflickscreendat = [
					0,
					0,
					Math.min(state.metadata.zoomscreen[0], level.width),
					Math.min(state.metadata.zoomscreen[1], level.height)
				];
			}
		}

		backups = []
		restartTarget = backupLevel();
		keybuffer = [];

		if ('run_rules_on_level_start' in state.metadata) {
			runrulesonlevelstart_phase = true;
			processInput(-1, true);
			runrulesonlevelstart_phase = false;
		}
	} else {
		ignoreNotJustPressedAction = true;
		tryPlayShowMessageSound();
		drawMessageScreen();
		canvasResize();
	}

	if (clearinputhistory === true) {
		clearInputHistory();
	}
}

function loadLevelFromStateTarget(state, levelindex, target, randomseed) {
	let leveldat = target;
	curlevel = levelindex;
	curlevelTarget = target;
	if (leveldat.message === undefined) {
		if (levelindex === 0) {
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();
		}
	}
	loadLevelFromLevelDat(state, state.levels[levelindex], randomseed);
	restoreLevel(target);
	restartTarget = target;
}

function loadLevelFromState(state, levelindex, randomseed) {
	let leveldat = state.levels[levelindex];
	curlevel = levelindex;
	curlevelTarget = null;
	if (leveldat !== undefined && leveldat.message === undefined) {
		if (levelindex === 0) {
			tryPlayStartLevelSound();
		} else {
			tryPlayStartLevelSound();
		}
	}
	loadLevelFromLevelDat(state, leveldat, randomseed);
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
if (titleMode > 0) {
	titleSelection = 1;
}


function tryPlaySimpleSound(soundname) {
	if (state.sfx_Events[soundname] !== undefined) {
		let seed = state.sfx_Events[soundname];
		playSound(seed, true);
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

function tryPlayUndoSound() {
	tryPlaySimpleSound("undo");
}

function tryPlayRestartSound() {
	tryPlaySimpleSound("restart");
}

function tryPlayShowMessageSound() {
	tryPlaySimpleSound("showmessage");
}

function tryPlayCloseMessageSound() {
	tryPlaySimpleSound("closemessage");
}

let backups = [];
let restartTarget;

function backupLevel() {
	let ret = {
		dat: new Int32Array(level.objects),
		width: level.width,
		height: level.height,
		oldflickscreendat: oldflickscreendat.concat([])
	};
	return ret;
}

function level4Serialization() {
	let ret = {
		dat: Array.from(level.objects),
		width: level.width,
		height: level.height,
		oldflickscreendat: oldflickscreendat.concat([])
	};
	return ret;
}



function setGameState(_state, command, randomseed) {

	if (_state === undefined) {
		_state = introstate;
		return;
	}
	oldflickscreendat = [];
	timer = 0;
	autotick = 0;
	winning = false;
	againing = false;
	messageselected = false;
	STRIDE_MOV = _state.STRIDE_MOV;
	STRIDE_OBJ = _state.STRIDE_OBJ;
	LAYER_COUNT = _state.LAYER_COUNT;
	RebuildGameArrays();

	sfxCreateMask = new BitVec(STRIDE_OBJ);
	sfxDestroyMask = new BitVec(STRIDE_OBJ);

	if (command === undefined) {
		command = ["restart"];
	}
	if ((state.levels.length === 0 || _state.levels.length === 0) && command.length > 0 && command[0] === "rebuild") {
		command = ["restart"];
	}
	if (randomseed === undefined) {
		randomseed = null;
	}
	RandomGen = new RNG(randomseed);

	state = _state;

	if (command[0] !== "rebuild") {
		backups = [];
	}
	//set sprites
	sprites = [];
	let keys = Object.keys(state.objects);
	let keys_l = keys.length;
	for (let k_i = 0; k_i < keys_l; k_i++) {
		const k = keys[k_i];
		let object = state.objects[k];
		let sprite = {
			colors: object.colors,
			dat: object.spritematrix
		};
		sprites[object.id] = sprite;
	}
	if (state.metadata.realtime_interval !== undefined) {
		autotick = 0;
		autotickinterval = state.metadata.realtime_interval * 1000;
	} else {
		autotick = 0;
		autotickinterval = 0;
	}

	if (state.metadata.key_repeat_interval !== undefined) {
		repeatinterval = state.metadata.key_repeat_interval * 1000;
	} else {
		repeatinterval = 150;
	}

	if (state.metadata.again_interval !== undefined) {
		againinterval = state.metadata.again_interval * 1000;
	} else {
		againinterval = 150;
	}
	if (throttle_movement && autotickinterval === 0) {
		logWarning("throttle_movement is designed for use in conjunction with realtime_interval. Using it in other situations makes games gross and unresponsive, broadly speaking.  Please don't.");
	}
	norepeat_action = state.metadata.norepeat_action !== undefined;

	switch (command[0]) {
		case "restart":
			{
				winning = false;
				timer = 0;
				titleScreen = true;
				tryPlayTitleSound();
				textMode = true;
				titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
				titleSelected = false;
				quittingMessageScreen = false;
				quittingTitleScreen = false;
				messageselected = false;
				titleMode = 0;
				if (showContinueOptionOnTitleScreen()) {
					titleMode = 1;
				}
				generateTitleScreen();
				break;
			}
		case "rebuild":
			{
				if (titleScreen) {
					generateTitleScreen();
				}
				break;
			}
		case "loadFirstNonMessageLevel": {
			for (let i = 0; i < state.levels.length; i++) {
				if (state.levels[i].hasOwnProperty("message")) {
					continue;
				}
				let targetLevel = i;
				curlevel = targetLevel;
				curlevelTarget = null;
				winning = false;
				timer = 0;
				titleScreen = false;
				textMode = false;
				tick_lazy_function_generation(false);
				titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
				titleSelected = false;
				quittingMessageScreen = false;
				quittingTitleScreen = false;
				messageselected = false;
				titleMode = 0;
				loadLevelFromState(state, targetLevel, randomseed);
				break;
			}
			break;
		}
		case "loadLevel":
			{
				let targetLevel = command[1];
				curlevel = targetLevel;
				curlevelTarget = null;
				winning = false;
				timer = 0;
				titleScreen = false;
				textMode = false;
				tick_lazy_function_generation(false);
				titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
				titleSelected = false;
				quittingMessageScreen = false;
				quittingTitleScreen = false;
				messageselected = false;
				titleMode = 0;
				loadLevelFromState(state, targetLevel, randomseed);
				break;
			}
		case "levelline":
			{
				let targetLine = command[1];
				for (let i = state.levels.length - 1; i >= 0; i--) {
					let level = state.levels[i];
					if (level.lineNumber <= targetLine + 1) {
						curlevel = i;
						curlevelTarget = null;
						winning = false;
						timer = 0;
						titleScreen = false;
						textMode = false;
						tick_lazy_function_generation(false);
						titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
						titleSelected = false;
						quittingMessageScreen = false;
						quittingTitleScreen = false;
						messageselected = false;
						titleMode = 0;
						loadLevelFromState(state, i);
						break;
					}
				}
				break;
			}
	}

	if (command[0] !== "rebuild") {
		clearInputHistory();
	}
	canvasResize();


	if (state.sounds.length === 0) {
		killAudioButton();
	} else {
		showAudioButton();
	}

}

function RebuildGameArrays(){
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
	_movementVecs = [new BitVec(STRIDE_MOV), new BitVec(STRIDE_MOV), new BitVec(STRIDE_MOV)];

	for (let i = 0; i < level.height; i++) {
		level.rowCellContents[i] = new BitVec(STRIDE_OBJ);
	}
	for (let i = 0; i < level.width; i++) {
		level.colCellContents[i] = new BitVec(STRIDE_OBJ);
	}

	for (let i = 0; i < level.height; i++) {
		level.rowCellContents_Movements[i] = new BitVec(STRIDE_MOV);
	}
	for (let i = 0; i < level.width; i++) {
		level.colCellContents_Movements[i] = new BitVec(STRIDE_MOV);
	}

	if (state.rigid){
		for (let i = 0; i < level.n_tiles; i++) {
			level.rigidMovementAppliedMask[i] = new BitVec(STRIDE_MOV);
			level.rigidGroupIndexMask[i] = new BitVec(STRIDE_MOV);
		}
	}
}

let messagetext = "";

function applyDiff(diff, level_objects) {

	let index = 0;

	while (index < diff.dat.length) {
		let start_index = diff.dat[index];
		let copy_length = diff.dat[index + 1];
		if (copy_length === 0) {
			break;//tail of buffer is all 0s
		}
		for (let j = 0; j < copy_length; j++) {
			level_objects[start_index + j] = diff.dat[index + 2 + j];
		}
		index += 2 + copy_length;
	}
}

function unconsolidateDiff(before, after) {

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

	oldflickscreendat = lev.oldflickscreendat.concat([]);

	if (diffing) {
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
	else {
		// layercount doesn't change

		for (let i = 0; i < level.n_tiles; i++) {
			level.movements[i] = 0;
		}

		if (state.rigid){
			for (let i = 0; i < level.n_tiles; i++) {
				level.rigidMovementAppliedMask[i].setZero();
				level.rigidGroupIndexMask[i].setZero();
			}
		}

		for (let i = 0; i < level.height; i++) {
			let rcc = level.rowCellContents[i];
			rcc.setZero();
		}
		for (let i = 0; i < level.width; i++) {
			let ccc = level.colCellContents[i];
			ccc.setZero();
		}
	}

	againing = false;
	level.commandQueue = [];
	level.commandQueueSourceRules = [];
}

let zoomscreen = false;
let flickscreen = false;
let screenwidth = 0;
let screenheight = 0;


//compresses 'before' into diff
function consolidateDiff(before, after) {
	if (before.width !== after.width || before.height !== after.height || before.dat.length !== after.dat.length) {
		return before;
	}
	if (before.hasOwnProperty("diff") || after.hasOwnProperty("diff")) {
		return before;
	}
	//only generate diffs if level size is bigger than this
	if (before.dat.length < 1024) {
		return before;
	}
	//diff structure: repeating ( start,length, [ data ] )
	let result = new Int32Array(128);
	let position = 0;
	let chain = false;
	let chain_start_idx_in_diff = -1;
	let before_dat = before.dat;
	let after_dat = after.dat;
	for (let i = 0; i < before_dat.length; i++) {
		if (chain === false) {
			if (before_dat[i] !== after_dat[i]) {
				chain = true;
				chain_start_idx_in_diff = position;

				if (result.length < position + 4) {
					let doubled = new Int32Array(2 * result.length);
					doubled.set(result);
					result = doubled;
				}

				result[position + 0] = i;
				result[position + 1] = 1;
				result[position + 2] = before_dat[i];
				position += 3;
			}
		} else {
			if (before_dat[i] !== after_dat[i]) {

				if (position + 1 >= result.length) {
					if (result.length < position + 4) {
						let doubled = new Int32Array(2 * result.length);
						doubled.set(result);
						result = doubled;
					}
				}
				result[chain_start_idx_in_diff + 1]++;
				result[position] = before_dat[i];
				position++;
			} else {
				chain = false;
			}
		}
	}
	return {
		diff: true,
		dat: result,
		width: before.width,
		height: before.height,
		oldflickscreendat: before.oldflickscreendat
	}
}

function addUndoState(state) {
	backups.push(state);
	if (backups.length > 2 && !backups[backups.length - 1].hasOwnProperty("diff")) {
		backups[backups.length - 3] = consolidateDiff(backups[backups.length - 3], backups[backups.length - 2]);
	}
}

function DoRestart(force) {
	if (restarting === true) {
		return;
	}
	if (force !== true && ('norestart' in state.metadata)) {
		return;
	}

	if (againing) {
		DoUndo(force, true);
	}
	restarting = true;
	if (force !== true) {
		addUndoState(backupLevel());
	}

	if (verbose_logging) {
		consolePrint("--- restarting ---", true);
	}

	restoreLevel(restartTarget);
	tryPlayRestartSound();

	if ('run_rules_on_level_start' in state.metadata) {
		processInput(-1, true);
	}

	level.commandQueue = [];
	level.commandQueueSourceRules = [];
	restarting = false;
}

function backupDiffers() {
	if (backups.length === 0) {
		return true;
	}
	let bak = backups[backups.length - 1];

	if (bak.hasOwnProperty("diff")) {
		return bak.dat.length !== 0 && bak.dat[1] !== 0;//if it's empty or if it's all 0s
	} else {
		for (let i = 0; i < level.objects.length; i++) {
			if (level.objects[i] !== bak.dat[i]) {
				return true;
			}
		}
		return false;
	}
}

function DoUndo(force, ignoreDuplicates) {
	if ((!levelEditorOpened) && ('noundo' in state.metadata && force !== true)) {
		return;
	}
	if (verbose_logging) {
		consolePrint("--- undoing ---", true);
	}

	if (ignoreDuplicates) {
		while (backupDiffers() === false) {
			backups.pop();
		}
	}

	if (backups.length > 0) {
		let torestore = backups[backups.length - 1];
		restoreLevel(torestore);
		backups = backups.splice(0, backups.length - 1);
		if (!force) {
			tryPlayUndoSound();
		}
	}
}

function getPlayerPositions() {
	let result = [];
	let [aggregate,playerMask] = state.playerMask;
	if (aggregate){
		for (let i = 0; i < level.n_tiles; i++) {
			level.getCellInto(i, _o11);
			if (playerMask.bitsSetInArray(_o11.data)) {
				result.push(i);
			}
		}
	} else {
		for (let i = 0; i < level.n_tiles; i++) {
			level.getCellInto(i, _o11);
			if (playerMask.anyBitsInCommon(_o11)) {
				result.push(i);
			}
		}
	}
	return result;
}

function getLayersOfMask(cellMask) {
	let layers = [];
	for (let i = 0; i < state.objectCount; i++) {
		if (cellMask.get(i)) {
			let n = state.idDict[i];
			let o = state.objects[n];
			layers.push(o.layer)
		}
	}
	return layers;
}

let CACHE_MOVEENTITIESATINDEX = {}
function generate_moveEntitiesAtIndex(OBJECT_SIZE, MOVEMENT_SIZE) {
	
	const fn = `'use strict';
    let cellMask = level.getCell(positionIndex);
	${UNROLL("cellMask &= entityMask", OBJECT_SIZE)}
    let layers = getLayersOfMask(cellMask);

	let movementMask=_movementVecs[_movementVecIndex];
	_movementVecIndex=(_movementVecIndex+1)%_movementVecs.length;
	${LEVEL_GET_MOVEMENTS_INTO( "positionIndex", "movementMask", MOVEMENT_SIZE)}

    for (let i=0;i<layers.length;i++) {
    	${ISHIFTOR("movementMask", "dirMask", "(5 * layers[i])")}
    }
		
    ${LEVEL_SET_MOVEMENTS( "positionIndex", "movementMask", MOVEMENT_SIZE)}

	const colIndex=(positionIndex/level.height)|0;
	const rowIndex=(positionIndex%level.height);
	${UNROLL("level.colCellContents_Movements[colIndex] |= movementMask", MOVEMENT_SIZE)}
	${UNROLL("level.rowCellContents_Movements[rowIndex] |= movementMask", MOVEMENT_SIZE)}
	${UNROLL("level.mapCellContents_Movements |= movementMask", MOVEMENT_SIZE)}
	`
	if (fn in CACHE_MOVEENTITIESATINDEX) {
		return CACHE_MOVEENTITIESATINDEX[fn];
	}
	return CACHE_MOVEENTITIESATINDEX[fn] = new Function("level", "positionIndex", "entityMask", "dirMask", fn);
}


let CACHE_CALCULATEROWCOLMASKS = {}
function generate_calculateRowColMasks(OBJECT_SIZE, MOVEMENT_SIZE) {
	const fn = `'use strict';
		level.mapCellContents.data.fill(0);
		level.mapCellContents_Movements.data.fill(0);

		for (let i=0;i<level.width;i++) {
			let ccc = level.colCellContents[i];
			${SET_ZERO("ccc")}
			let ccc_Movements = level.colCellContents_Movements[i];
			${SET_ZERO("ccc_Movements")}
		}

		for (let i=0;i<level.height;i++) {
			let rcc = level.rowCellContents[i];
			${SET_ZERO("rcc")}
			let rcc_Movements = level.rowCellContents_Movements[i];
			${SET_ZERO("rcc_Movements")}
		}

		for (let i=0;i<level.width;i++) {
			for (let j=0;j<level.height;j++) {
				let index = j+i*level.height;
				let cellContents=_o9;
				${LEVEL_GET_CELL_INTO("level", "index", "cellContents", OBJECT_SIZE)}
				${UNROLL("level.mapCellContents |= cellContents", OBJECT_SIZE)}
				${UNROLL("level.rowCellContents[j] |= cellContents", OBJECT_SIZE)}
				${UNROLL("level.colCellContents[i] |= cellContents", OBJECT_SIZE)}
				
				let mapCellContents_Movements=level.getMovementsInto(index,_m1);
				${UNROLL("level.mapCellContents_Movements |= mapCellContents_Movements", MOVEMENT_SIZE)}
				${UNROLL("level.rowCellContents_Movements[j] |= mapCellContents_Movements", MOVEMENT_SIZE)}
				${UNROLL("level.colCellContents_Movements[i] |= mapCellContents_Movements", MOVEMENT_SIZE)}
			}
		}`
	if (fn in CACHE_CALCULATEROWCOLMASKS) {
		return CACHE_CALCULATEROWCOLMASKS[fn];
	}
	return CACHE_CALCULATEROWCOLMASKS[fn] = new Function("level", fn);
}

function startMovement(dir) {
	let movedany = false;
	let playerPositions = getPlayerPositions();
	for (let i = 0; i < playerPositions.length; i++) {
		let playerPosIndex = playerPositions[i];
		state.moveEntitiesAtIndex(level, playerPosIndex, state.playerMask[1], dir);
	}
	return playerPositions;
}

let dirMasksDelta = {
	1: [0, -1],//up
	2: [0, 1],//'down'  : 
	4: [-1, 0],//'left'  : 
	8: [1, 0],//'right' : 
	15: [0, 0],//'?' : 
	16: [0, 0],//'action' : 
	3: [0, 0]//'no'
};

let dirMaskName = {
	1: 'up',
	2: 'down',
	4: 'left',
	8: 'right',
	15: '?',
	16: 'action',
	3: 'no'
};

let seedsToPlay_CanMove = [];
let seedsToPlay_CantMove = [];

function repositionEntitiesOnLayer(positionIndex, layer, dirMask) {
	let delta = dirMasksDelta[dirMask];

	let dx = delta[0];
	let dy = delta[1];
	let tx = ((positionIndex / level.height) | 0);
	let ty = ((positionIndex % level.height));
	let maxx = level.width - 1;
	let maxy = level.height - 1;

	if ((tx === 0 && dx < 0) || (tx === maxx && dx > 0) || (ty === 0 && dy < 0) || (ty === maxy && dy > 0)) {
		return false;
	}

	let targetIndex = (positionIndex + delta[1] + delta[0] * level.height);

	let layerMask = state.layerMasks[layer];
	let targetMask = level.getCellInto(targetIndex, _o7);
	let sourceMask = level.getCellInto(positionIndex, _o8);

	if (layerMask.anyBitsInCommon(targetMask) && (dirMask != 16)) {
		return false;
	}

	for (let i = 0; i < state.sfx_MovementMasks[layer].length; i++) {
		let o = state.sfx_MovementMasks[layer][i];
		let objectMask = o.objectMask;
		if (objectMask.anyBitsInCommon(sourceMask)) {
			let movementMask = level.getMovements(positionIndex);
			let directionMask = o.directionMask;
			if (movementMask.anyBitsInCommon(directionMask) && seedsToPlay_CanMove.indexOf(o.seed) === -1) {
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

	let colIndex = (targetIndex / level.height) | 0;
	let rowIndex = (targetIndex % level.height);

	level.colCellContents[colIndex].ior(movingEntities);
	level.rowCellContents[rowIndex].ior(movingEntities);
	//corresponding movement stuff in setmovements
	return true;
}

let CACHE_REPOSITIONENTITIESATCELL = {}
function generate_repositionEntitiesAtCell(OBJECT_SIZE, MOVEMENT_SIZE) {
	let fn = `'use strict';
    const movementMask = level.getMovements(positionIndex);
    if (${IS_ZERO("movementMask",MOVEMENT_SIZE)}){
        return false;
	}

    let moved = false;
    ${FOR(0,LAYER_COUNT,layer=>`{
        const layerMovement = ${GETSHIFTOR("movementMask",0x1f, 5*layer)};
        if (layerMovement !== 0) {
            const thismoved = repositionEntitiesOnLayer(positionIndex, ${layer}, layerMovement);
            if (thismoved) {
                ${ISHIFTCLEAR("movementMask","layerMovement", 5*layer)}
                moved = true;
            }
        }
	}`)}

	${FOR(0,MOVEMENT_SIZE,i=>`
		level.movements[positionIndex * STRIDE_MOV + ${i}] = movementMask.data[${i}];
	`)}
		
	//corresponding object stuff in repositionEntitiesOnLayer
	const colIndex=(positionIndex/this.height)|0;
	const rowIndex=(positionIndex%level.height);
	${UNROLL("level.colCellContents_Movements[colIndex] |= movementMask",MOVEMENT_SIZE)}
	${UNROLL("level.rowCellContents_Movements[rowIndex] |= movementMask",MOVEMENT_SIZE)}
	${UNROLL("level.mapCellContents_Movements |= movementMask",MOVEMENT_SIZE)}
	
    return moved;`
	if (fn in CACHE_REPOSITIONENTITIESATCELL) {
		return CACHE_REPOSITIONENTITIESATCELL[fn];
	}
	return CACHE_REPOSITIONENTITIESATCELL[fn] = new Function("level", "positionIndex", fn);
}

let ellipsisPattern = ['ellipsis'];


function Rule(rule) {
	this.direction = rule[0]; 		/* direction rule scans in */
	this.patterns = rule[1];		/* lists of CellPatterns to match */
	this.hasReplacements = rule[2];
	this.lineNumber = rule[3];		/* rule source for debugging */
	this.ellipsisCount = rule[4];		/* number of ellipses present */
	this.groupNumber = rule[5];		/* execution group number of rule */
	this.rigid = rule[6];
	this.commands = rule[7];		/* cancel, restart, sfx, etc */
	this.isRandom = rule[8];
	this.cellRowMasks = rule[9];
	this.cellRowMasks_Movements = rule[10];
	this.ruleMask = new BitVec(STRIDE_OBJ);
	this.applyAt = this.generateApplyAt(this.patterns, this.ellipsisCount, STRIDE_OBJ, STRIDE_MOV);
	for (const m of this.cellRowMasks) {
		this.ruleMask.ior(m);
	}

	/*I tried out doing a ruleMask_movements as well along the lines of the above,
	but it didn't help at all - I guess because almost every tick there are movements 
	somewhere on the board - move filtering works well at a row/col level, but is pretty 
	useless (or worse than useless) on a boardwide level*/

	
	this.cellRowMatches = [];
	
	if (lazyFunctionGeneration){
		WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.push(this);
	} else {
		this.generate_all_MatchFunctions();
	}
	/* TODO: eliminate rigid, groupNumber, isRandom
	from this class by moving them up into a RuleGroup class */

	this.findMatches = this.generateFindMatchesFunction();
}

Rule.prototype.generate_all_MatchFunctions = function(){
	for (let i = 0; i < this.patterns.length; i++) {
		this.cellRowMatches.push(this.generateCellRowMatchesFunction(this.patterns[i], this.ellipsisCount[i]));
	}
}

let CACHE_RULE_CELLROWMATCHESFUNCTION = {}
Rule.prototype.generateCellRowMatchesFunction = function (cellRow, ellipsisCount) {
	if (ellipsisCount === 0) {
		let cr_l = cellRow.length;

		// Find which object and movement indices are actually used
		let usedObjectIndices = new Set();
		let usedMovementIndices = new Set();
		
		for (let i = 0; i < cr_l; i++) {
			const pattern = cellRow[i];
			for (let j = 0; j < STRIDE_OBJ; j++) {
				if (pattern.objectsPresent.data[j] || pattern.objectsMissing.data[j]) {
					usedObjectIndices.add(j);
				}
			}
			//and anyObjectsPresent
			for (let j = 0; j < pattern.anyObjectsPresent.length; j++) {
				for (let k = 0; k < STRIDE_OBJ; k++) {
					if (pattern.anyObjectsPresent[j].data[k]) {
						usedObjectIndices.add(k);
					}
				}
			}
			for (let j = 0; j < STRIDE_MOV; j++) {
				if (pattern.movementsPresent.data[j] || pattern.movementsMissing.data[j]) {
					usedMovementIndices.add(j);
				}
			}
		}

		// Generate function with only used indices
		let fn = "";
		const objStride = STRIDE_OBJ === 1 ? '' : '*' + STRIDE_OBJ;
		const movStride = STRIDE_MOV === 1 ? '' : '*' + STRIDE_MOV;
		
		for (let i of usedObjectIndices) {
			fn += 'let cellObjects' + i + ' = objects[i' + objStride + (i ? '+' + i : '') + '];\n';
		}
		for (let i of usedMovementIndices) {
			fn += 'let cellMovements' + i + ' = movements[i' + movStride + (i ? '+' + i : '') + '];\n';
		}
		
		fn += "return " + cellRow[0].generateMatchString('0_');
		for (let cellIndex = 1; cellIndex < cr_l; cellIndex++) {
			fn += "&&cellRow[" + cellIndex + "].matches(i+" + cellIndex + "*d, objects, movements)";
		}
		fn += ";";

		if (fn in CACHE_RULE_CELLROWMATCHESFUNCTION) {
			return CACHE_RULE_CELLROWMATCHESFUNCTION[fn];
		}
		return CACHE_RULE_CELLROWMATCHESFUNCTION[fn] = new Function("cellRow", "i", 'd', 'objects', 'movements', fn);
	} else if (ellipsisCount === 1) {
		let cr_l = cellRow.length;

		let fn = `let result = [];
if(cellRow[0].matches(i, objects, movements)`;
		let cellIndex = 1;
		for (; cellRow[cellIndex] !== ellipsisPattern; cellIndex++) {
			fn += "&&cellRow[" + cellIndex + "].matches(i+" + cellIndex + "*d, objects, movements)";
		}
		cellIndex++;
		fn += `) {
	for (let k=kmin;k<kmax;k++) {
		if(cellRow[`+ cellIndex + `].matches((i+d*(k+` + (cellIndex - 1) + `)), objects, movements)`;
		cellIndex++;
		for (; cellIndex < cr_l; cellIndex++) {
			fn += "&&cellRow[" + cellIndex + "].matches((i+d*(k+" + (cellIndex - 1) + ")), objects, movements)";
		}
		fn += `){
			result.push([i,k]);
		}
	}
}
`;
		fn += "return result;"


		if (fn in CACHE_RULE_CELLROWMATCHESFUNCTION) {
			return CACHE_RULE_CELLROWMATCHESFUNCTION[fn];
		}
		return CACHE_RULE_CELLROWMATCHESFUNCTION[fn] = new Function("cellRow", "i", "kmax", "kmin", 'd', "objects", "movements", fn);
	} else { //ellipsisCount===2
		let cr_l = cellRow.length;

		let ellipsis_index_1 = -1;
		let ellipsis_index_2 = -1;
		for (let cellIndex = 0; cellIndex < cr_l; cellIndex++) {
			if (cellRow[cellIndex] === ellipsisPattern) {
				if (ellipsis_index_1 === -1) {
					ellipsis_index_1 = cellIndex;
				} else {
					ellipsis_index_2 = cellIndex;
					break;
				}
			}
		}

		let fn = `let result = [];
if(cellRow[0].matches(i, objects, movements)`;

		for (let idx = 1; idx < ellipsis_index_1; idx++) {
			fn += "&&cellRow[" + idx + "].matches(i+" + idx + "*d, objects, movements)";
		}
		fn += ") {\n";

		//try match middle part
		fn += `
	for (let k1=k1min;k1<k1max;k1++) {
		if(cellRow[`+ (ellipsis_index_1 + 1) + `].matches((i+d*(k1+` + (ellipsis_index_1 + 1 - 1) + `)), objects, movements)`;
		for (let idx = ellipsis_index_1 + 2; idx < ellipsis_index_2; idx++) {
			fn += "&&cellRow[" + idx + "].matches((i+d*(k1+" + (idx - 1) + ")), objects, movements)";
		}
		fn += "		){\n";
		//try match right part

		fn += `
			for (let k2=k2min;k1+k2<kmax && k2<k2max;k2++) {
				if(cellRow[`+ (ellipsis_index_2 + 1) + `].matches((i+d*(k1+k2+` + (ellipsis_index_2 + 1 - 2) + `)), objects, movements)`;
		for (let idx = ellipsis_index_2 + 2; idx < cr_l; idx++) {
			fn += "&&cellRow[" + idx + "].matches((i+d*(k1+k2+" + (idx - 2) + ")), objects, movements)";
		}
		fn += `
				){
					result.push([i,k1,k2]);
				}
			}
		}
	}			
}	
return result;`;

		if (fn in CACHE_RULE_CELLROWMATCHESFUNCTION) {
			return CACHE_RULE_CELLROWMATCHESFUNCTION[fn];
		}
		return CACHE_RULE_CELLROWMATCHESFUNCTION[fn] = new Function("cellRow", "i", "kmax", "kmin", "k1max", "k1min", "k2max", "k2min", 'd', "objects", "movements", fn);
	}
}


let STRIDE_OBJ = 1;
let STRIDE_MOV = 1;
let LAYER_COUNT = 1;
const FALSE_FUNCTION = new Function("return false;");

// We don't generate the matches functions all at once at initailization, we generate them in the background/as needed

function CellPattern(row) {
	this.objectsPresent = row[0];
	this.objectsMissing = row[1];
	this.anyObjectsPresent = row[2];
	this.movementsPresent = row[3];
	this.movementsMissing = row[4];
	if (lazyFunctionGeneration){
		WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.push(this);
	} else {
		this.matches = this.generateMatchFunction();
	}
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
	this.replace = null;
};

CellPattern.prototype.replace = function (level, rule, currentIndex) {
	let fn = this.generateReplaceFunction(
			STRIDE_OBJ,
			STRIDE_MOV,
			rule
		);
	this.replace = fn;
	return this.replace(level, rule, currentIndex);
}

CellPattern.prototype.generateMatchString = function () {
	let fn = "(true";
	for (let i = 0; i < Math.max(STRIDE_OBJ, STRIDE_MOV); ++i) {
		const co = 'cellObjects' + i;
		const cm = 'cellMovements' + i;
		const op = this.objectsPresent.data[i];
		const om = this.objectsMissing.data[i];
		const mp = this.movementsPresent.data[i];
		const mm = this.movementsMissing.data[i];
		if (op) {
			if (op & (op - 1))
				fn += '\t\t&& ((' + co + '&' + op + ')===' + op + ')\n';
			else
				fn += '\t\t&& (' + co + '&' + op + ')\n';
		}
		if (om)
			fn += '\t\t&& !(' + co + '&' + om + ')\n';
		if (mp) {
			if (mp & (mp - 1))
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

let CACHE_CELLPATTERN_MATCHFUNCTION = new Map();
let _generateMatchFunction_key_array = new Int32Array(0);
CellPattern.prototype.generateMatchFunction = function() {
    // Calculate total size needed for the key array
    const keyLength = STRIDE_OBJ * 2 + STRIDE_MOV * 2 + 
                     this.anyObjectsPresent.length * STRIDE_OBJ + 2;
	if (keyLength!==_generateMatchFunction_key_array.length) {
		_generateMatchFunction_key_array = new Int32Array(keyLength);
	}
    const keyArray = _generateMatchFunction_key_array;
    let keyIndex = 0;

    // Fill the array with data
    for (let i = 0; i < STRIDE_OBJ; i++) {
        keyArray[keyIndex++] = this.objectsPresent.data[i] || 0;
        keyArray[keyIndex++] = this.objectsMissing.data[i] || 0;
    }
    for (let i = 0; i < STRIDE_MOV; i++) {
        keyArray[keyIndex++] = this.movementsPresent.data[i] || 0;
        keyArray[keyIndex++] = this.movementsMissing.data[i] || 0;
    }
    for (let i = 0; i < this.anyObjectsPresent.length; i++) {
        for (let j = 0; j < STRIDE_OBJ; j++) {
            keyArray[keyIndex++] = this.anyObjectsPresent[i].data[j] || 0;
        }
    }
    keyArray[keyIndex++] = STRIDE_OBJ;
    keyArray[keyIndex++] = STRIDE_MOV;
	let str_key = keyArray.toString();

    if (CACHE_CELLPATTERN_MATCHFUNCTION.has(str_key)) {
        return CACHE_CELLPATTERN_MATCHFUNCTION.get(str_key);
    }

    const objStride = STRIDE_OBJ === 1 ? '' : `*${STRIDE_OBJ}`;
    const movStride = STRIDE_MOV === 1 ? '' : `*${STRIDE_MOV}`;
    
    let fn = '';
    
    for (let i = 0; i < STRIDE_OBJ; ++i) {
        fn += `const cellObjects${i} = objects[i${objStride}${i ? '+' + i : ''}];\n`;
    }
    
    for (let i = 0; i < STRIDE_MOV; ++i) {
        fn += `const cellMovements${i} = movements[i${movStride}${i ? '+' + i : ''}];\n`;
    }
    
    fn += `return ${this.generateMatchString()};`;

    const result = new Function("i", "objects", "movements", fn);
    CACHE_CELLPATTERN_MATCHFUNCTION.set(str_key, result);
    return result;
}

let _o1, _o2, _o2_5, _o3, _o4, _o5, _o6, _o7, _o8, _o9, _o10, _o11, _o12;
let _m1, _m2, _m3;

let CACHE_CELLPATTERN_REPLACEFUNCTION = {}
let CACHE_CHECK_COUNT=0;
let CACHE_HIT_COUNT=0;
let _replace_function_key_array = new Int32Array(0);


CellPattern.prototype.generateReplaceFunction = function (OBJECT_SIZE, MOVEMENT_SIZE,rule) {
	if (this.replacement===null){
		return FALSE_FUNCTION;
	}

	const array_len = 3*OBJECT_SIZE + 4*MOVEMENT_SIZE + 3;
	if (array_len!==_replace_function_key_array.length) {
		_replace_function_key_array = new Int32Array(array_len);
	}

	const key_array = _replace_function_key_array;
	for (let i = 0; i < OBJECT_SIZE; i++) {
		key_array[i] = this.replacement.objectsSet.data[i] || 0;
		key_array[i+OBJECT_SIZE] = this.replacement.objectsClear.data[i] || 0;
		key_array[i+2*OBJECT_SIZE+3*MOVEMENT_SIZE] = this.replacement.randomEntityMask.data[i] || 0;
	}
	for (let i = 0; i < MOVEMENT_SIZE; i++) {
		key_array[i+2*OBJECT_SIZE] = this.replacement.movementsSet.data[i] || 0;
		key_array[i+2*OBJECT_SIZE+MOVEMENT_SIZE] = this.replacement.movementsClear.data[i] || 0;
		key_array[i+2*OBJECT_SIZE+2*MOVEMENT_SIZE] = this.replacement.movementsLayerMask.data[i] || 0;
		key_array[i+3*OBJECT_SIZE+3*MOVEMENT_SIZE] = this.replacement.randomDirMask.data[i] || 0;
	}
	key_array[3*OBJECT_SIZE + 4*MOVEMENT_SIZE] = OBJECT_SIZE;
	key_array[3*OBJECT_SIZE + 4*MOVEMENT_SIZE+1] = MOVEMENT_SIZE;
	key_array[3*OBJECT_SIZE + 4*MOVEMENT_SIZE+2] = rule.rigid;

	const key = key_array.toString();
	if (key in CACHE_CELLPATTERN_REPLACEFUNCTION) {
		return CACHE_CELLPATTERN_REPLACEFUNCTION[key];
	}
	
	const replace_randomEntityMask_zero = this.replacement.randomEntityMask.iszero()
	const replace_randomDirMask_zero = this.replacement.randomDirMask.iszero()
	let deterministic = replace_randomEntityMask_zero && replace_randomDirMask_zero;

	let fn = `	
		let replace = this.replacement;

		if (replace === null) {
			return false;
		}

		const replace_RandomEntityMask = replace.randomEntityMask;
		const replace_RandomDirMask = replace.randomDirMask;

		// Using IMPORT_COMPILE_TIME_ARRAY should make the following three declarations faster,
		// but it really slows down the compiler.
		const objectsSet = _o1;	
		${UNROLL("objectsSet = replace.objectsSet", OBJECT_SIZE)}
	
		const objectsClear = _o2;
		${UNROLL("objectsClear = replace.objectsClear", OBJECT_SIZE)}

		const movementsSet = _m1;
		${UNROLL("movementsSet = replace.movementsSet", MOVEMENT_SIZE)}
		
		const movementsClear = _m2;
		
		${FOR(0,MOVEMENT_SIZE,i=>
			`movementsClear.data[${i}] = ${this.replacement.movementsClear.data[i] | this.replacement.movementsLayerMask.data[i]};\n`
		)}

		${IF_LAZY(!replace_randomEntityMask_zero,()=>`
			const choices=[];
			${FOR(0,(32*OBJECT_SIZE),i =>
			`{
				if (${this.replacement.randomEntityMask.get(i)}) {
					choices.push(${i});
				}
			}`
			)}
			const rand = choices[Math.floor(RandomGen.uniform() * choices.length)];
			const n = state.idDict[rand];
			const o = state.objects[n];
			${IBITSET("objectsSet", "rand")}
			${UNROLL("objectsClear |= state.layerMasks[o.layer]", OBJECT_SIZE)}
			${ISHIFTOR("movementsClear", "0x1f", "(5 * o.layer)")}
		`)}
		${IF_LAZY(!replace_randomDirMask_zero,()=>`
			${FOR(0, LAYER_COUNT, layerIndex =>
			`{
				if (${GET("replace_RandomDirMask", 5*layerIndex )}) {
					const randomDir = Math.floor(RandomGen.uniform()*4);
					${IBITSET("movementsSet", `(randomDir + 5 * ${layerIndex})`)}
				}
			}`
			)}
		`)}
		


		const oldCellMask = _o3;
		${LEVEL_GET_CELL_INTO("level", "currentIndex", "oldCellMask", OBJECT_SIZE)}

		const curCellMask = _o2_5;
		${FOR(0, OBJECT_SIZE,i=>`
			curCellMask.data[${i}] = (oldCellMask.data[${i}] & (~objectsClear.data[${i}])) | objectsSet.data[${i}];
		`)}

		const oldMovementMask = level.getMovements(currentIndex);
		const curMovementMask = _m3;
		${FOR(0, MOVEMENT_SIZE, i => `
			curMovementMask.data[${i}] = (oldMovementMask.data[${i}] & (~movementsClear.data[${i}])) | movementsSet.data[${i}]
		`)}


		let curRigidGroupIndexMask;
		let curRigidMovementAppliedMask;
		let rigidchange=false;		
		${IF_LAZY(rule.rigid,()=>`
			const rigidGroupIndex = ${state.groupNumber_to_RigidGroupIndex[rule.groupNumber]+1};
			const rigidMask = new BitVec(${STRIDE_MOV});
			${FOR(0,level.layerCount,layer=>`
				${ISHIFTOR("rigidMask", "rigidGroupIndex", layer * 5)}
			`)}
			${UNROLL("rigidMask &= replace.movementsLayerMask", MOVEMENT_SIZE)}
			
			curRigidGroupIndexMask = level.rigidGroupIndexMask[currentIndex] || new BitVec(${STRIDE_MOV});
			curRigidMovementAppliedMask = level.rigidMovementAppliedMask[currentIndex] || new BitVec(${STRIDE_MOV});

			if (${NOT_BITS_SET_IN_ARRAY("rigidMask", "curRigidGroupIndexMask.data", MOVEMENT_SIZE)} &&
				${NOT_BITS_SET_IN_ARRAY("replace.movementsLayerMask", "curRigidMovementAppliedMask.data", MOVEMENT_SIZE)}) 
			{
				${UNROLL("curRigidGroupIndexMask |= rigidMask", MOVEMENT_SIZE)}
				${UNROLL("curRigidMovementAppliedMask |= replace.movementsLayerMask", MOVEMENT_SIZE)}
				rigidchange=true;
			}
		`)}
		
		if (${EQUALS("oldCellMask", "curCellMask", OBJECT_SIZE)} 
			&& ${EQUALS("oldMovementMask", "curMovementMask", MOVEMENT_SIZE)} 
			&& !rigidchange) { 
			//nothing changed
			return false;
		}
				
		if (rigidchange) {
			level.rigidGroupIndexMask[currentIndex] = curRigidGroupIndexMask;
			level.rigidMovementAppliedMask[currentIndex] = curRigidMovementAppliedMask;
		}

		const created = _o4;
		${UNROLL("created = curCellMask", OBJECT_SIZE)}
		${UNROLL("created &= ~oldCellMask", OBJECT_SIZE)}
		${UNROLL("sfxCreateMask |= created", OBJECT_SIZE)}
		
		const destroyed = _o5;
		${UNROLL("destroyed = oldCellMask", OBJECT_SIZE)}
		${UNROLL("destroyed &= ~curCellMask", OBJECT_SIZE)}
		${UNROLL("sfxDestroyMask |= destroyed", OBJECT_SIZE)}

		${LEVEL_SET_CELL("level", "currentIndex", "curCellMask", OBJECT_SIZE)}
		${LEVEL_SET_MOVEMENTS( "currentIndex", "curMovementMask", MOVEMENT_SIZE)}

		const colIndex=(currentIndex/level.height)|0;
		const rowIndex=(currentIndex%level.height);

		${UNROLL("level.colCellContents[colIndex] |= curCellMask", OBJECT_SIZE)}
		${UNROLL("level.rowCellContents[rowIndex] |= curCellMask", OBJECT_SIZE)}
		${UNROLL("level.mapCellContents |= curCellMask", OBJECT_SIZE)}
		return true;	
	`

	return CACHE_CELLPATTERN_REPLACEFUNCTION[key] = new Function("level", "rule", "currentIndex", fn);
}


let CACHE_MATCHCELLROW = {}
function generateMatchCellRow(OBJECT_SIZE, MOVEMENT_SIZE) {
	const fn = `'use strict';
	let result=[];
	
	if ((${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.mapCellContents.data", OBJECT_SIZE)})||
	(${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.mapCellContents_Movements.data", MOVEMENT_SIZE)})) {
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
			if (${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.rowCellContents[y].data", OBJECT_SIZE)} 
			|| ${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.rowCellContents_Movements[y].data", MOVEMENT_SIZE)}) {
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
			if (${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.colCellContents[x].data", OBJECT_SIZE)}
			|| ${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.colCellContents_Movements[x].data", MOVEMENT_SIZE)}) {
				continue;
			}

			for (let y=ymin;y<ymax;y++) {
				const i = x*level.height+y;
				if (cellRowMatch(cellRow,i, d, level.objects, level.movements))
				{
					result.push(i);
				}
			}
		}		
	}

	return result;`
	if (fn in CACHE_MATCHCELLROW) {
		return CACHE_MATCHCELLROW[fn];
	}
	return CACHE_MATCHCELLROW[fn] = new Function("level", "direction", "cellRowMatch", "cellRow", "cellRowMask", "cellRowMask_Movements", "d", fn);
}

let CACHE_MATCHCELLROWWILDCARD = {}
function generateMatchCellRowWildCard(OBJECT_SIZE, MOVEMENT_SIZE) {
	const fn = `'use strict';
	let result=[];
	if ((${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.mapCellContents.data", OBJECT_SIZE)})||
	(${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.mapCellContents_Movements.data", MOVEMENT_SIZE)})) {
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
			if (${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.rowCellContents[y].data", OBJECT_SIZE)}
			|| ${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.rowCellContents_Movements[y].data", MOVEMENT_SIZE)}) {
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
			if (${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.colCellContents[x].data", OBJECT_SIZE)}
			|| ${NOT_BITS_SET_IN_ARRAY("cellRowMask_Movements", "level.colCellContents_Movements[x].data", MOVEMENT_SIZE)}) {
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

	return result;`
	//function matchCellRowWildCard(direction, cellRowMatch, cellRow,cellRowMask,cellRowMask_Movements,d,wildcardCount) {
	if (fn in CACHE_MATCHCELLROWWILDCARD) {
		return CACHE_MATCHCELLROWWILDCARD[fn];
	}
	return CACHE_MATCHCELLROWWILDCARD[fn] = new Function("direction", "cellRowMatch", "cellRow", "cellRowMask", "cellRowMask_Movements", "d", "wildcardCount", fn);
}

function generateTuples(lists) {
	let tuples = [[]];

	for (let i = 0; i < lists.length; i++) {
		const row = lists[i];
		const newtuples = [];
		for (let j = 0; j < row.length; j++) {
			let valtoappend = row[j];
			for (let k = 0; k < tuples.length; k++) {
				const tuple = tuples[k];
				const newtuple = tuple.concat([valtoappend]);
				newtuples.push(newtuple);
			}
		}
		tuples = newtuples;
	}
	return tuples;
}

Rule.prototype.findMatches = function () {
	if (!this.ruleMask.bitsSetInArray(level.mapCellContents.data))
		return [];

	const d = level.delta_index(this.direction)

	let matches = [];
	const cellRowMasks = this.cellRowMasks;
	const cellRowMasks_Movements = this.cellRowMasks_Movements;
	for (let cellRowIndex = 0; cellRowIndex < this.patterns.length; cellRowIndex++) {
		const cellRow = this.patterns[cellRowIndex];
		const matchFunction = this.cellRowMatches[cellRowIndex];
		let match;
		if (this.ellipsisCount[cellRowIndex] === 0) {
			match = state.matchCellRow(level, this.direction, matchFunction, cellRow, cellRowMasks[cellRowIndex], cellRowMasks_Movements[cellRowIndex], d);
		} else { // ellipsiscount===1/2
			match = state.matchCellRowWildCard(this.direction, matchFunction, cellRow, cellRowMasks[cellRowIndex], cellRowMasks_Movements[cellRowIndex], d, this.ellipsisCount[cellRowIndex]);
		}
		if (match.length === 0) {
			return [];
		} else {
			matches.push(match);
		}
	}
	return matches;
};

Rule.prototype.directional = function () {
	//Check if other rules in its rulegroup with the same line number.
	for (let i = 0; i < state.rules.length; i++) {
		const rg = state.rules[i];
		let copyCount = 0;
		for (let j = 0; j < rg.length; j++) {
			if (this.lineNumber === rg[j].lineNumber) {
				copyCount++;
			}
			if (copyCount > 1) {
				return true;
			}
		}
	}

	return false;
}

function IF(condition) {
	if (condition) {
		return "";
	} else {
		return "/*";
	}
}

function IF_LAZY(condition, fn) {
	if (condition) {
		return fn();
	} else {
		return "";
	}
}

function IF_ELSE_LAZY(condition, fn_if, fn_else) {
	if (condition) {
		return fn_if();
	} else {
		return fn_else();
	}
}

function ENDIF(condition) {
	if (condition) {
		return "";
	} else {
		return "*/";
	}
}
function ELSE(condition) {
	if (condition) {
		return "/*";
	} else {
		return "*/";
	}
}
function ENDELSE(condition) {
	if (condition) {
		return "*/";
	} else {
		return "";
	}
}

function FOR(start, end, fn) {
	let result = "";
	for (let i = start; i < end; i++) {
		result += fn(i);
	}
	return result;
}

let CACHE_RULE_APPLYAT = {}
Rule.prototype.generateApplyAt = function (patterns, ellipsisCount, OBJECT_SIZE, MOVEMENT_SIZE) {
	const fn = `'use strict';
	//have to double check they apply 
	//(cf test ellipsis bug: rule matches two candidates, first replacement invalidates second)
	if (check)
	{
	${FOR(0, patterns.length, cellRowIndex => `
		{
			${IF(ellipsisCount[cellRowIndex] === 0)}
				if ( ! this.cellRowMatches[${cellRowIndex}](
					this.patterns[${cellRowIndex}], 
						tuple[${cellRowIndex}], 
						delta, level.objects, level.movements
						) )
				return false
			${ENDIF(ellipsisCount[cellRowIndex] === 0)}
			${IF(ellipsisCount[cellRowIndex] === 1)}
				if ( this.cellRowMatches[${cellRowIndex}](
						this.patterns[${cellRowIndex}], 
						tuple[${cellRowIndex}][0], 
						tuple[${cellRowIndex}][1]+1, 
							tuple[${cellRowIndex}][1], 
						delta, level.objects, level.movements
					).length === 0 )
					return false
			${ENDIF(ellipsisCount[cellRowIndex] === 1)}
			${IF(ellipsisCount[cellRowIndex] === 2)}
				if ( this.cellRowMatches[${cellRowIndex}](
						this.patterns[${cellRowIndex}], 
						tuple[${cellRowIndex}][0],  
						tuple[${cellRowIndex}][1]+tuple[${cellRowIndex}][2]+1, 
							tuple[${cellRowIndex}][1]+tuple[${cellRowIndex}][2], 
						tuple[${cellRowIndex}][1]+1, 
							tuple[${cellRowIndex}][1],  
						tuple[${cellRowIndex}][2]+1, 
							tuple[${cellRowIndex}][2], 
							delta, level.objects, level.movements
						).length === 0 )
					return false
			${ENDIF(ellipsisCount[cellRowIndex] === 2)}
		}`)}
	}

    let result=false;
	let anyellipses=false;

    //APPLY THE RULE
	${FOR(0, patterns.length, cellRowIndex => {
		const preRow = patterns[cellRowIndex];
		return `
			{
				let ellipse_index=0;
				let currentIndex = ${ellipsisCount[cellRowIndex] > 0 ? `tuple[${cellRowIndex}][0]` : `tuple[${cellRowIndex}]`}
				${FOR(0, preRow.length, cellIndex => `
					{
						${IF(preRow[cellIndex] === ellipsisPattern)}
							const k = tuple[${cellRowIndex}][1+ellipse_index];
							ellipse_index++;
							anyellipses=true;
							currentIndex += delta*k;
						${ELSE(preRow[cellIndex] === ellipsisPattern)}
							const preCell = this.patterns[${cellRowIndex}][${cellIndex}];
							result = preCell.replace(level,this, currentIndex) || result;
							currentIndex += delta;
						${ENDELSE(preRow[cellIndex] === ellipsisPattern)}
					}
				`)}
			}`
	}
	)}

	if (verbose_logging && result){
		let ruleDirection = dirMaskName[this.direction];
		if (!this.directional()){
			ruleDirection="";
		}

		let inspect_ID =  addToDebugTimeline(level,this.lineNumber);
		let gapMessage="";
		
		let logString = '<font color="green">Rule <a onclick="jumpToLine(' + this.lineNumber + ');" href="javascript:void(0);">' + this.lineNumber + '</a> ' + ruleDirection + ' applied' + gapMessage + '.</font>';
		consolePrint(logString,false,this.lineNumber,inspect_ID);
		
	}

    return result;
	`
	if (fn in CACHE_RULE_APPLYAT) {
		return CACHE_RULE_APPLYAT[fn];
	}
	return CACHE_RULE_APPLYAT[fn] = new Function("level", "tuple", "check", "delta", fn);
};

Rule.prototype.tryApply = function (level) {
	const delta = level.delta_index(this.direction);

	//get all cellrow matches
	let matches = this.findMatches(level);
	if (matches.length === 0) {
		return false;
	}

	let result = false;
	if (this.hasReplacements) {
		let tuples = generateTuples(matches);
		for (let tupleIndex = 0; tupleIndex < tuples.length; tupleIndex++) {
			let tuple = tuples[tupleIndex];
			let shouldCheck = tupleIndex > 0;
			let success = this.applyAt(level, tuple, shouldCheck, delta);
			result = success || result;
		}
	}

	if (matches.length > 0) {
		this.queueCommands();
	}
	return result;
};

Rule.prototype.queueCommands = function () {

	if (this.commands.length === 0) {
		return;
	}

	//commandQueue is an array of strings, message.commands is an array of array of strings (For messagetext parameter), so I search through them differently
	let preexisting_cancel = level.commandQueue.indexOf("cancel") >= 0;
	let preexisting_restart = level.commandQueue.indexOf("restart") >= 0;

	let currule_cancel = false;
	let currule_restart = false;
	for (let i = 0; i < this.commands.length; i++) {
		let cmd = this.commands[i][0];
		if (cmd === "cancel") {
			currule_cancel = true;
		} else if (cmd === "restart") {
			currule_restart = true;
		}
	}

	//priority cancel > restart > everything else
	//if cancel is the queue from other rules, ignore everything
	if (preexisting_cancel) {
		return;
	}
	//if restart is in the queue from other rules, only apply if there's a cancel present here
	if (preexisting_restart && !currule_cancel) {
		return;
	}

	//if you are writing a cancel or restart, clear the current queue
	if (currule_cancel || currule_restart) {
		level.commandQueue = [];
		level.commandQueueSourceRules = [];
		messagetext = "";
	}

	for (let i = 0; i < this.commands.length; i++) {
		const command = this.commands[i];
		let already = false;
		if (level.commandQueue.indexOf(command[0]) >= 0) {
			continue;
		}
		level.commandQueue.push(command[0]);
		level.commandQueueSourceRules.push(this);

		if (verbose_logging) {
			const lineNumber = this.lineNumber;
			const ruleDirection = dirMaskName[this.direction];
			const logString = '<font color="green">Rule <a onclick="jumpToLine(' + lineNumber.toString() + ');"  href="javascript:void(0);">' + lineNumber.toString() + '</a> triggers command ' + command[0] + '.</font>';
			consolePrint(logString, false, lineNumber, null);
		}

		if (command[0] === 'message') {
			messagetext = command[1];
		}
	}
};

function showTempMessage() {
	keybuffer = [];
	textMode = true;
	titleScreen = false;
	quittingMessageScreen = false;
	messageselected = false;
	ignoreNotJustPressedAction = true;
	tryPlayShowMessageSound();
	drawMessageScreen();
	canvasResize();
}

function processOutputCommands(commands) {
	for (let i = 0; i < commands.length; i++) {
		let command = commands[i];
		if (command.charAt(1) === 'f') {//identifies sfxN
			tryPlaySimpleSound(command);
		}
		if (unitTesting === false) {
			if (command === 'message') {
				showTempMessage();
			}
		}
	}
}

function applyRandomRuleGroup(level, ruleGroup) {
	let propagated = false;

	let matches = [];
	for (let ruleIndex = 0; ruleIndex < ruleGroup.length; ruleIndex++) {
		let rule = ruleGroup[ruleIndex];
		let ruleMatches = rule.findMatches(level);
		if (ruleMatches.length > 0) {
			let tuples = generateTuples(ruleMatches);
			for (let j = 0; j < tuples.length; j++) {
				let tuple = tuples[j];
				matches.push([ruleIndex, tuple]);
			}
		}
	}

	if (matches.length === 0) {
		return false;
	}

	let match = matches[Math.floor(RandomGen.uniform() * matches.length)];
	let ruleIndex = match[0];
	let rule = ruleGroup[ruleIndex];
	let tuple = match[1];
	let check = false;
	const delta = level.delta_index(rule.direction)
	let modified = rule.applyAt(level, tuple, check, delta);

	rule.queueCommands();

	return modified;
}


function applyRuleGroup(ruleGroup) {
    if (ruleGroup[0].isRandom) {
        return applyRandomRuleGroup(level, ruleGroup);
    }

    const MAX_LOOP_COUNT = 200;
    const GROUP_LENGTH = ruleGroup.length;
    const shouldLog = verbose_logging;
    let hasChanges = false;        
    let madeChangeThisLoop = true; 
    let loopcount = 0;
    
    while (madeChangeThisLoop && loopcount++ < MAX_LOOP_COUNT) {
        madeChangeThisLoop = false;
        let consecutiveFailures = 0;

        for (let ruleIndex = 0; ruleIndex < GROUP_LENGTH; ruleIndex++) {
            const rule = ruleGroup[ruleIndex];
            
            if (rule.tryApply(level)) {
                madeChangeThisLoop = true;
                consecutiveFailures = 0;
            } else {
                consecutiveFailures++;
                if (consecutiveFailures === GROUP_LENGTH) {
                    break;  // No rule can apply - exit early
                }
            }            
        }

        if (madeChangeThisLoop) {
            hasChanges = true;
            if (shouldLog) {
                debugger_turnIndex++;
                addToDebugTimeline(level, -2);
            }
        }
    }

    if (loopcount >= MAX_LOOP_COUNT) {
        logErrorCacheable("Got caught looping lots in a rule group :O", ruleGroup[0].lineNumber, true);
    }

    return hasChanges;
}

function applyRules(rules, loopPoint, bannedGroup) {
    let loopPropagated = false;
    let loopCount = 0;
    let ruleGroupIndex = 0;
	const RULES_COUNT = rules.length;
    while (ruleGroupIndex < RULES_COUNT) {
        // Apply rules if not banned
        if (!bannedGroup || !bannedGroup[ruleGroupIndex]) {
            loopPropagated = applyRuleGroup(rules[ruleGroupIndex]) || loopPropagated;
        }

        // Handle mid-sequence loop point
        if (loopPropagated && loopPoint[ruleGroupIndex] !== undefined) {
            ruleGroupIndex = loopPoint[ruleGroupIndex];
            loopPropagated = false;
            loopCount++;
            
            if (loopCount > 200) {
                logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", rules[ruleGroupIndex][0].lineNumber, true);
                break;
            }

            if (verbose_logging) {
                debugger_turnIndex++;
                addToDebugTimeline(level, -2);
            }
            continue;
        }

        ruleGroupIndex++;
        
        // Handle end-sequence loop point
        if (ruleGroupIndex === rules.length && loopPropagated && loopPoint[ruleGroupIndex] !== undefined) {
            ruleGroupIndex = loopPoint[ruleGroupIndex];
            loopPropagated = false;
            loopCount++;
            
            if (loopCount > 200) {
                logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", rules[ruleGroupIndex][0].lineNumber, true);
                break;
            }
        }

        if (verbose_logging) {
            debugger_turnIndex++;
            addToDebugTimeline(level, -2);
        }
    }
}

let CACHE_RESOLVEMOVEMENTS = {}
function generate_resolveMovements(OBJECT_SIZE, MOVEMENT_SIZE,state) {
	const fn = `'use strict';
		let moved=true;
		while(moved){
			moved=false;
			for (let i=0;i<level.n_tiles;i++) {
				moved = state.repositionEntitiesAtCell(level,i) || moved;
			}
		}
		let doUndo=false;
	
		//Search for any rigidly-caused movements remaining
		for (let i=0;i<level.n_tiles;i++) {
			let cellMask = level.getCellInto(i,_o6);
			let movementMask = level.getMovements(i);
			if (${IS_NONZERO("movementMask", MOVEMENT_SIZE)}) {

				${IF_LAZY(state.rigid,()=>`
					let rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];
				if (${IS_NONZERO("rigidMovementAppliedMask", MOVEMENT_SIZE)}) {
					${UNROLL("movementMask &= rigidMovementAppliedMask", MOVEMENT_SIZE)}
					if (${IS_NONZERO("movementMask", MOVEMENT_SIZE)}) 
				outer_area: {
						//find what layer was restricted
						${FOR(0,LAYER_COUNT,j=>`{
							let layerSection = ${GETSHIFTOR("movementMask", 0x1f, 5*j)};
							if (layerSection!==0) {
								//this is our layer!
								let rigidGroupIndexMask = level.rigidGroupIndexMask[i];
								let rigidGroupIndex = ${GETSHIFTOR("rigidGroupIndexMask", 0x1f, 5*j)};
								rigidGroupIndex--;//group indices start at zero, but are incremented for storing in the bitfield
								let groupIndex = state.rigidGroupIndex_to_GroupIndex[rigidGroupIndex];
								if (bannedGroup[groupIndex]!==true){
									bannedGroup[groupIndex]=true
									doUndo=true;
								}
								break outer_area;
							}
						}`)}
					}
				}`)}


				for (let j=0;j<state.sfx_MovementFailureMasks.length;j++) {
					let o = state.sfx_MovementFailureMasks[j];
					let objectMask = o.objectMask;
		
					if (${ANY_BITS_IN_COMMON("cellMask", "objectMask", OBJECT_SIZE)} 
					&& ${ANY_BITS_IN_COMMON("o.directionMask","movementMask", MOVEMENT_SIZE)} 
					&& seedsToPlay_CantMove.indexOf(o.seed)===-1) {
						seedsToPlay_CantMove.push(o.seed);
					}
				}
			}

			for (let j=0;j<STRIDE_MOV;j++) {
				level.movements[j+i*STRIDE_MOV]=0;
			}

			${IF(state.rigid)}
				${SET_ZERO("level.rigidGroupIndexMask[i]")}
				${SET_ZERO("level.rigidMovementAppliedMask[i]")}
			${ENDIF(state.rigid)}

		}
		return doUndo;
	`
	//	function resolveMovements(level, bannedGroup){
	if (fn in CACHE_RESOLVEMOVEMENTS) {
		return CACHE_RESOLVEMOVEMENTS[fn];
	}
	return CACHE_RESOLVEMOVEMENTS[fn] = new Function("level", "bannedGroup", fn);
}

let sfxCreateMask = null;
let sfxDestroyMask = null;

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

	// Initialize the variables we make use of in the course of our turn
	let bannedGroup = [];
	level.commandQueue = [];
	level.commandQueueSourceRules = [];
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
	state.calculateRowColMasks(level);
	let alreadyResolved = [];

	// The main loop (dominated by rigid, but if you ignore that nonsense, it's
	// not too complicated)!
	let i = 0;
	do {
		rigidloop = false;
		i++;

		//everything outside of these two lines in this loop is rigid-body nonsense
		applyRules(state.rules, state.loopPoint, bannedGroup);
		let shouldUndo = state.resolveMovements(level, bannedGroup);

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
			applyRules(state.lateRules, state.lateLoopPoint);
		}
	} while (i < 50 && rigidloop);

	if (i >= 50) {
		consolePrint("Looped through 50 times, gave up.  too many loops!");
	}

	// Post move stuff
	if (playerPositions.length > 0 && state.metadata.require_player_movement !== undefined) {
		let somemoved = false;
		for (let i = 0; i < playerPositions.length; i++) {
			let pos = playerPositions[i];
			let val = level.getCell(pos);
			if (state.playerMask[1].bitsClearInArray(val.data)) {
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
			tick_lazy_function_generation(false);
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
		dontDoWin = true;
	}

	if (state.winconditions.length === 0) {
		return false;
	}

	if (level.commandQueue.indexOf('win') >= 0) {
		if (runrulesonlevelstart_phase) {
			consolePrint("Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)");
		} else {
			consolePrint("Win Condition Satisfied");
		}
		if (!dontDoWin) {
			DoWin();
		}
		return;
	}

	let won = false;

	let passed = true;
	const WINCONDITIONS_LENGTH = state.winconditions.length;
	for (let wcIndex = 0; wcIndex < WINCONDITIONS_LENGTH; wcIndex++) {
		let wincondition = state.winconditions[wcIndex];
		let filter1 = wincondition[1];
		let filter2 = wincondition[2];
		let aggr1 = wincondition[4];
		let aggr2 = wincondition[5];

		let rulePassed = true;

		const f1 = aggr1 ? c => filter1.bitsSetInArray(c) : c => !filter1.bitsClearInArray(c);
		const f2 = aggr2 ? c => filter2.bitsSetInArray(c) : c => !filter2.bitsClearInArray(c);

		switch (wincondition[0]) {
			case -1://NO
				{
					for (let i = 0; i < level.n_tiles; i++) {
						let cell = level.getCellInto(i, _o10);
						if ((f1(cell.data)) &&
							(f2(cell.data))) {
							rulePassed = false;
							break;
						}
					}

					break;
				}
			case 0://SOME
				{
					let passedTest = false;
					for (let i = 0; i < level.n_tiles; i++) {
						let cell = level.getCellInto(i, _o10);
						if ((f1(cell.data)) &&
							(f2(cell.data))) {
							passedTest = true;
							break;
						}
					}
					if (passedTest === false) {
						rulePassed = false;
					}
					break;
				}
			case 1://ALL
				{
					for (let i = 0; i < level.n_tiles; i++) {
						let cell = level.getCellInto(i, _o10);
						if ((f1(cell.data)) &&
							(!f2(cell.data))) {
							rulePassed = false;
							break;
						}
					}
					break;
				}
		}
		if (rulePassed === false) {
			passed = false;
		}
	}
	won = passed;

	if (won) {
		if (runrulesonlevelstart_phase) {
			consolePrint("Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)");
		} else {
			consolePrint("Win Condition Satisfied");
		}
		if (!dontDoWin) {
			DoWin();
		}
	}
}

function DoWin() {
	if (winning) {
		return;
	}
	againing = false;
	tryPlayEndLevelSound();
	if (unitTesting) {
		nextLevel();
		return;
	}

	winning = true;
	timer = 0;
}

function nextLevel() {
	againing = false;
	messagetext = "";
	if (state && state.levels && (curlevel > state.levels.length)) {
		curlevel = state.levels.length - 1;
	}

	ignoreNotJustPressedAction = true;
	if (titleScreen) {
		if (titleSelection === 0) {
			//new game
			curlevel = 0;
			curlevelTarget = null;
		}
		if (curlevelTarget !== null) {
			loadLevelFromStateTarget(state, curlevel, curlevelTarget);
		} else {
			loadLevelFromState(state, curlevel);
		}
	} else {
		if (hasUsedCheckpoint) {
			curlevelTarget = null;
			hasUsedCheckpoint = false;
		}
		if (curlevel < (state.levels.length - 1)) {
			curlevel++;
			curlevelTarget = null;
			textMode = false;
			tick_lazy_function_generation(false);
			tick_lazy_function_generation(false);
			titleScreen = false;
			quittingMessageScreen = false;
			messageselected = false;
			loadLevelFromState(state, curlevel);			
		} else {
			try {
				storage_remove(document.URL);
				storage_remove(document.URL + '_checkpoint');
			} catch (ex) {

			}

			curlevel = 0;
			curlevelTarget = null;
			goToTitleScreen();
			tryPlayEndGameSound();
		}
		//continue existing game
	}
	try {
		storage_set(document.URL, curlevel);
		if (curlevelTarget !== null) {
			restartTarget = level4Serialization();
			let backupStr = JSON.stringify(restartTarget);
			storage_set(document.URL + '_checkpoint', backupStr);
		} else {
			storage_remove(document.URL + "_checkpoint");
		}
	} catch (ex) {

	}

	if (state !== undefined && state.metadata.flickscreen !== undefined) {
		oldflickscreendat = [0, 0, Math.min(state.metadata.flickscreen[0], level.width), Math.min(state.metadata.flickscreen[1], level.height)];
	}
	canvasResize();
}

function goToTitleScreen() {
	againing = false;
	messagetext = "";
	titleScreen = true;
	textMode = true;
	doSetupTitleScreenLevelContinue();
	titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
	generateTitleScreen();
	if (canvas !== null) {//otherwise triggers error in cat bastard test
		regenSpriteImages();
	}
}

let CACHE_RULE_FINDMATCHES = {}
Rule.prototype.generateFindMatchesFunction = function () {
	let fn = '';

	// Initial mask check
	fn += `if (${NOT_BITS_SET_IN_ARRAY("this.ruleMask", "level.mapCellContents.data", STRIDE_OBJ)}) return [];\n`;
	fn += 'const d = level.delta_index(this.direction);\n';
	fn += 'const matches = [];\n';

	// Unroll the pattern matching loop
	for (let i = 0; i < this.patterns.length; i++) {
		fn += `let match${i};\n`;

		// Generate specialized matching code based on ellipsis count
		if (this.ellipsisCount[i] === 0) {
			fn += `match${i} = state.matchCellRow(level,this.direction, this.cellRowMatches[${i}], ` +
				`this.patterns[${i}], this.cellRowMasks[${i}], ` +
				`this.cellRowMasks_Movements[${i}], d);\n`;
		} else if (this.ellipsisCount[i] === 1) {
			fn += `match${i} = state.matchCellRowWildCard(this.direction, this.cellRowMatches[${i}], ` +
				`this.patterns[${i}], this.cellRowMasks[${i}], ` +
				`this.cellRowMasks_Movements[${i}], d, 1);\n`;
		} else { // ellipsisCount === 2
			fn += `match${i} = state.matchCellRowWildCard(this.direction, this.cellRowMatches[${i}], ` +
				`this.patterns[${i}], this.cellRowMasks[${i}], ` +
				`this.cellRowMasks_Movements[${i}], d, 2);\n`;
		}

		// Early return if no matches
		fn += `if (match${i}.length === 0) return [];\n`;
		fn += `matches.push(match${i});\n`;
	}

	fn += 'return matches;';

	if (fn in CACHE_RULE_FINDMATCHES) {
		return CACHE_RULE_FINDMATCHES[fn];
	}
	return CACHE_RULE_FINDMATCHES[fn] = new Function('level', fn);
}