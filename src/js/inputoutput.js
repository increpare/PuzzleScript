var keyRepeatTimer = 0;
var keyRepeatIndex = 0;
var input_throttle_timer = 0.0;
var lastinput = -100;

var dragging = false;
var rightdragging = false;
var columnAdded = false;

function selectText(containerid, e) {
	e = e || window.event;
	var myspan = document.getElementById(containerid);
	if (e && (e.ctrlKey || e.metaKey)) {
		var levelarr = ["console"].concat(myspan.innerText.split("\n"));
		var leveldat = levelFromString(state, levelarr);
		loadLevelFromLevelDat(state, leveldat, null);
		canvasResize();
	} else {
		if (document.selection) {
			var range = document.body.createTextRange();
			range.moveToElementText(myspan);
			range.select();
		} else if (window.getSelection) {
			var range = document.createRange();
			range.selectNode(myspan);
			var selection = window.getSelection();
			//why removeallranges? https://stackoverflow.com/a/43443101 whatever...
			selection.removeAllRanges();
			selection.addRange(range);
		}
	}
}

function recalcLevelBounds() {
}

function arrCopy(from, fromoffset, to, tooffset, len) {
	while (len--)
		to[tooffset++] = from[fromoffset]++;
}

function adjustLevel(level, widthdelta, heightdelta) {
	backups.push(backupLevel());
	var oldlevel = level.clone();
	level.width += widthdelta;
	level.height += heightdelta;
	level.n_tiles = level.width * level.height;
	level.objects = new Int32Array(level.n_tiles * STRIDE_OBJ);
	var bgMask = new BitVec(STRIDE_OBJ);
	bgMask.ibitset(state.backgroundid);
	for (var i = 0; i < level.n_tiles; ++i)
		level.setCell(i, bgMask);
	level.movements = new Int32Array(level.objects.length);
	columnAdded = true;
	RebuildLevelArrays();
	return oldlevel;
}

function addLeftColumn() {
	var oldlevel = adjustLevel(level, 1, 0);
	for (var x = 1; x < level.width; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index - level.height))
		}
	}
}

function addRightColumn() {
	var oldlevel = adjustLevel(level, 1, 0);
	for (var x = 0; x < level.width - 1; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index))
		}
	}
}

function addTopRow() {
	var oldlevel = adjustLevel(level, 0, 1);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 1; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index - x - 1))
		}
	}
}

function addBottomRow() {
	var oldlevel = adjustLevel(level, 0, 1);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 0; y < level.height - 1; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index - x));
		}
	}
}

function removeLeftColumn() {
	if (level.width <= 1) {
		return;
	}
	var oldlevel = adjustLevel(level, -1, 0);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index + level.height))
		}
	}
}

function removeRightColumn() {
	if (level.width <= 1) {
		return;
	}
	var oldlevel = adjustLevel(level, -1, 0);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index))
		}
	}
}

function removeTopRow() {
	if (level.height <= 1) {
		return;
	}
	var oldlevel = adjustLevel(level, 0, -1);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index + x + 1))
		}
	}
}
function removeBottomRow() {
	if (level.height <= 1) {
		return;
	}
	var oldlevel = adjustLevel(level, 0, -1);
	for (var x = 0; x < level.width; ++x) {
		for (var y = 0; y < level.height; ++y) {
			var index = x * level.height + y;
			level.setCell(index, oldlevel.getCell(index + x))
		}
	}
}

function matchGlyph(inputmask, glyphAndMask) {
	// find mask with closest match
	var highestbitcount = -1;
	var highestmask;
	for (var i = 0; i < glyphAndMask.length; ++i) {
		var glyphname = glyphAndMask[i][0];
		var glyphmask = glyphAndMask[i][1];
		var glyphbits = glyphAndMask[i][2];
		//require all bits of glyph to be in input
		if (glyphmask.bitsSetInArray(inputmask.data)) {
			var bitcount = 0;
			for (var bit = 0; bit < 32 * STRIDE_OBJ; ++bit) {
				if (glyphbits.get(bit) && inputmask.get(bit))
					bitcount++;
				if (glyphmask.get(bit) && inputmask.get(bit))
					bitcount++;
			}
			if (bitcount > highestbitcount) {
				highestbitcount = bitcount;
				highestmask = glyphname;
			}
		}
	}
	if (highestbitcount > 0) {
		return highestmask;
	}

	compiling = true; //i'm so sorry cf #999 "can't print level to console if too many approximations lol"
	logWarningNoLine("Wasn't able to approximate glyph values for some tiles, using '.' as a placeholder.", false);
	compiling = false;
	return '.';
}

var htmlEntityMap = {
	"&": "&amp;",
	"<": "&lt;",
	">": "&gt;",
	'"': '&quot;',
	"'": '&#39;',
	"/": '&#x2F;'
};

var selectableint = 0;

function printLevel() {
	try {
		errorCount = 0;
		errorStrings = [];
		var glyphMasks = [];

		for (var glyphName in state.glyphDict) {
			if (state.glyphDict.hasOwnProperty(glyphName) && glyphName.length === 1) {
				var glyph = state.glyphDict[glyphName];
				var glyphmask = new BitVec(STRIDE_OBJ);
				for (var i = 0; i < glyph.length; i++) {
					var id = glyph[i];
					if (id >= 0) {
						glyphmask.ibitset(id);
					}
				}
				var glyphbits = glyphmask.clone();
				//register the same - backgroundmask with the same name
				var bgMask = state.layerMasks[state.backgroundlayer];
				glyphmask.iclear(bgMask);
				glyphMasks.push([glyphName, glyphmask, glyphbits]);
			}
		}
		selectableint++;
		var tag = 'selectable' + selectableint;
		var output = "Printing level contents:<br><br><span id=\"" + tag + "\" onclick=\"selectText('" + tag + "',event)\">";
		for (var j = 0; j < level.height; j++) {
			for (var i = 0; i < level.width; i++) {
				var cellIndex = j + i * level.height;
				var cellMask = level.getCell(cellIndex);
				var glyph = matchGlyph(cellMask, glyphMasks);
				if (glyph in htmlEntityMap) {
					glyph = htmlEntityMap[glyph];
				}
				output = output + glyph;
			}
			if (j < level.height - 1) {
				output = output + "<br>";
			}
		}
		output += "</span><br><br>"
		consolePrint(output, true);
	} catch (e) {
		consolePrint("unable to print level contents because of errors", true);
	}
}

function levelEditorClick(event, click) {
	if (mouseCoordY <= -2) {
		var ypos = editorRowCount - (-mouseCoordY - 2) - 1;
		var newindex = mouseCoordX + (screenwidth - 1) * ypos;
		if (mouseCoordX === -1) {
			printLevel();
		} else if (mouseCoordX >= 0 && newindex < glyphImages.length) {
			glyphSelectedIndex = newindex;
			redraw();
		}

	} else if (mouseCoordX > -1 && mouseCoordY > -1 && mouseCoordX < screenwidth - 2 && mouseCoordY < screenheight - 2 - editorRowCount) {
		var glyphname = glyphImagesCorrespondance[glyphSelectedIndex];
		var glyph = state.glyphDict[glyphname];
		var glyphmask = new BitVec(STRIDE_OBJ);
		for (var i = 0; i < glyph.length; i++) {
			var id = glyph[i];
			if (id >= 0) {
				glyphmask.ibitset(id);
			}
		}

		var backgroundMask = state.layerMasks[state.backgroundlayer];
		if (glyphmask.bitsClearInArray(backgroundMask.data)) {
			// If we don't already have a background layer, mix in
			// the default one.
			glyphmask.ibitset(state.backgroundid);
		}

		var coordIndex = mouseCoordY + mouseCoordX * level.height;
		var getcell = level.getCell(coordIndex);
		if (getcell.equals(glyphmask)) {
			return;
		} else {
			if (anyEditsSinceMouseDown === false) {
				anyEditsSinceMouseDown = true;
				backups.push(backupLevel());
			}
			level.setCell(coordIndex, glyphmask);
			redraw();
		}
	}
	else if (click) {
		if (mouseCoordX === -1) {
			//add a left row to the map
			addLeftColumn();
			canvasResize();
		} else if (mouseCoordX === screenwidth - 2) {
			addRightColumn();
			canvasResize();
		}
		if (mouseCoordY === -1) {
			addTopRow();
			canvasResize();
		} else if (mouseCoordY === screenheight - 2 - editorRowCount) {
			addBottomRow();
			canvasResize();
		}
	}
}

function levelEditorRightClick(event, click) {
	if (mouseCoordY === -2) {
		if (mouseCoordX <= glyphImages.length) {
			glyphSelectedIndex = mouseCoordX;
			redraw();
		}
	} else if (mouseCoordX > -1 && mouseCoordY > -1 && mouseCoordX < screenwidth - 2 && mouseCoordY < screenheight - 2 - editorRowCount) {
		var coordIndex = mouseCoordY + mouseCoordX * level.height;
		var glyphmask = new BitVec(STRIDE_OBJ);
		glyphmask.ibitset(state.backgroundid);
		level.setCell(coordIndex, glyphmask);
		redraw();
	}
	else if (click) {
		if (mouseCoordX === -1) {
			//add a left row to the map
			removeLeftColumn();
			canvasResize();
		} else if (mouseCoordX === screenwidth - 2) {
			removeRightColumn();
			canvasResize();
		}
		if (mouseCoordY === -1) {
			removeTopRow();
			canvasResize();
		} else if (mouseCoordY === screenheight - 2 - editorRowCount) {
			removeBottomRow();
			canvasResize();
		}
	}
}

var anyEditsSinceMouseDown = false;

function onMouseDown(event) {

	if (event.handled) {
		return;
	}

	ULBS();

	var lmb = event.button === 0;
	var rmb = event.button === 2;
	if (event.type == "touchstart") {
		lmb = true;
	}
	if (lmb && (event.ctrlKey || event.metaKey)) {
		lmb = false;
		rmb = true;
	}

	if (lmb) {
		lastDownTarget = event.target;
		keybuffer = [];
		if (event.target === canvas || event.target.className === "tapFocusIndicator") {
			setMouseCoord(event);
			dragging = true;
			rightdragging = false;
			if (levelEditorOpened) {
				anyEditsSinceMouseDown = false;
				return levelEditorClick(event, true);
			}
		}
		dragging = false;
		rightdragging = false;
	} else if (rmb) {
		if (event.target === canvas || event.target.className === "tapFocusIndicator") {
			setMouseCoord(event);
			dragging = false;
			rightdragging = true;
			if (levelEditorOpened) {
				return levelEditorRightClick(event, true);
			}
		}
	}

	event.handled = true;

}

function rightClickCanvas(event) {
	return prevent(event);
}

function onMouseUp(event) {
	if (event.handled) {
		return;
	}

	dragging = false;
	rightdragging = false;

	event.handled = true;
}

function onKeyDown(event) {

	ULBS();

	event = event || window.event;

	// Prevent arrows/space from scrolling page
	if ((!IDE) && ([32, 37, 38, 39, 40].indexOf(event.keyCode) > -1)) {
		if (event && (event.ctrlKey || event.metaKey)) {
		} else {
			prevent(event);
		}
	}

	if ((!IDE) && event.keyCode === 77) {//m
		toggleMute();
	}


	if (keybuffer.indexOf(event.keyCode) >= 0) {
		return;
	}

	if (lastDownTarget === canvas || (window.Mobile && (lastDownTarget === window.Mobile.focusIndicator))) {
		if (keybuffer.indexOf(event.keyCode) === -1) {
			if (event && (event.ctrlKey || event.metaKey)) {
			} else {
				keybuffer.splice(keyRepeatIndex, 0, event.keyCode);
				keyRepeatTimer = 0;
				checkKey(event, !event.repeat);
			}
		}
	}


	if (canDump === true) {
		if (event.keyCode === 74 && (event.ctrlKey || event.metaKey)) {//ctrl+j
			dumpTestCase();
			prevent(event);
		} else if (event.keyCode === 75 && (event.ctrlKey || event.metaKey)) {//ctrl+k
			makeGIF();
			prevent(event);
		} else if (event.keyCode === 83 && (event.ctrlKey || event.metaKey)) {//ctrl+s
			saveClick();
			prevent(event);
		} else if (event.keyCode === 13 && (event.ctrlKey || event.metaKey)) {//ctrl+enter
			canvas.focus();
			editor.display.input.blur();
			if (event.shiftKey) {
				runClick();
			} else {
				rebuildClick();
			}
			prevent(event);
		}
	}
}

function relMouseCoords(event) {
	var totalOffsetX = 0;
	var totalOffsetY = 0;
	var canvasX = 0;
	var canvasY = 0;
	var currentElement = this;

	do {
		totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
		totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
	}
	while (currentElement = currentElement.offsetParent)

	if (event.touches == null) {
		canvasX = event.pageX - totalOffsetX;
		canvasY = event.pageY - totalOffsetY;
	} else {
		canvasX = event.touches[0].pageX - totalOffsetX;
		canvasY = event.touches[0].pageY - totalOffsetY;

	}

	return { x: canvasX, y: canvasY }
}
HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

function onKeyUp(event) {
	event = event || window.event;
	var index = keybuffer.indexOf(event.keyCode);
	if (index >= 0) {
		keybuffer.splice(index, 1);
		if (keyRepeatIndex >= index) {
			keyRepeatIndex--;
		}
	}
}

function onMyFocus(event) {
	keybuffer = [];
	keyRepeatIndex = 0;
	keyRepeatTimer = 0;
}

function onMyBlur(event) {
	keybuffer = [];
	keyRepeatIndex = 0;
	keyRepeatTimer = 0;
}

var mouseCoordX = 0;
var mouseCoordY = 0;

function setMouseCoord(e) {
	var coords = canvas.relMouseCoords(e);
	mouseCoordX = coords.x - xoffset;
	mouseCoordY = coords.y - yoffset;
	mouseCoordX = Math.floor(mouseCoordX / cellwidth);
	mouseCoordY = Math.floor(mouseCoordY / cellheight);
}

function mouseMove(event) {

	if (event.handled) {
		return;
	}

	if (levelEditorOpened) {
		setMouseCoord(event);
		if (dragging) {
			levelEditorClick(event, false);
		} else if (rightdragging) {
			levelEditorRightClick(event, false);
		}
		redraw();
	}

	event.handled = true;
	//window.console.log("showcoord ("+ canvas.width+","+canvas.height+") ("+x+","+y+")");
}

function mouseOut() {
	//  window.console.log("clear");
}

document.addEventListener('touchstart', onMouseDown, false);
document.addEventListener('touchmove', mouseMove, false);
document.addEventListener('touchend', onMouseUp, false);

document.addEventListener('mousedown', onMouseDown, false);
document.addEventListener('mouseup', onMouseUp, false);

document.addEventListener('keydown', onKeyDown, false);
document.addEventListener('keyup', onKeyUp, false);

window.addEventListener('focus', onMyFocus, false);
window.addEventListener('blur', onMyBlur, false);


function prevent(e) {
	if (e.preventDefault) e.preventDefault();
	if (e.stopImmediatePropagation) e.stopImmediatePropagation();
	if (e.stopPropagation) e.stopPropagation();
	e.returnValue = false;
	return false;
}

function checkKey(e, justPressed) {
	ULBS();

	if (winning) {
		return;
	}
	if (e && (e.ctrlKey || e.metaKey || e.altKey)) {
		return;
	}

	var inputdir = -1;
	switch (e.keyCode) {
		case 65://a
		case 37: //left
			{
				inputdir = 1;
				break;
			}
		case 38: //up
		case 87: //w
			{
				inputdir = 0;
				break;
			}
		case 68://d
		case 39: //right
			{
				inputdir = 3;
				break;
			}
		case 83://s
		case 40: //down
			{

				inputdir = 2;
				break;
			}
		case 80://p
			{
				printLevel();
				break;
			}
		case 13://enter
		case 32://space
		case 67://c
		case 88://x
			{
				//            window.console.log("ACTION");
				if (justPressed && ignoreNotJustPressedAction) {
					ignoreNotJustPressedAction = false;
				}
				if (justPressed === false && ignoreNotJustPressedAction) {
					return;
				}
				if (norepeat_action === false || justPressed) {
					inputdir = 4;
				} else {
					return;
				}
				break;
			}
		case 85://u
		case 90://z
			{
				//undo
				if (textMode === false) {
					pushInput("undo");
					DoUndo(false, true);
					canvasResize(); // calls redraw
					return prevent(e);
				}
				break;
			}
		case 82://r
			{
				if (textMode === false) {
					if (justPressed) {
						pushInput("restart");
						DoRestart();
						canvasResize(); // calls redraw
						return prevent(e);
					}
				}
				break;
			}
		case 27://escape
			{
				if (titleScreen === false) {
					goToTitleScreen();
					tryPlayTitleSound();
					canvasResize();
					return prevent(e)
				}
				break;
			}
		case 69: {//e
			if (canOpenEditor) {
				if (justPressed) {
					if (titleScreen) {
						if (state.title === "EMPTY GAME") {
							compile(["loadFirstNonMessageLevel"]);
						} else {
							nextLevel();
						}
					}
					levelEditorOpened = !levelEditorOpened;
					if (levelEditorOpened === false) {
						printLevel();
					}
					restartTarget = backupLevel();
					canvasResize();
				}
				return prevent(e);
			}
			break;
		}
		case 48://0
		case 49://1
		case 50://2
		case 51://3
		case 52://4
		case 53://5
		case 54://6
		case 55://7
		case 56://8
		case 57://9
			{
				if (levelEditorOpened && justPressed) {
					var num = 9;
					if (e.keyCode >= 49) {
						num = e.keyCode - 49;
					}

					if (num < glyphImages.length) {
						glyphSelectedIndex = num;
					} else {
						consolePrint("Trying to select tile outside of range in level editor.", true)
					}

					canvasResize();
					return prevent(e);
				}
				break;
			}
		case 189://-
		case 109://numpad -		
			{
				if (levelEditorOpened && justPressed) {
					if (glyphSelectedIndex > 0) {
						glyphSelectedIndex--;
						canvasResize();
						return prevent(e);
					}
				}
				break;
			}
		case 187://+
		case 107://numpad +
			{
				if (levelEditorOpened && justPressed) {
					if (glyphSelectedIndex + 1 < glyphImages.length) {
						glyphSelectedIndex++;
						canvasResize();
						return prevent(e);
					}
				}
				break;
			}
	}
	if (throttle_movement && inputdir >= 0 && inputdir <= 3) {
		if (lastinput == inputdir && input_throttle_timer < repeatinterval) {
			return;
		} else {
			lastinput = inputdir;
			input_throttle_timer = 0;
		}
	}
	if (textMode) {
		if (state.levels.length === 0) {
			//do nothing
		} else if (titleScreen) {
			if (quittingTitleScreen === false) {
				if (titleMode === 0) {
					if (inputdir === 4 && justPressed) {
						if (titleSelected === false) {
							tryPlayStartGameSound();
							titleSelected = true;
							messageselected = false;
							timer = 0;
							quittingTitleScreen = true;
							generateTitleScreen();
							canvasResize();
							clearInputHistory();
						}
					}
				} else {
					if (inputdir == 4 && justPressed) {
						if (titleSelected === false) {
							tryPlayStartGameSound();
							titleSelected = true;
							messageselected = false;
							timer = 0;
							quittingTitleScreen = true;
							generateTitleScreen();
							redraw();
						}
					}
					else if (inputdir === 0 || inputdir === 2) {
						if (inputdir === 0) {
							titleSelection = 0;
						} else {
							titleSelection = 1;
						}
						generateTitleScreen();
						redraw();
					}
				}
			}
		} else {
			if (inputdir == 4 && justPressed) {
				if (unitTesting) {
					nextLevel();
					return;
				} else if (messageselected === false) {
					messageselected = true;
					timer = 0;
					quittingMessageScreen = true;
					tryPlayCloseMessageSound();
					titleScreen = false;
					drawMessageScreen();
				}
			}
		}
	} else {
		if (!againing && inputdir >= 0) {
			if (inputdir === 4 && ('noaction' in state.metadata)) {

			} else {
				pushInput(inputdir);
				if (processInput(inputdir)) {
					redraw();
				}
			}
		}
	}
	
	return prevent(e);
}

function update() {
	const iterative_generation = textMode && !unitTesting;
	tick_lazy_function_generation(iterative_generation);
	const lastframe = get_title_animation_frame();
	timer += deltatime;
	input_throttle_timer += deltatime;
	if (quittingTitleScreen) {
		const thisFrame = get_title_animation_frame();
		if ((timer / 1000 > 0.3)&&WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length===0) {
			quittingTitleScreen = false;
			nextLevel();
		} else if (thisFrame > lastframe){
			generateTitleScreen();
			redraw();
		}
	}
	if (againing) {
		if (timer > againinterval && messagetext.length == 0) {
			if (processInput(-1)) {
				redraw();
				keyRepeatTimer = 0;
				autotick = 0;
			}
		}
	}
	if (quittingMessageScreen) {
		if (timer / 1000 > 0.15) {
			quittingMessageScreen = false;
			if (messagetext === "") {
				nextLevel();
			} else {
				messagetext = "";
				textMode = false;
				titleScreen = false;
				titleMode = (curlevel > 0 || curlevelTarget !== null) ? 1 : 0;
				titleSelected = false;
				ignoreNotJustPressedAction = true;
				titleSelection = 0;
				canvasResize();
				checkWin();
			}
		}
	}
	if (winning) {
		if (timer / 1000 > 0.5) {
			winning = false;
			nextLevel();
		}
	}
	if (keybuffer.length > 0) {
		keyRepeatTimer += deltatime;
		var ticklength = throttle_movement ? repeatinterval : repeatinterval / (Math.sqrt(keybuffer.length));
		if (keyRepeatTimer > ticklength) {
			keyRepeatTimer = 0;
			keyRepeatIndex = (keyRepeatIndex + 1) % keybuffer.length;
			var key = keybuffer[keyRepeatIndex];
			checkKey({ keyCode: key }, false);
		}
	}

	if (autotickinterval > 0 && !textMode && !levelEditorOpened && !againing && !winning) {
		autotick += deltatime;
		if (autotick > autotickinterval) {
			autotick = 0;
			pushInput("tick");
			if (processInput(-1)) {
				redraw();
			}
		}
	}
}

var prevTimestamp;
// Lights, camera…function!
var loop = function (timestamp) {
	if (prevTimestamp !== undefined) {
		deltatime = timestamp - prevTimestamp
	}
	prevTimestamp = timestamp
	update();
	// requestAnimationFrame will call loop() at the 
	// browser's refresh rate (generally 60fps)
	// and will auto pause/unpause when the window is minimized
	window.requestAnimationFrame(loop);
}

window.requestAnimationFrame(loop);
