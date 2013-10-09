var keyRepeatTimer=0;
var keyRepeatIndex=0;

var dragging=false;
var rightdragging=false;
var columnAdded=false;

function recalcLevelBounds(){
	level.movementMask = level.dat.concat([]);
    level.rigidMovementAppliedMask = level.dat.concat([]);
	level.rigidGroupIndexMask = level.dat.concat([]);
    level.commandQueue=[];

    for (var i=0;i<level.movementMask.length;i++)
    {
        level.movementMask[i]=0;
        level.rigidMovementAppliedMask[i]=0;
        level.rigidGroupIndexMask[i]=0;
    }
}
function addLeftColumn() {
	var bgMask = 1<<state.backgroundid;
	for (var i=0;i<level.height;i++) {
		level.dat.splice(i,0,bgMask);
	}
	level.width++;
	recalcLevelBounds();
	columnAdded=true;
}

function addRightColumn(){
	var bgMask = 1<<state.backgroundid;
	for (var i=0;i<level.height;i++) {
		level.dat.push(bgMask);
	}
	level.width++;
	recalcLevelBounds();
	columnAdded=true;
}

function addTopRow(){
	var bgMask = 1<<state.backgroundid;
	for (var i=level.width-1;i>=0;i--) {
		level.dat.splice(i*level.height,0,bgMask);
	}
	level.height++;
	recalcLevelBounds();
	columnAdded=true;
}
function addBottomRow(){
	var bgMask = 1<<state.backgroundid;
	for (var i=level.width-1;i>=0;i--) {
		level.dat.splice(level.height+i*level.height,0,bgMask);
	}
	level.height++;
	recalcLevelBounds();
	columnAdded=true;
}

function removeLeftColumn() {
	if (level.width<=1) {
		return;
	}
	var bgMask = 1<<state.backgroundid;
	for (var i=0;i<level.height;i++) {
		level.dat.splice(0,1);
	}
	level.width--;
	recalcLevelBounds();
	columnAdded=true;
}

function removeRightColumn(){
	if (level.width<=1) {
		return;
	}
	var bgMask = 1<<state.backgroundid;
	for (var i=0;i<level.height;i++) {
		level.dat.splice(level.dat.length-1,1);
	}
	level.width--;
	recalcLevelBounds();
	columnAdded=true;
}

function removeTopRow(){
	if (level.height<=1) {
		return;
	}
	var bgMask = 1<<state.backgroundid;
	for (var i=level.width-1;i>=0;i--) {
		level.dat.splice(i*level.height,1);
	}
	level.height--;
	recalcLevelBounds();
	columnAdded=true;
}
function removeBottomRow(){
	if (level.height<=1) {
		return;
	}
	var bgMask = 1<<state.backgroundid;
	for (var i=level.width-1;i>=0;i--) {
		level.dat.splice(level.height+i*level.height,1);
	}
	level.height--;
	recalcLevelBounds();
	columnAdded=true;
}

var m1  = 0x55555555; //binary: 0101...
var m2  = 0x33333333; //binary: 00110011..
var m4  = 0x0f0f0f0f; //binary:  4 zeros,  4 ones ...
var m8  = 0x00ff00ff; //binary:  8 zeros,  8 ones ...
var m16 = 0x0000ffff; //binary: 16 zeros, 16 ones ...
var hff = 0xffffffff; //binary: all ones
var h01 = 0x01010101; //the sum of 256 to the power of 0,1,2,3...

//from http://jsperf.com/hamming-weight/4
function CountBits(x) {
    x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
    x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
    x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
    x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
    x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
    return x;
}

function matchGlyph(inputmask,maskToGlyph) {
	if (inputmask in maskToGlyph) {
		return maskToGlyph[inputmask];
	}

	//if what you have doesn't fit, look for mask with the most bits that does
	var highestbitcount=-1;
	var highestmask;
	for (var glyphmask in maskToGlyph) {
		if (maskToGlyph.hasOwnProperty(glyphmask)) {
			//require all bits of glyph to be in input
			if (glyphmask == (glyphmask&inputmask)) {
				var bitcount = CountBits(glyphmask);			
				if (bitcount>highestbitcount) {
					highestbitcount=bitcount;
					highestmask=maskToGlyph[glyphmask];
				}
			}
		}
	}
	if (highestbitcount>0) {
		return highestmask;
	}
	return maskToGlyph[0];
}

var htmlEntityMap = {
	"&": "&amp;",
	"<": "&lt;",
	">": "&gt;",
	'"': '&quot;',
	"'": '&#39;',
	"/": '&#x2F;'
};

function printLevel() {
	var maskToGlyph = {};
	var glyphmask = 0;
	for (var glyphName in state.glyphDict) {
		if (state.glyphDict.hasOwnProperty(glyphName)&&glyphName.length===1) {
			var glyph = state.glyphDict[glyphName];
			var glyphmask=0;
			for (var i=0;i<glyph.length;i++)
			{
				var id = glyph[i];
				if (id>=0) {
					glyphmask = (glyphmask|(1<<id));
				}			
			}
			maskToGlyph[glyphmask]=glyphName;
			//register the same - backgroundmask with the same name
			var  bgMask = state.layerMasks[state.backgroundid];
			var glyphmaskMinusBackground = glyphmask & (~bgMask);
			if (! (glyphmask in maskToGlyph)) {
				maskToGlyph[glyphmask]=glyphName;
			}
			for (var i=0;i<32;i++) {
				var bgid = 1<<i;
				if ((bgid&bgMask)!==0) {
					var glyphmasnewbg = glyphmaskMinusBackground|bgid;
					if (! (glyphmasnewbg in maskToGlyph)) {
						maskToGlyph[glyphmasnewbg]=glyphName;						
					}
				}
			}
		}
	}
	var output="Printing level contents:<br><br>";
	for (var j=0;j<level.height;j++) {
		for (var i=0;i<level.width;i++) {
			var cellIndex = j+i*level.height;
			var cellMask = level.dat[cellIndex];
			var glyph = matchGlyph(cellMask,maskToGlyph);
			if (glyph in htmlEntityMap) {
				glyph = htmlEntityMap[glyph]; 
			}
			output = output+glyph;
		}
		output=output+"<br>";
	}
	consolePrint(output);
}

function levelEditorClick(event,click) {
	if (mouseCoordY<=-2) {
		var ypos = editorRowCount-(-mouseCoordY-2)-1;
		var newindex=mouseCoordX+(screenwidth-1)*ypos;
		if (mouseCoordX===-1) {
			printLevel();
		} else if (mouseCoordX>=0&&newindex<glyphImages.length) {
			glyphSelectedIndex=newindex;
			redraw();
		}

	} else if (mouseCoordX>-1&&mouseCoordY>-1&&mouseCoordX<screenwidth-2&&mouseCoordY<screenheight-2-editorRowCount	) {
		var glyphname = glyphImagesCorrespondance[glyphSelectedIndex];
		var glyph = state.glyphDict[glyphname];
		var glyphmask = 1<<state.backgroundid;
		for (var i=0;i<glyph.length;i++)
		{
			var id = glyph[i];
			if (id>=0) {
				glyphmask = (glyphmask|(1<<id));
			}			
		}

		var backgroundMask = state.layerMasks[state.backgroundlayer];
		if ((glyphmask&backgroundMask)===0) {
			// If we don't already have a background layer, mix in
			// the default one.
			glyphmask = glyphmask|(1<<state.backgroundid);
		}

		var coordIndex = mouseCoordY + mouseCoordX*level.height;
		level.dat[coordIndex]=glyphmask;
		redraw();
	}
	else if (click) {
		if (mouseCoordX===-1) {
			//add a left row to the map
			addLeftColumn();			
			canvasResize();
		} else if (mouseCoordX===screenwidth-2) {
			addRightColumn();
			canvasResize();
		} 
		if (mouseCoordY===-1) {
			addTopRow();
			canvasResize();
		} else if (mouseCoordY===screenheight-2-editorRowCount) {
			addBottomRow();
			canvasResize();
		}
	}
}

function levelEditorRightClick(event,click) {
	if (mouseCoordY===-2) {
		if (mouseCoordX<=glyphImages.length) {
			glyphSelectedIndex=mouseCoordX;
			redraw();
		}
	} else if (mouseCoordX>-1&&mouseCoordY>-1&&mouseCoordX<screenwidth-2&&mouseCoordY<screenheight-2-editorRowCount	) {
		var glyphname = glyphImagesCorrespondance[glyphSelectedIndex];
		var glyph = state.glyphDict[glyphname];
		var glyphmask = 1<<state.backgroundid;
		var coordIndex = mouseCoordY + mouseCoordX*level.height;
		level.dat[coordIndex]=glyphmask;
		redraw();
	}
	else if (click) {
		if (mouseCoordX===-1) {
			//add a left row to the map
			removeLeftColumn();			
			canvasResize();
		} else if (mouseCoordX===screenwidth-2) {
			removeRightColumn();
			canvasResize();
		} 
		if (mouseCoordY===-1) {
			removeTopRow();
			canvasResize();
		} else if (mouseCoordY===screenheight-2-editorRowCount) {
			removeBottomRow();
			canvasResize();
		}
	}
}

function onMouseDown(event) {
	if (event.button===0 && !(event.ctrlKey||event.metaKey) ) {
        lastDownTarget = event.target;
        keybuffer=[];
        if (event.target===canvas) {
        	setMouseCoord(event);
        	dragging=true;
        	rightdragging=false;
        	if (levelEditorOpened) {
        		return levelEditorClick(event,true);
        	}
        }
        dragging=false;
        rightdragging=false; 
    } else if (event.button===2 || (event.button===0 && (event.ctrlKey||event.metaKey)) ) {

	    dragging=false;
	    rightdragging=true;
        	if (levelEditorOpened) {
        		return levelEditorRightClick(event,true);
        	}
    }

}

function rightClickCanvas(event) {
    return prevent(event);
}

function onMouseUp(event) {
	dragging=false;
    rightdragging=false;
}

function onKeyDown(event) {
    event = event || window.event;
    if (keybuffer.indexOf(event.keyCode)>=0) {
    	return;
    }

    if(lastDownTarget === canvas) {
    	if (keybuffer.indexOf(event.keyCode)===-1) {
    		keybuffer.splice(keyRepeatIndex,0,event.keyCode);
	    	keyRepeatTimer=0;
	    	checkKey(event,true);
		}
	}


    if (canDump===true) {
        if (event.keyCode===74 && (event.ctrlKey||event.metaKey)) {//ctrl+j
            dumpTestCase();
        } else if (event.keyCode===75 && (event.ctrlKey||event.metaKey)) {//ctrl+k
            makeGIF();
        } 
    }
}

function relMouseCoords(event){
    var totalOffsetX = 0;
    var totalOffsetY = 0;
    var canvasX = 0;
    var canvasY = 0;
    var currentElement = this;

    do{
        totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
        totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
    }
    while(currentElement = currentElement.offsetParent)

    canvasX = event.pageX - totalOffsetX;
    canvasY = event.pageY - totalOffsetY;

    return {x:canvasX, y:canvasY}
}
HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

function onKeyUp(event) {
	event = event || window.event;
	var index=keybuffer.indexOf(event.keyCode);
	if (index>=0){
    	keybuffer.splice(index,1);
    	if (keyRepeatIndex>=index){
    		keyRepeatIndex--;
    	}
    }
}

var mouseCoordX=0;
var mouseCoordY=0;

function setMouseCoord(e){
    var coords = canvas.relMouseCoords(e);
    mouseCoordX=coords.x-xoffset;
	mouseCoordY=coords.y-yoffset;
	mouseCoordX=Math.floor(mouseCoordX/cellwidth);
	mouseCoordY=Math.floor(mouseCoordY/cellheight);
}

function mouseMove(event) {
    if (levelEditorOpened) {
    	setMouseCoord(event);  
    	if (dragging) { 	
    		levelEditorClick(event,false);
    	} else if (rightdragging){
    		levelEditorRightClick(event,false);    		
    	}
    }

    //window.console.log("showcoord ("+ canvas.width+","+canvas.height+") ("+x+","+y+")");
    redraw();
}

function mouseOut() {
//  window.console.log("clear");
}

    document.addEventListener('mousedown', onMouseDown, false);
    document.addEventListener('mouseup', onMouseUp, false);
    document.addEventListener('keydown', onKeyDown, false);
    document.addEventListener('keyup', onKeyUp, false);

function prevent(e) {
    if (e.preventDefault) e.preventDefault();
    if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    if (e.stopPropagation) e.stopPropagation();
    e.returnValue=false;
    return false;
}

function checkKey(e,justPressed) {

    if (winning) {
    	return;
    }
    var inputdir=-1;
    switch(e.keyCode) {
        case 65://a
        case 37: //left
        {
            window.console.log("LEFT");
            inputdir=1;
        break;
        }
        case 38: //up
        case 87: //w
        {
            window.console.log("UP");
            inputdir=0;
        break;
        }
        case 68://d
        case 39: //right
        {
            window.console.log("RIGHT");
            inputdir=3;
        break;
        }
        case 83://s
        case 40: //down
        {
            window.console.log("DOWN");
            inputdir=2;
        break;
        }
        case 13://enter
        case 32://space
        case 67://c
        case 88://x
        {
            window.console.log("ACTION");
            inputdir=4;
        break;
        }
        case 85://u
        case 90://z
        {
            //undo
            if (textMode===false) {

                if (canDump===true) {
                    inputHistory.push("undo");
                }
            	DoUndo();
            	return prevent(e);
            }
            break;
        }
        case 82://r
        {
        	if (textMode===false) {

                if (canDump===true) {
                    inputHistory.push("restart");
                }
        		DoRestart();
            	//restart
            	return prevent(e);
            }
            break;
        }
        case 27://escape
        {
        	if (titleScreen===false) {
				goToTitleScreen();	
		    	tryPlayTitleSound();
				canvasResize();			
				return prevent(e)
        	}
        	break;
        }
        case 69: {//e
        	if (canOpenEditor) {
        		levelEditorOpened=!levelEditorOpened;
        		canvasResize();
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
        	if (levelEditorOpened&&justPressed) {
        		var num=9;
        		if (e.keyCode>=49)  {
        			num = e.keyCode-49;
        		}

				if (num<glyphImages.length) {
					glyphSelectedIndex=num;
				} else {
					consolePrint("Trying to select tile outside of range in level editor.")
				}

        		canvasResize();
        		return prevent(e);
        	}		
        }
    }

    if (textMode) {
    	if (state.levels.length===0) {
    		//do nothing
    	} else if (titleScreen) {
    		if (titleMode===0) {
    			if (inputdir===4&&justPressed) {
    				if (titleSelected===false) {    				
						tryPlayStartGameSound();
	    				titleSelected=true;
	    				messageselected=false;
	    				timer=0;
	    				quittingTitleScreen=true;
	    				generateTitleScreen();
	    				canvasResize();
	    			}
    			}
    		} else {
    			if (inputdir==4&&justPressed) {
    				if (titleSelected===false) {    				
						tryPlayStartGameSound();
	    				titleSelected=true;
	    				messageselected=false;
	    				timer=0;
	    				quittingTitleScreen=true;
	    				generateTitleScreen();
	    				redraw();
	    			}
    			}
    			else if (inputdir===0||inputdir===2) {
    				titleSelection=1-titleSelection;
    				generateTitleScreen();
    				redraw();
    			}
    		}
    	} else {
    		if (inputdir==4&&justPressed) {    				
				if (unitTesting) {
					nextLevel();
					return;
				} else if (messageselected===false) {
    				messageselected=true;
    				timer=0;
    				quittingMessageScreen=true;
    				tryPlayCloseMessageSound();
    				titleScreen=false;
    				drawMessageScreen();
    			}
    		}
    	}
    } else {
	    if (!againing && inputdir>=0) {
            if (canDump===true) {
                inputHistory.push(inputdir);
            }
            if (inputdir===4 && ('noaction' in state.metadata)) {

            } else {
	        	processInput(inputdir);
	        }
	       	return prevent(e);
    	}
    }
}

function update() {
    timer+=deltatime;
    if (quittingTitleScreen) {
        if (timer/1000>0.3) {
            quittingTitleScreen=false;
            nextLevel();
        }
    }
    if (againing) {
    	if (timer>againinterval) {
    		againing=false;
    		processInput(-1);
    	}
    }
    if (quittingMessageScreen) {
        if (timer/1000>0.15) {
            quittingMessageScreen=false;
            if (messagetext==="") {
            	nextLevel();
            } else {
            	messagetext="";
            	textMode=false;
				titleScreen=false;
				titleMode=curlevel>0?1:0;
				titleSelected=false;
				titleSelection=0;
    			canvasResize();  
    			checkWin();          	
            }
        }
    }
    if (winning) {
        if (timer/1000>0.5) {
            winning=false;
            nextLevel();
        }
    }
    if (keybuffer.length>0) {
	    keyRepeatTimer+=deltatime;
	    if (keyRepeatTimer>repeatinterval/(Math.sqrt(keybuffer.length))) {
	    	keyRepeatTimer=0;
	    	keyRepeatIndex = (keyRepeatIndex+1)%keybuffer.length;
	    	var key = keybuffer[keyRepeatIndex];
	        checkKey({keyCode:key},false);
	    }
	}

    if (autotickinterval>0) {
        autotick+=deltatime;
        if (autotick>autotickinterval) {
            autotick=0;
            autoTickGame();
        }
    }
}

// Lights, camera…function!
setInterval(function() {
    update();
}, deltatime);
