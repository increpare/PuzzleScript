var keybuffer = [];

function onMouseDown(event) {
        lastDownTarget = event.target;
        keybuffer={};
/*        if (lastDownTarget!==canvas) {
        }*/
       // alert('mousedown');
}

function onKeyDown(event) {
    event = event || window.event;
    if(lastDownTarget == canvas) {
	    if (keybuffer[event.keyCode]===undefined) {
	    	keybuffer[event.keyCode]=0;
	    	checkKey(event);
	    }
    }


    if (canDump===true) {
        if (event.keyCode===74 && (event.ctrlKey||event.metaKey)) {//ctrl+j
            dumpTestCase();
        } else if (event.keyCode===75 && (event.ctrlKey||event.metaKey)) {//ctrl+j
            makeGIF();
        } 
    }
}

function onKeyUp(event) {
	event = event || window.event;
/*	if(lastDownTarget == canvas) {
    }*/
    delete keybuffer[event.keyCode];
}


    document.addEventListener('mousedown', onMouseDown, false);


    document.addEventListener('keydown', onKeyDown, false);
    document.addEventListener('keyup', onKeyUp, false);

function prevent(e) {
    if (e.preventDefault) e.preventDefault();
    if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    if (e.stopPropagation) e.stopPropagation();
    e.returnValue=false;
    return false;
}

var messageselected=false;

function checkKey(e) {

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
    }

    if (textMode) {

    	if (state.levels.length===0) {
    		//do nothing
    	} else if (titleScreen) {
    		if (titleMode===0) {
    			if (inputdir===4) {
    				if (titleSelected===false) {    				
						tryPlayStartGameSound();
	    				titleSelected=true;
	    				timer=0;
	    				quittingTitleScreen=true;
	    				generateTitleScreen();
	    				canvasResize();
	    			}
    			}
    		} else {
    			if (inputdir==4) {
    				if (titleSelected===false) {    				
						tryPlayStartGameSound();
	    				titleSelected=true;
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
    		if (inputdir==4) {    				
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
	    if (inputdir>=0) {
            if (canDump===true) {
                inputHistory.push(inputdir);
            }
	        processInput(inputdir);
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
            }
        }
    }
    if (winning) {
        if (timer/1000>0.3) {
            winning=false;
            nextLevel();
        }
    }
    for(var n in keybuffer) {
        keybuffer[n]+=deltatime;
        if (keybuffer[n]>repeatinterval) {
            keybuffer[n]=0;
            checkKey({keyCode:parseInt(n)});
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
