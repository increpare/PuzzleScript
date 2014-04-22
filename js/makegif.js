function makeGIF() {
	levelEditorOpened=false;
	var targetlevel=curlevel;
	var gifcanvas = document.createElement('canvas');
	gifcanvas.width=screenwidth*cellwidth;
	gifcanvas.height=screenheight*cellheight;
	gifcanvas.style.width=screenwidth*cellwidth;
	gifcanvas.style.height=screenheight*cellheight;

	var gifctx = gifcanvas.getContext('2d');

	var inputDat = inputHistory.concat([]);
    replayQueue = inputDat.slice().reverse();

	unitTesting=true;
	levelString=compiledText;

	if (errorStrings.length>0) {
		throw(errorStrings[0]);
	}

	var encoder = new GIFEncoder();
	encoder.setRepeat(0); //auto-loop
	encoder.setDelay(200);
	encoder.start();

	compile(["loadLevel",curlevel],levelString);
	canvasResize();
	redraw();
	gifctx.drawImage(canvas,-xoffset,-yoffset);
  	encoder.addFrame(gifctx);

    while(replayQueue.length) {
        var val=replayQueue.pop();
        if(isNaN(val) && val.substr(0,6) == "random") {
            throw new Exception("Replay queue has unconsumed random "+val);
        }
        if (val==="undo") {
            pushInput("undo");
	        DoUndo();
        } else if (val==="restart") {
            pushInput("restart");
	    	DoRestart();
	    } else if (val==="wait") {
            autoTickGame();
        } else if (val==="quit" || val==="win") {
            continue;
        } else {
            pushInput(val);
	    	processInput(val);
	    }
	    while (againing) {
	    	againing=false;
	    	processInput(-1);
			redraw();
			encoder.setDelay(againinterval);
			gifctx.drawImage(canvas,-xoffset,-yoffset);
	  		encoder.addFrame(gifctx);	
	    }
		redraw();
		gifctx.drawImage(canvas,-xoffset,-yoffset);
  		encoder.addFrame(gifctx);
		encoder.setDelay(repeatinterval);
    }

  	encoder.finish();
  	var dat = 'data:image/gif;base64,'+encode64(encoder.stream().getData());
  	window.open(dat);
	unitTesting = false;
}