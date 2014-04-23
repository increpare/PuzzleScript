function makeGIF() {
	var randomseed = RandomGen.seed;
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


	var encoder = new GIFEncoder();
	encoder.setRepeat(0); //auto-loop
	encoder.setDelay(200);
	encoder.start();

	compile(["loadLevel",curlevel],levelString,randomseed);
	canvasResize();
	redraw();
	gifctx.drawImage(canvas,-xoffset,-yoffset);
  	encoder.addFrame(gifctx);
	var autotimer=0;

    while(replayQueue.length) {
  		var realtimeframe=false;
        var val=replayQueue.pop();
        if (val==="undo") {
            pushInput("undo");
	        DoUndo();
        } else if (val==="restart") {
            pushInput("restart");
	    	DoRestart();
	    } else if (val==="tick") {
            autoTickGame();
			realtimeframe=true;
        } else if (val==="quit" || val==="win") {
            continue;
        } else {
            pushInput(val);
	    	processInput(val);
	    }
		redraw();
		gifctx.drawImage(canvas,-xoffset,-yoffset);
		encoder.addFrame(gifctx);
		encoder.setDelay(realtimeframe?autotickinterval:repeatinterval);
		autotimer+=repeatinterval;

		while (againing) {
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