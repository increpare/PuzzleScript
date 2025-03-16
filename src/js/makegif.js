function makeGIF() {
	var randomseed = RandomGen.seed;
	levelEditorOpened=false;
	var targetlevel=curlevel;
	var gifcanvas = document.createElement('canvas');
	gifcanvas.width=screenwidth*cellwidth;
	gifcanvas.height=screenheight*cellheight;
	gifcanvas.style.width=screenwidth*cellwidth;
	gifcanvas.style.height=screenheight*cellheight;

	var gifctx = gifcanvas.getContext('2d', {willReadFrequently: true});

	var inputDat = inputHistory.concat([]);
	var soundDat = soundHistory.concat([]);
	

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

  	for(var i=0;i<inputDat.length;i++) {
  		var realtimeframe=false;
		var val=inputDat[i];
		if (val==="undo") {
			DoUndo(false,true);
		} else if (val==="restart") {
			DoRestart();
		} else if (val=="tick") {			
			processInput(-1);
			realtimeframe=true;
		} else {
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
	}

	encoder.finish();
	const data_url = 'data:image/gif;base64,'+btoa(encoder.stream().getData());
	consolePrint('<img class="generatedgif" src="'+data_url+'">');
	const gametitle = state.metadata.title ? state.metadata.title : 'puzzlescript-anim';
	var filename = gametitle.replace(/\s+/g, '-').toLowerCase()+'.gif';
	//also remove double-quotes (this actually the only important bit tbh)
	filename = filename.replace(/"/g,'');
	consolePrint('<a href="'+data_url+'" download="'+filename+'">Download GIF</a>');
  	
  	unitTesting = false;

    inputHistory = inputDat;
	soundHistory = soundDat;
}
