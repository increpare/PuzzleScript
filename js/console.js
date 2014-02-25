function jumpToLine(i) {

    var code = parent.form1.code;

    var editor = code.editorreference;

    // editor.getLineHandle does not help as it does not return the reference of line.
    editor.scrollIntoView(i - 1 - 10);
    editor.scrollIntoView(i - 1 + 10);
    editor.scrollIntoView(i - 1);
    editor.setCursor(i - 1, 0);
}

function playSound(seed) {
	var params = generateFromSeed(seed);
	params.sound_vol = SOUND_VOL;
	params.sample_rate = SAMPLE_RATE;
	params.sample_size = SAMPLE_SIZE;
	var sound = generate(params);
	var audio = new Audio();
	audio.src = sound.dataURI;
	audio.play();
}

var consolecache = "";
function consolePrint(text) {
	if (cache_console_messages) {		
		consolecache = consolecache + '<br'> + text;
	} else {
		var code = document.getElementById('consoletextarea');
		code.innerHTML = code.innerHTML + '<br>'+ text;
		var objDiv = document.getElementById('lowerarea');
		objDiv.scrollTop = objDiv.scrollHeight;
	}
}

function consoleCacheDump() {
	if (cache_console_messages===false) {
		return;
	}
	var code = document.getElementById('consoletextarea');
	code.innerHTML = code.innerHTML + consolecache;
	consolecache="";
	var objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
	cache_console_messages=false;
}

function consoleError(text) {	
        var errorString = '<span class="errorText">' + text + '</span>';
        consolePrint(errorString);
}
function clearConsole() {
	var code = document.getElementById('consoletextarea');
	code.innerHTML = '';
	var objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
}

var clearConsoleClick = document.getElementById("clearConsoleClick");
clearConsoleClick.addEventListener("click", clearConsole, false);