function jumpToLine(i) {

    var code = parent.form1.code;

    var editor = code.editorreference;

    // editor.getLineHandle does not help as it does not return the reference of line.
    editor.scrollIntoView(i - 1 - 10);
    editor.scrollIntoView(i - 1 + 10);
    editor.scrollIntoView(i - 1);
    editor.setCursor(i - 1, 0);
}

var consolecache = [];
function consolePrint(text,urgent) {
	if (urgent===undefined) {
		urgent=false;
	}
	if (cache_console_messages&&urgent==false) {		
		consolecache.push(text);
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
	
	var lastline = "";
	var times_repeated = 0;
	var summarised_message = "<br>";
	for (var i = 0; i < consolecache.length; i++) {
		if (consolecache[i] == lastline) {
			times_repeated++;
		} else {
			lastline = consolecache[i];
			if (times_repeated > 0) {
				summarised_message = summarised_message + " (x" + (times_repeated + 1) + ")";
			}
			summarised_message += "<br>"
			summarised_message += lastline;
			times_repeated = 0;
		}
	}
	
	var code = document.getElementById('consoletextarea');
	code.innerHTML = code.innerHTML + summarised_message;
	consolecache=[];
	var objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
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