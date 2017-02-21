function jumpToLine(i) {

    var code = parent.form1.code;

    var editor = code.editorreference;

    // editor.getLineHandle does not help as it does not return the reference of line.
    var ll = editor.doc.lastLine();
    var low=i-1-10;    
    var high=i-1+10;    
    var mid=i-1;
    if (low<0){
    	low=0;
    }
    if (high>ll){
    	high=ll;
    }
    if (mid>ll){
    	mid=ll;
    }

    editor.scrollIntoView(low);
    editor.scrollIntoView(high);
    editor.scrollIntoView(mid);
    editor.setCursor(mid, 0);
}

var consolecache = [];
function consolePrint(text,urgent) {
	if (level!=null && level.commandQueue!=null && level.commandQueue.indexOf('cancel')>=0 && text.indexOf('CANCEL')===-1) {	
		return;
	}

	if (urgent===undefined) {
		urgent=false;
	}
	if (cache_console_messages&&urgent==false) {		
		consolecache.push(text);
	} else {
		addToConsole(text);
	}
}


var cache_n = 0;

function addToConsole(text) {
	cache = document.createElement("div");
	cache.id = "cache" + cache_n;
	cache.innerHTML = text;
	cache_n++;
	
	var code = document.getElementById('consoletextarea');
	code.appendChild(cache);
	consolecache=[];
	var objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
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
	

	addToConsole(summarised_message);
}

function consoleError(text) {	
        var errorString = '<span class="errorText">' + text + '</span>';
        consolePrint(errorString,true);
}
function clearConsole() {
	var code = document.getElementById('consoletextarea');
	code.innerHTML = '';
	var objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
}

var clearConsoleClick = document.getElementById("clearConsoleClick");
clearConsoleClick.addEventListener("click", clearConsole, false);