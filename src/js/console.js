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


function consolePrintFromRule(text,rule,urgent) {

	if (urgent===undefined) {
		urgent=false;
	}


	var ruleDirection = dirMaskName[rule.direction];

	var logString = '<font color="green">Rule <a onclick="jumpToLine(' + rule.lineNumber + ');"  href="javascript:void(0);">' + 
			rule.lineNumber + '</a> ' + ruleDirection + " : "  + text + '</font>';

	if (cache_console_messages&&urgent==false) {		
		consolecache.push([logString,null,null,1]);
	} else {
		addToConsole(logString);
	}
}

function consolePrint(text,urgent,linenumber,inspect_ID) {

	if (urgent===undefined) {
		urgent=false;
	}


	if (cache_console_messages && urgent===false) {		
		consolecache.push([text,linenumber,inspect_ID,1]);
	} else {
		consoleCacheDump();
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
	
	//pass 1 : aggregate identical messages
	for (var i = 0; i < consolecache.length-1; i++) {
		var this_row = consolecache[i];
		var this_row_text=this_row[0];

		var next_row = consolecache[i+1];
		var next_row_text=next_row[0];

		if (this_row_text===next_row_text){			
			consolecache.splice(i,1);
			i--;
			//need to preserve visual_id from later one
			next_row[3]=this_row[3]+1;
		}
	}

	var batched_messages=[];
	var current_batch_row=[];
	//pass 2 : group by debug visibility
	for (var i=0;i<consolecache.length;i++){
		var row = consolecache[i];

		var message = row[0];
		var lineNumber = row[1];
		var inspector_ID = row[2];
		var count = row[3];
		
		if (i===0||lineNumber==null){
			current_batch_row=[lineNumber,inspector_ID,[row]]; 
			batched_messages.push(current_batch_row);
			continue;
		} 

		var batch_lineNumber = current_batch_row[0];
		var batch_inspector_ID = current_batch_row[1];

		if (inspector_ID===null && lineNumber==batch_lineNumber){
			current_batch_row[2].push(row);
		} else {
			current_batch_row=[lineNumber,inspector_ID,[row]]; 
			batched_messages.push(current_batch_row);
		}
	}

	var summarised_message = "<br>";
	for (var j=0;j<batched_messages.length;j++){
		var batch_row = batched_messages[j];
		var batch_lineNumber = batch_row[0];
		var inspector_ID = batch_row[1];
		var batch_messages = batch_row[2];
		
		summarised_message+="<br>"

		if (inspector_ID!= null){
			summarised_message+=`<span class="hoverpreview" onmouseover="debugPreview(${inspector_ID})" onmouseleave="debugUnpreview()">`;
		}
		for (var i = 0; i < batch_messages.length; i++) {

			if(i>0){
				summarised_message+=`<br><span class="noeye_indent"></span>`
			}
			var curdata = batch_messages[i];
			var curline = curdata[0];
			var times_repeated = curdata[3];
			if (times_repeated>1){
				curline += ` (x${times_repeated})`;
			}
			summarised_message += curline
		}

		if (inspector_ID!= null){
			summarised_message+=`</span>`;
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
		
	//clear up debug stuff.
	debugger_turnIndex=0;
	debug_visualisation_array=[];
	diffToVisualize=null;
}

var clearConsoleClick = document.getElementById("clearConsoleClick");
clearConsoleClick.addEventListener("click", clearConsole, false);

function UnitTestingThrow(error){}