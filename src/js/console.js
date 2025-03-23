'use strict';

function jumpToLine(i) {

    let code = parent.form1.code;

    let editor = code.editorreference;

    // editor.getLineHandle does not help as it does not return the reference of line.
    let ll = editor.doc.lastLine();
    let low=i-1-10;    
    let high=i-1+10;    
    let mid=i-1;
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

let consolecache = [];


function consolePrintFromRule(text,rule,urgent) {

	if (urgent===undefined) {
		urgent=false;
	}


	let ruleDirection = dirMaskName[rule.direction];

	let logString = '<font color="green">Rule <a onclick="jumpToLine(' + rule.lineNumber + ');"  href="javascript:void(0);">' + 
			rule.lineNumber + '</a> ' + ruleDirection + " : "  + text + '</font>';

	if (cache_console_messages&&urgent==false) {		
		consolecache.push([logString,null,null,1]);
	} else {
		addToConsole(logString);
	}
}

function consolePrint(text,urgent,linenumber,inspect_ID,scrolldown=true) {

	if (urgent===undefined) {
		urgent=false;
	}


	if (cache_console_messages && urgent===false) {		
		consolecache.push([text,linenumber,inspect_ID,1]);
	} else {
		consoleCacheDump(scrolldown);
		addToConsole(text,scrolldown);
	}
}


let cache_n = 0;

function addToConsole(text,scrolldown=true) {
	if (suppress_all_console_output){
		return;
	}
	const cache = document.createElement("div");
	cache.id = "cache" + cache_n;
	cache.innerHTML = text;
	cache_n++;
	
	let code = document.getElementById('consoletextarea');
	code.appendChild(cache);
	consolecache=[];
	if (scrolldown){
		let objDiv = document.getElementById('lowerarea');
		objDiv.scrollTop = objDiv.scrollHeight;
	}
}

function consoleCacheDump(scrolldown=true) {
	if (suppress_all_console_output){
		return;
	}
	if (cache_console_messages===false) {
		return;
	}
	
	//pass 1 : aggregate identical messages
	for (let i = 0; i < consolecache.length-1; i++) {
		let this_row = consolecache[i];
		let this_row_text=this_row[0];

		let next_row = consolecache[i+1];
		let next_row_text=next_row[0];

		if (this_row_text===next_row_text){			
			consolecache.splice(i,1);
			i--;
			//need to preserve visual_id from later one
			next_row[3]=this_row[3]+1;
		}
	}

	let batched_messages=[];
	let current_batch_row=[];
	//pass 2 : group by debug visibility
	for (let i=0;i<consolecache.length;i++){
		let row = consolecache[i];

		let message = row[0];
		let lineNumber = row[1];
		let inspector_ID = row[2];
		let count = row[3];
		
		if (i===0||lineNumber==null){
			current_batch_row=[lineNumber,inspector_ID,[row]]; 
			batched_messages.push(current_batch_row);
			continue;
		} 

		let batch_lineNumber = current_batch_row[0];
		let batch_inspector_ID = current_batch_row[1];

		if (inspector_ID===null && lineNumber==batch_lineNumber){
			current_batch_row[2].push(row);
		} else {
			current_batch_row=[lineNumber,inspector_ID,[row]]; 
			batched_messages.push(current_batch_row);
		}
	}

	let summarised_message = "<br>";
	for (let j=0;j<batched_messages.length;j++){
		let batch_row = batched_messages[j];
		let batch_lineNumber = batch_row[0];
		let inspector_ID = batch_row[1];
		let batch_messages = batch_row[2];
		
		summarised_message+="<br>"

		if (inspector_ID!= null){
			summarised_message+=`<span class="hoverpreview" onmouseover="debugPreview(${inspector_ID})" onmouseleave="debugUnpreview()">`;
		}
		for (let i = 0; i < batch_messages.length; i++) {

			if(i>0){
				summarised_message+=`<br><span class="noeye_indent"></span>`
			}
			let curdata = batch_messages[i];
			let curline = curdata[0];
			let times_repeated = curdata[3];
			if (times_repeated>1){
				curline += ` (x${times_repeated})`;
			}
			summarised_message += curline
		}

		if (inspector_ID!= null){
			summarised_message+=`</span>`;
		}
	}


	addToConsole(summarised_message,scrolldown);
}

function consoleError(text) {	
        let errorString = '<span class="errorText">' + text + '</span>';
        consolePrint(errorString,true);
}
function clearConsole() {
	let code = document.getElementById('consoletextarea');
	code.innerHTML = '';
	let objDiv = document.getElementById('lowerarea');
	objDiv.scrollTop = objDiv.scrollHeight;
		
	//clear up debug stuff.
	debugger_turnIndex=0;
	debug_visualisation_array=[];
	diffToVisualize=null;
}

let clearConsoleClick = document.getElementById("clearConsoleClick");
clearConsoleClick.addEventListener("click", clearConsole, false);

function UnitTestingThrow(error){}