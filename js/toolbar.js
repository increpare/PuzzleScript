function runClick() {
	clearConsole();
	compile(["restart"]);
}

function dateToReadable(title,time) {
	var year = time.getFullYear();
	var month = time.getMonth()+1;
	var date1 = time.getDate();
	var hour = time.getHours();
	var minutes = time.getMinutes();
	var seconds = time.getSeconds();
	var result = hour+":"+minutes+" "+year + "-" + month+"-"+date1+" "+title;
	return result;
}

function saveClick() {
	var title = "Untitled";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
	}
	var text=editor.getValue();
	var saveDat = {
		title:title,
		text:text,
		date: new Date()
	}

	var curSaveArray = [];
	if (localStorage['saves']===undefined) {

	} else {
		var curSaveArray = JSON.parse(localStorage.saves);
	}

	if (curSaveArray.length>19) {
		curSaveArray.splice(0,1);
	}
	curSaveArray.push(saveDat);
	var savesDatStr = JSON.stringify(curSaveArray);
	localStorage['saves']=savesDatStr;

	repopulateSaveDropdown(curSaveArray);
	consolePrint("saved file to local storage");
}



function loadDropDownChange() {
	var saveString = localStorage['saves'];
	if (saveString===undefined) {
			consolePrint("Eek, trying to load a file, but there's no local storage found. Eek!");
	} 

	saves = JSON.parse(saveString);
	
	for (var i=0;i<saves.length;i++) {
		var sd = saves[i];
	    var key = dateToReadable(sd.title,new Date(sd.date));
	    if (key==this.value) {
	    	var saveText = sd.text;
			editor.setValue(saveText);
			return;
	    }
	}
	
	consolePrint("Eek, trying to load a save, but couldn't find it. :(");
}


function repopulateSaveDropdown(saves) {
	var loadDropdown = document.getElementById('loadDropDown');
	loadDropdown.options.length = 0;

	if (saves===undefined) {
		if (localStorage['saves']===undefined) {
			return;
		} else {
			saves = JSON.parse(localStorage["saves"]);
		}
	}

    var optn = document.createElement("OPTION");
    optn.text = "";
    optn.value = "";
    loadDropdown.options.add(optn);  

	for (var i=saves.length-1;i>=0;i--) {			
		var sd = saves[i];
	    var optn = document.createElement("OPTION");
	    var key = dateToReadable(sd.title,new Date(sd.date));
	    optn.text = key;
	    optn.value = key;
	    loadDropdown.options.add(optn);  
	}
	loadDropdown.selectedIndex=1;
}

repopulateSaveDropdown();
var loadDropdown = document.getElementById('loadDropDown');
loadDropdown.selectedIndex=-1;

function levelEditorClick_Fn() {
	if (textMode || state.levels.length===0) {
		consolePrint("You can only open the editor when a level is open.");
	} else {
		levelEditorOpened=!levelEditorOpened;
    	canvasResize();
    }
    lastDownTarget=canvas;	
}

function shareClick() {
	consolePrint("Sending code to github...")
	var title = "Untitled PuzzleScript Script";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title + " PuzzleScript Script";
	}
	compile();

	if (errorCount>0) {
		consolePrint("Cannot share code that doesn't compile");
		return;
	}

	var source=editor.getValue();

	var gistToCreate = {
		"description" : "title",
		"public" : true,
		"files": {
			"script.txt" : {
				"content": source
			}
		}
	};

	var githubURL = 'https://api.github.com/gists';
	var githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open('POST', githubURL);
	githubHTTPClient.onreadystatechange = function() {		
		if(githubHTTPClient.readyState!=4) {
			return;
		}		
		var result = JSON.parse(githubHTTPClient.responseText);
		var id = result.id;
		var url = "http://www.puzzlescript.net/play/?p="+id;
		consolePrint("GitHub submission successful - the game can now be played at this url:<br><a href=\""+url+"\">"+url+"</a>");
	}
	githubHTTPClient.setRequestHeader("Content-type","application/x-www-form-urlencoded");
	var stringifiedGist = JSON.stringify(gistToCreate);
	githubHTTPClient.send(stringifiedGist);
    lastDownTarget=canvas;	
}

function rebuildClick() {
	clearConsole();
	compile(["rebuild"]);
}

function post_to_url(path, params, method) {
    method = method || "post"; // Set method to post by default if not specified.

    // The rest of this code assumes you are not using a library.
    // It can be made less wordy if you use one.
    var form = document.createElement("form");
    form.setAttribute("method", method);
    form.setAttribute("action", path);

    for(var key in params) {
        if(params.hasOwnProperty(key)) {
            var hiddenField = document.createElement("input");
            hiddenField.setAttribute("type", "hidden");
            hiddenField.setAttribute("name", key);
            hiddenField.setAttribute("value", params[key]);

            form.appendChild(hiddenField);
         }
    }

    document.body.appendChild(form);
    form.submit();
}

function exportClick() {
	var sourceCode = editor.getValue();

	compile("restart");

	var stateString = JSON.stringify(state);
	
	buildStandalone(stateString);
}



