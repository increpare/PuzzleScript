'use strict';

function runClick() {
	clearConsole();
	compile(["restart"]);
}

function dateToReadable(title,time) {
	let year = time.getFullYear();
	let month = time.getMonth()+1;
	let date1 = time.getDate();
	let hour = time.getHours();
	let minutes = time.getMinutes();
	let seconds = time.getSeconds();

	if (month < 10) {
    	month = "0"+month;
	}
	if (date1 < 10) {
		date1 = "0"+date1;
	}
	if (hour < 10) {
		hour = "0"+hour;
	}
	if (minutes < 10) {
		minutes = "0"+minutes;
	}
	if (seconds < 10) {
		seconds = "0"+seconds;
	}

	let result = hour+":"+minutes+" "+year + "-" + month+"-"+date1+" "+title;
	return result;
}


function saveClick() {
	suppress_all_console_output = true;
	if (saveToGroup("saves",true)){ 
		suppress_all_console_output = false;
		consolecache=[];
		consolePrint("saved file to local storage",true);
	} else{
		suppress_all_console_output = false;
		consolecache=[];
		consolePrint("no need to save, file is identical to last save",true);
	}
}

//every five minutes, do an autosave
function autosave(){
	suppress_all_console_output = true;
	if (saveToGroup("autosaves",false)){
		suppress_all_console_output = false;
		consolecache=[];
		consolePrint("autosaved file to local storage",true,undefined,undefined,false);
	} else{
		//no need to say anything
		suppress_all_console_output = false;
	}
}

setInterval(autosave, 10*60*1000);

function saveToGroup(savegroup,rebuild){
	// I never want a game causing the compiler to somehow throw errors to stopping you from saving it
	try {	
		if (rebuild){	
			compile(["rebuild"]);//to regenerate/extract title
		}
	} catch (error) {
		console.log(error);
	}
	let title = "Untitled";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
	}
	let text=editor.getValue();
	let saveDat = {
		title:title,
		text:text,
		date: new Date()
	}

	let curSaveArray = [];
	if (storage_has(savegroup)) {
		curSaveArray = JSON.parse(storage_get(savegroup));
	}

	//if you try to save a file that's identical as the last time you saved, delete the older version
	if (curSaveArray.length>0){
		if (curSaveArray[curSaveArray.length-1].text===text){
			return false;
		}
	}
	
	if (curSaveArray.length>20) {
		curSaveArray.splice(0,1);
	}
	curSaveArray.push(saveDat);


	let savesDatStr = JSON.stringify(curSaveArray);
	storage_set(savegroup,savesDatStr);

	repopulateSaveDropdown();

	let loadDropdown = document.getElementById('loadDropDown');
	loadDropdown.selectedIndex=0;

	setEditorClean();


	if (window.location.href.indexOf("?hack")>=0){
		let currURL= window.location.href; 
		let afterDomain= currURL.substring(currURL.lastIndexOf('/') + 1);
		let beforeQueryString= afterDomain.split("?")[0];  
 
		window.history.pushState({}, document.title, "./" +beforeQueryString);
	}
	return true;
}

window.addEventListener( "pageshow", function ( event ) {
	let historyTraversal = event.persisted || 
						   ( typeof window.performance != "undefined" && 
								window.performance.navigation.type === 2 );
	if ( historyTraversal ) {
	  // Handle page restore.
	  window.location.reload();
	}
  });

window.addEventListener("popstate", function(event){
	console.log("hey");
	location.reload();
});

//change event for select element
function loadDropDownChange(event) {
    if(!canExit()) {
        this.selectedIndex = 0;
        return;
    }
    
    // Get the selected option and determine which optgroup it belongs to
    let selectedOption = this.options[this.selectedIndex];
    let parentOptgroup = selectedOption.parentNode;
    
    // Skip if the "Load" option (not in an optgroup) is selected
    if (parentOptgroup.tagName !== 'OPTGROUP') {
        return;
    }
    
    // Determine which storage group to load from based on optgroup id
    let storageGroup = 'saves';
    if (parentOptgroup.id === 'loadDropdown_autosaves') {
        storageGroup = 'autosaves';
    }
    
    let saveString = storage_get(storageGroup);
    if (saveString === null) {
        consolePrint(`Eek, trying to load a file, but there's no local storage found for ${storageGroup}. Eek!`, true);
        return;
    }
    
    const saves = JSON.parse(saveString);
    
    for (let i = 0; i < saves.length; i++) {
        let sd = saves[i];
        let key = dateToReadable(sd.title, new Date(sd.date));
        if (key == this.value) {
            let saveText = sd.text;
            editor.setValue(saveText);
            clearConsole();
            setEditorClean();
            let loadDropdown = document.getElementById('loadDropDown');
            loadDropdown.selectedIndex = 0;
            unloadGame();
            compile(["restart"]);
            return;
        }
    }
    
    consolePrint("Eek, trying to load a save, but couldn't find it. :(", true);
}


function repopulateSaveDropdown() {
	let loadDropdown = document.getElementById('loadDropDown');

	loadDropdown.options.length = 0;

	let optn = document.createElement("OPTION");
	optn.text = "Load";
	optn.value = "Load";
	loadDropdown.insertBefore(optn,loadDropdown.firstChild);  

	function populateDropdownSection(optgroup_name,savegroup_name){
		let loadDropdownGroup = document.getElementById(optgroup_name);
		let saves;
		try {
			if (!storage_has(savegroup_name)) {
				return;
			} else {
				saves = JSON.parse(storage_get(savegroup_name));
			}
		} catch (ex) {
			return;
		}
		
		for (let i=saves.length-1;i>=0;i--) {			
			let sd = saves[i];
			let optn = document.createElement("OPTION");
			let key = dateToReadable(sd.title,new Date(sd.date));
			optn.text = key;
			optn.value = key;
			loadDropdownGroup.appendChild(optn);
		}	
	}

	populateDropdownSection("loadDropdown_userSaves","saves");
	populateDropdownSection("loadDropdown_autosaves","autosaves");

	//now for autosaves 
	// firstly, create a separator
	loadDropdown.selectedIndex=0;
}

repopulateSaveDropdown();
let loadDropdown = document.getElementById('loadDropDown');
loadDropdown.selectedIndex=0;

function levelEditorClick_Fn() {
	if (textMode || state.levels.length===0) {
		compile(["loadLevel",0]);
		levelEditorOpened=true;
    	canvasResize();
	} else {
		levelEditorOpened=!levelEditorOpened;
    	canvasResize();
    }
    lastDownTarget=canvas;	
}

let lightMode = false;

if (localStorage.hasOwnProperty("lightMode")){
	lightMode = localStorage.getItem("lightMode") == "true"; //returns stored value or null if not set
}

setColorScheme(lightMode);

function setColorScheme(light){
	if (light){
		document.body.style.colorScheme = 'light';
		document.body.classList.add('light-theme');
		document.body.classList.remove('dark-theme');
	} else {
		document.body.style.colorScheme = 'dark'
		document.body.classList.add('dark-theme');
		document.body.classList.remove('light-theme');		
	}
	generateTitleScreen();
	regenSpriteImages();
}
function toggleThemeClick() {
	let lightMode = document.body.style.colorScheme === 'light';
	lightMode = !lightMode;
	localStorage.setItem("lightMode", lightMode);
	setColorScheme(lightMode);
	if (state.levels.length===0){
		redraw();
	}
}


function printUnauthorized(){
	let authUrl = github_authURL();
	consolePrint(
			"<br>" +
			"PuzzleScript needs permission to share games through GitHub:<br>" +
			"<ul>" +
			"<li><a target=\"_blank\" href=\"" + authUrl + "\">Give PuzzleScript permission</a></li>" +
			"</ul>",true);
}

function shareClick() {
	if (!github_isSignedIn()) {
		printUnauthorized();
		return;
	}

	consolePrint("<br>Sending code to github...",true)

	
	compile(["rebuild"]);

	let title = "Untitled PuzzleScript Script";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title + " (PuzzleScript Script)";
	}
	
	let source=editor.getValue();
	github_save(title, source, function(id, e) {
		if (e !== null) {
			consoleError(e);
			if (!github_isSignedIn()) {
				printUnauthorized();
			}
			return;
		}

		let url = qualifyURL("play.html?p="+id);
		let editurl = qualifyURL("editor.html?hack="+id);
		let sourceCodeLink = "Link to source code:<br><a target=\"_blank\"  href=\""+editurl+"\">"+editurl+"</a>";

		consolePrint('GitHub (<a onclick="githubLogOut();"  href="javascript:void(0);">log out</a>) submission successful.<br>',true);
		consolePrint('<br>'+sourceCodeLink,true);


		if (errorCount>0) {
			consolePrint("<br>Cannot link directly to playable game, because there are compiler errors.",true);
		} else {
			consolePrint("<br>The game can now be played at this url:<br><a target=\"_blank\" href=\""+url+"\">"+url+"</a>",true);
		} 
	});
	lastDownTarget=canvas;	
}

function githubLogOut(){
	github_signOut();

	let authUrl = github_authURL();
	consolePrint(
		"<br>Logged out of Github.<br>" +
		"<ul>" +
		"<li><a target=\"_blank\" href=\"" + authUrl + "\">Give PuzzleScript permission</a></li>" +
		"</ul>"
				,true);
}

function rebuildClick() {
	compile(["rebuild"]);
}

function post_to_url(path, params, method) {
    method = method || "post"; // Set method to post by default if not specified.

    // The rest of this code assumes you are not using a library.
    // It can be made less wordy if you use one.
    let form = document.createElement("form");
    form.setAttribute("method", method);
    form.setAttribute("action", path);

    for(let key in params) {
        if(params.hasOwnProperty(key)) {
            let hiddenField = document.createElement("input");
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
	let sourceCode = editor.getValue();

	compile("restart");

	let sourceString = JSON.stringify(sourceCode);
	
	buildStandalone(sourceString);
}



