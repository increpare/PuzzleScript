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
	// I never want a game causing the compiler to somehow throw errors to stopping you from saving it
	try {		
		compile(["rebuild"]);//to regenerate/extract title
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
	if (storage_has('saves')) {
		curSaveArray = JSON.parse(storage_get('saves'));
	}

	if (curSaveArray.length>20) {
		curSaveArray.splice(0,1);
	}
	curSaveArray.push(saveDat);


	let savesDatStr = JSON.stringify(curSaveArray);
	storage_set('saves',savesDatStr);

	repopulateSaveDropdown(curSaveArray);

	let loadDropdown = document.getElementById('loadDropDown');
	loadDropdown.selectedIndex=0;

	setEditorClean();

	consolePrint("saved file to local storage",true);

	if (window.location.href.indexOf("?hack")>=0){
		let currURL= window.location.href; 
		let afterDomain= currURL.substring(currURL.lastIndexOf('/') + 1);
		let beforeQueryString= afterDomain.split("?")[0];  
 
		window.history.pushState({}, document.title, "./" +beforeQueryString);
	}
	//clear parameters from url bar if any present
	if (curSaveArray.length===20){
		consolePrint("WARNING: your <i>locally saved file list</i> has reached its maximum capacity of 20 files - older saved files will be deleted when you save in future.",true);
	}
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

function loadDropDownChange() {

	if(!canExit()) {
 		this.selectedIndex = 0;
 		return;
 	}

	let saveString = storage_get('saves');
	if (saveString === null) {
			consolePrint("Eek, trying to load a file, but there's no local storage found. Eek!",true);
	} 

	saves = JSON.parse(saveString);
	
	for (let i=0;i<saves.length;i++) {
		let sd = saves[i];
	    let key = dateToReadable(sd.title,new Date(sd.date));
	    if (key==this.value) {

	    	let saveText = sd.text;
			editor.setValue(saveText);
			clearConsole();
			setEditorClean();
			let loadDropdown = document.getElementById('loadDropDown');
			loadDropdown.selectedIndex=0;
			unloadGame();
			compile(["restart"]);
			return;
	    }
	}		

	consolePrint("Eek, trying to load a save, but couldn't find it. :(",true);
}


function repopulateSaveDropdown(saves) {
	let loadDropdown = document.getElementById('loadDropDown');
	loadDropdown.options.length = 0;

	if (saves===undefined) {
		try {
			if (!storage_has('saves')) {
				return;
			} else {
				saves = JSON.parse(storage_get("saves"));
			}
		} catch (ex) {
			return;
		}
	}

    let optn = document.createElement("OPTION");
    optn.text = "Load";
    optn.value = "Load";
    loadDropdown.options.add(optn);  

	for (let i=saves.length-1;i>=0;i--) {			
		let sd = saves[i];
	    let optn = document.createElement("OPTION");
	    let key = dateToReadable(sd.title,new Date(sd.date));
	    optn.text = key;
	    optn.value = key;
	    loadDropdown.options.add(optn);  
	}
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

if(lightMode){
	document.body.style.colorScheme = 'light'
	document.body.classList.add('light-theme');
} else {
	document.body.style.colorScheme = 'dark'
	document.body.classList.add('dark-theme');
}
generateTitleScreen();
regenSpriteImages();


function toggleThemeClick() {
	if (document.body.style.colorScheme === 'light') {
		document.body.style.colorScheme = 'dark';
		document.body.classList.remove('light-theme');
		document.body.classList.add('dark-theme');
	} else {
		document.body.style.colorScheme = 'light';
		document.body.classList.remove('dark-theme');
		document.body.classList.add('light-theme');
	}
	localStorage.setItem("lightMode", document.body.style.colorScheme==='light');
	if (state.levels.length===0){
		generateTitleScreen();
		regenSpriteImages();
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
	let title = "Untitled PuzzleScript Script";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title + " (PuzzleScript Script)";
	}
	
	compile(["rebuild"]);

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



