'use strict';

let code = document.getElementById('code');
let _editorDirty = false;
let _editorCleanState = "";

let fileToOpen=getParameterByName("demo");
if (fileToOpen!==null&&fileToOpen.length>0) {
	tryLoadFile(fileToOpen);
	code.value = "loading...";
} else {
	let gistToLoad=getParameterByName("hack");
	if (gistToLoad!==null&&gistToLoad.length>0) {
		let id = gistToLoad.replace(/[\\\/]/,"");
		tryLoadGist(id);
		code.value = "loading...";
	} else {
		try {
			if (storage_has('saves')) {
					let curSaveArray = JSON.parse(storage_get('saves'));
					let sd = curSaveArray[curSaveArray.length-1];
					code.value = sd.text;
					let loadDropdown = document.getElementById('loadDropDown');
					loadDropdown.selectedIndex=0;
			}
		} catch(ex) {
			
		}
	}
}

CodeMirror.commands.swapLineUp = function(cm) {
    let ranges = cm.listSelections(), linesToMove = [], at = cm.firstLine() - 1, newSels = [];
    for (let i = 0; i < ranges.length; i++) {
      let range = ranges[i], from = range.from().line - 1, to = range.to().line;
      newSels.push({anchor: CodeMirror.Pos(range.anchor.line - 1, range.anchor.ch),
                    head: CodeMirror.Pos(range.head.line - 1, range.head.ch)});
    //   if (range.to().ch == 0 && !range.empty()) --to;
      if (from > at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
	if (linesToMove.length===0){
		return;
	}
    cm.operation(function() {
      for (let i = 0; i < linesToMove.length; i += 2) {
        let from = linesToMove[i], to = linesToMove[i + 1];
        let line = cm.getLine(from);
        cm.replaceRange("", CodeMirror.Pos(from, 0), CodeMirror.Pos(from + 1, 0), "+swapLine");
        if (to > cm.lastLine())
          cm.replaceRange("\n" + line, CodeMirror.Pos(cm.lastLine()), null, "+swapLine");
        else
          cm.replaceRange(line + "\n", CodeMirror.Pos(to, 0), null, "+swapLine");
      }
      cm.setSelections(newSels);
      cm.scrollIntoView();
    });
  };

  CodeMirror.commands.swapLineDown = function(cm) {
    let ranges = cm.listSelections(), linesToMove = [], at = cm.lastLine() + 1;
    for (let i = ranges.length - 1; i >= 0; i--) {
      let range = ranges[i], from = range.to().line + 1, to = range.from().line;
    //   if (range.to().ch == 0 && !range.empty()) from--;
      if (from < at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
    cm.operation(function() {
      for (let i = linesToMove.length - 2; i >= 0; i -= 2) {
        let from = linesToMove[i], to = linesToMove[i + 1];
        let line = cm.getLine(from);
        if (from == cm.lastLine())
          cm.replaceRange("", CodeMirror.Pos(from - 1), CodeMirror.Pos(from), "+swapLine");
        else
          cm.replaceRange("", CodeMirror.Pos(from, 0), CodeMirror.Pos(from + 1, 0), "+swapLine");
        cm.replaceRange(line + "\n", CodeMirror.Pos(to, 0), null, "+swapLine");
      }
      cm.scrollIntoView();
    });
  };

let editor = window.CodeMirror.fromTextArea(code, {
//	viewportMargin: Infinity,
	lineWrapping: true,
	lineNumbers: true,
	styleActiveLine: true,
	extraKeys: {
		"Ctrl-/": "toggleComment",
		"Cmd-/": "toggleComment",
		"Esc":CodeMirror.commands.clearSearch,
		"Shift-Ctrl-Up": "swapLineUp",
		"Shift-Ctrl-Down": "swapLineDown",
		}
	});
	
editor.on('mousedown', function(cm, event) {
  if (event.target.className == 'cm-SOUND') {
    let seed = parseInt(event.target.innerHTML);
    playSound(seed,true);
  } else if (event.target.className == 'cm-LEVEL') {
    if (event.ctrlKey||event.metaKey) {
	  document.activeElement.blur();  // unfocus code panel
	  editor.display.input.blur();
      prevent(event);         // prevent refocus
      compile(["levelline",cm.posFromMouse(event).line]);
    }
  }
});

_editorCleanState = editor.getValue();

function checkEditorDirty() {
	let saveLink = document.getElementById('saveClickLink');

	if (_editorCleanState !== editor.getValue()) {
		_editorDirty = true;
		if(saveLink) {
			saveLink.innerHTML = 'SAVE*';
		}
	} else {
		_editorDirty = false;
		if(saveLink) {
			saveLink.innerHTML = 'SAVE';
		}
	}
}

function setEditorClean() {
	_editorCleanState = editor.getValue();
	if (_editorDirty===true) {
		let saveLink = document.getElementById('saveClickLink');
		if(saveLink) {
			saveLink.innerHTML = 'SAVE';
		}
		_editorDirty = false;
	}
}

/* https://github.com/ndrake/PuzzleScript/commit/de4ac2a38865b74e66c1d711a25f0691079a290d */
editor.on('change', function(cm, changeObj) {
  // editor is dirty
  checkEditorDirty();
});

let mapObj = {
   parallel:"&#8741;",
   perpendicular:"&#8869;"
};

/*
editor.on("beforeChange", function(instance, change) {
    let startline = 
    for (let i = 0; i < change.text.length; ++i)
      text.push(change.text[i].replace(/parallel|perpendicular/gi, function(matched){ 
        return mapObj[matched];
      }));

    change.update(null, null, text);
});*/


code.editorreference = editor;
editor.setOption('theme', 'midnight');

function getParameterByName(name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    let regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    return results === null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

function tryLoadGist(id) {
	github_load(id, function(code, e) {
		if (e!==null) {
			consoleError(e);
			return;
		}
		editor.setValue(code);
		editor.clearHistory();
		clearConsole();
		setEditorClean();
		unloadGame();
		compile(["restart"],code);
	});
};

function tryLoadFile(fileName) {
	let fileOpenClient = new XMLHttpRequest();
	fileOpenClient.open('GET', 'demo/'+fileName+".txt");
	fileOpenClient.onreadystatechange = function() {
		
  		if(fileOpenClient.readyState!=4) {
  			return;
  		}
  		
		function doStuff(){
			editor.setValue(fileOpenClient.responseText);
			clearConsole();
			setEditorClean();
			unloadGame();
			compile(["restart"]);
		}
		if (document.readyState === "complete") {
			doStuff();
		} else {
			let handler = (event)=>{
				if (document.readyState === "complete") {
					doStuff();			
					document.removeEventListener("readystatechange",handler);							
				}
			};
			document.addEventListener("readystatechange",handler);
		}
	}
	fileOpenClient.send();
}

function canExit() {
 	if(!_editorDirty) {
 		return true;
 	}
 
 	return confirm("You haven't saved your game! Are you sure you want to lose your unsaved changes?")
}
 
function dropdownChange() {
	if(!canExit()) {
 		this.selectedIndex = 0;
 		return;
 	}

	tryLoadFile(this.value);
	this.selectedIndex=0;
}

editor.on('keyup', function (editor, event) {
	if (!CodeMirror.ExcludedIntelliSenseTriggerKeys[(event.keyCode || event.which).toString()])
	{
		CodeMirror.commands.autocomplete(editor, null, { completeSingle: false });
	}
});

function unescapeSlashes(str) {
	// add another escaped slash if the string ends with an odd
	// number of escaped slashes which will crash JSON.parse
	let parsedStr = str.replace(/(^|[^\\])(\\\\)*\\$/, "$&\\");

	// escape unescaped double quotes to prevent error with
	// added double quotes in json string
	parsedStr = parsedStr.replace(/(^|[^\\])((\\\\)*")/g, "$1\\$2");

	try {
		parsedStr = JSON.parse(`"${parsedStr}"`);
	} catch(e) {
		return str;
	}
	return parsedStr ;
}

function rip_source_from_html(s){
	let prebit=`sourceCode="`;
	let preindex = s.indexOf(prebit)+prebit.length;
	s = s.substring(preindex);
	let postbit=`";compile\(\["restart"\]`;
	let postindex = s.indexOf(postbit);
	s = s.substring(0,postindex);
	return unescapeSlashes(s);
}

editor.on("drop", function(editor, event) {
	files = event.dataTransfer.files;
	if (files.length > 0) {
		const file=files[0];
		try{
			reader = new FileReader();
			reader.onload = function(e) {
				let source_text = reader.result;
				//if filename ends with .html
				if (file.name.endsWith(".html")) {
					source_text = rip_source_from_html(source_text);
				} else if (!file.name.endsWith(".txt")) {
					consoleError("Only .html and .txt files are supported");
					return;
				} 
				editor.setValue(source_text);
				editor.clearHistory();
				consolePrint("Loaded file: " + file.name);
			};
			reader.readAsText(file);
		} catch(e) {
			consoleError(e);
		} finally{
			prevent(event);
		}
	}
});


function debugPreview(turnIndex,lineNumber){
	diffToVisualize=debug_visualisation_array[turnIndex][lineNumber];
	canvasResize(diffToVisualize.level);
}

function debugUnpreview(){
	diffToVisualize=null;
	canvasResize();
}

function addToDebugTimeline(level,lineNumber){

	if (!debug_visualisation_array.hasOwnProperty(debugger_turnIndex)){
		debug_visualisation_array[debugger_turnIndex]=[];
	}

	let debugTimelineSnapshot = {
		width:level.width,
		height:level.height,
		layerCount:level.layerCount,
		turnIndex:debugger_turnIndex,
		lineNumber:lineNumber,
		objects:new Int32Array(level.objects),
		movements:new Int32Array(level.movements),
		commandQueue:level.commandQueue.concat([]),
		commandQueueSourceRules:level.commandQueueSourceRules.concat([]),
		rigidMovementAppliedMask:level.rigidMovementAppliedMask.map(a=>a.clone()),
		level: level,
	};
	

	debug_visualisation_array[debugger_turnIndex][lineNumber]=debugTimelineSnapshot;
	return `${debugger_turnIndex},${lineNumber}`;
}
