var code = document.getElementById('code');
var _editorDirty = false;
var _editorCleanState = "";

var fileToOpen=getParameterByName("demo");
if (fileToOpen!==null&&fileToOpen.length>0) {
	tryLoadFile(fileToOpen);
	code.value = "loading...";
} else {
	var gistToLoad=getParameterByName("hack");
	if (gistToLoad!==null&&gistToLoad.length>0) {
		var id = gistToLoad.replace(/[\\\/]/,"");
		tryLoadGist(id);
		code.value = "loading...";
	} else {
		try {
			if (localStorage!==undefined && localStorage['saves']!==undefined) {
					var curSaveArray = JSON.parse(localStorage['saves']);
					var sd = curSaveArray[curSaveArray.length-1];
					code.value = sd.text;
					var loadDropdown = document.getElementById('loadDropDown');
					loadDropdown.selectedIndex=0;
			}
		} catch(ex) {
			
		}
	}
}



var editor = window.CodeMirror.fromTextArea(code, {
//	viewportMargin: Infinity,
	lineWrapping: true,
	lineNumbers: true,
    styleActiveLine: true,
    mode: "haxe",
    extraKeys: {"Ctrl-Space": "autocomplete"}
   	});
/*
CodeMirror.registerHelper("hintWords", "haxe",
*/

function isalnum (code){   
	if (!(code > 47 && code < 58) && // numeric (0-9)
		!(code > 64 && code < 91) && // upper alpha (A-Z)
		!(code > 96 && code < 123)) { // lower alpha (a-z)
	return false;
  }
  return true;
}


  var cls = "CodeMirror-Tern-";
  
  function makeTooltip(x, y, content) {
    var node = elt("div", cls + "tooltip", content);
    node.style.left = x + "px";
    node.style.top = y + "px";
    document.body.appendChild(node);
    return node;
  }


  function tempTooltip(cm, content, ts) {
    if (cm.state.ternTooltip) remove(cm.state.ternTooltip);
    var where = cm.cursorCoords();
    var tip = cm.state.ternTooltip = makeTooltip(where.right + 1, where.bottom, content);
    function maybeClear() {
      old = true;
      if (!mouseOnTip) clear();
    }
    function clear() {
      cm.state.ternTooltip = null;
      if (!tip.parentNode) return;
      cm.off("cursorActivity", clear);
      cm.off('blur', clear);
      cm.off('scroll', clear);
      fadeOut(tip);
    }
    var mouseOnTip = false, old = false;
    CodeMirror.on(tip, "mousemove", function() { mouseOnTip = true; });
    CodeMirror.on(tip, "mouseout", function(e) {
      if (!CodeMirror.contains(tip, e.relatedTarget || e.toElement)) {
        if (old) clear();
        else mouseOnTip = false;
      }
    });
    setTimeout(maybeClear, ts.options.hintDelay ? ts.options.hintDelay : 1700);
    cm.on("cursorActivity", clear);
    cm.on('blur', clear);
    cm.on('scroll', clear);
  }

function renderHint(elt,data,cur){
	var h = document.createElement("span")                // Create a <h1> element
	var t = document.createTextNode("Hello World");     // Create a text node
	h.appendChild(t);   

	elt.appendChild(h);//document.createTextNode(cur.displayText || getText(cur)));
}

var haxeHintWords =  ["Gfx.resizescreen","Gfx.clearscreen","Gfx.drawbox","Gfx.fillbox","Gfx.drawtri","Gfx.filltri","Gfx.drawcircle","Gfx.fillcircle","Gfx.drawhexagon","Gfx.fillhexagon","Gfx.drawline","Gfx.setlinethickness","Gfx.getpixel","Gfx.RGB","Gfx.HSL","Gfx.getred","Gfx.getgreen","Gfx.getblue","Gfx.screenwidth","Gfx.screenheight","Gfx.drawimage","Gfx.changetileset","Gfx.drawtile","Gfx.imagewidth","Gfx.imageheight","Gfx.tilewidth","Gfx.tileheight","Gfx.createimage","Gfx.createtiles","Gfx.numberoftiles","Gfx.drawtoscreen","Gfx.drawtoimage","Gfx.drawtotile","Gfx.copytile","Gfx.grabtilefromscreen","Gfx.grabtilefromimage","Gfx.grabimagefromscreen","Gfx.grabimagefromimage","Gfx.defineanimation","Gfx.drawanimation","Gfx.stopanimation","Col","Col.BLACK","Col.GREY","Col.WHITE","Col.RED","Col.PINK","Col.DARKBROWN","Col.BROWN","Col.ORANGE","Col.YELLOW","Col.DARKGREEN","Col.GREEN","Col.LIGHTGREEN","Col.NIGHTBLUE","Col.DARKBLUE","Col.BLUE","Col.LIGHTBLUE","Col.MAGENTA","Col","Text","Text.changefont","Text.changesize","Text.display","Text.input","Text.getinput","Music","Music.playsound","Key.A ","Key.B ","Key.C ","Key.D ","Key.E ","Key.F ","Key.G ","Key.H ","Key.I ","Key.J ","Key.K ","Key.L ","Key.M ","Key.N ","Key.O ","Key.P ","Key.Q ","Key.R ","Key.S ","Key.T ","Key.U ","Key.V ","Key.W ","Key.X ","Key.Y","Key.Z","Key.ZERO ","Key.ONE ","Key.TWO ","Key.THREE ","Key.FOUR ","Key.FIVE ","Key.SIX ","Key.SEVEN ","Key.EIGHT","Key.NINE","Key.F1 ","Key.F2 ","Key.F3 ","Key.F4 ","Key.F5 ","Key.F6 ","Key.F7 ","Key.F8 ","Key.F9 ","Key.F10 ","Key.F11","Key.F12","Key.MINUS","Key.PLUS","Key.DELETE","Key.BACKSPACE","Key.LBRACKET","Key.RBRACKET","Key.BACKSLASH","Key.CAPSLOCK","Key.SEMICOLON","Key.QUOTE","Key.COMMA","Key.PERIOD","Key.SLASH","Key.ESCAPE","Key.ENTER","Key.SHIFT","Key.CONTROL","Key.ALT","Key.SPACE","Key.UP","Key.DOWN","Key.LEFT","Key.RIGHT","Input.justpressed","Input.pressed","Input.justreleased","Input.delaypressed","Mouse.x","Mouse.y","Mouse.leftclick","Mouse.leftheld","Mouse.leftreleased","Mouse.middleclick","Mouse.middleheld","Mouse.middlereleased","Mouse.rightclick","Mouse.rightheld","Mouse.rightreleased","Mouse.mousewheel","Convert.tostring","Convert.toint","Convert.tofloat","Random.int","Random.float","Random.string","Random.bool","Random.occasional","Random.rare","Random.pickstring","Random.pickint","Random.pickfloat","Debug.log"];
var haxeHintDescriptions =  ["Gfx.resizescreen<hr>(width, height, scale)</hr>","Gfx.clearscreen","Gfx.drawbox","Gfx.fillbox","Gfx.drawtri","Gfx.filltri","Gfx.drawcircle","Gfx.fillcircle","Gfx.drawhexagon","Gfx.fillhexagon","Gfx.drawline","Gfx.setlinethickness","Gfx.getpixel","Gfx.RGB","Gfx.HSL","Gfx.getred","Gfx.getgreen","Gfx.getblue","Gfx.screenwidth","Gfx.screenheight","Gfx.drawimage","Gfx.changetileset","Gfx.drawtile","Gfx.imagewidth","Gfx.imageheight","Gfx.tilewidth","Gfx.tileheight","Gfx.createimage","Gfx.createtiles","Gfx.numberoftiles","Gfx.drawtoscreen","Gfx.drawtoimage","Gfx.drawtotile","Gfx.copytile","Gfx.grabtilefromscreen","Gfx.grabtilefromimage","Gfx.grabimagefromscreen","Gfx.grabimagefromimage","Gfx.defineanimation","Gfx.drawanimation","Gfx.stopanimation","Col","Col.BLACK","Col.GREY","Col.WHITE","Col.RED","Col.PINK","Col.DARKBROWN","Col.BROWN","Col.ORANGE","Col.YELLOW","Col.DARKGREEN","Col.GREEN","Col.LIGHTGREEN","Col.NIGHTBLUE","Col.DARKBLUE","Col.BLUE","Col.LIGHTBLUE","Col.MAGENTA","Col","Text","Text.changefont","Text.changesize","Text.display","Text.input","Text.getinput","Music","Music.playsound","Key.A ","Key.B ","Key.C ","Key.D ","Key.E ","Key.F ","Key.G ","Key.H ","Key.I ","Key.J ","Key.K ","Key.L ","Key.M ","Key.N ","Key.O ","Key.P ","Key.Q ","Key.R ","Key.S ","Key.T ","Key.U ","Key.V ","Key.W ","Key.X ","Key.Y","Key.Z","Key.ZERO ","Key.ONE ","Key.TWO ","Key.THREE ","Key.FOUR ","Key.FIVE ","Key.SIX ","Key.SEVEN ","Key.EIGHT","Key.NINE","Key.F1 ","Key.F2 ","Key.F3 ","Key.F4 ","Key.F5 ","Key.F6 ","Key.F7 ","Key.F8 ","Key.F9 ","Key.F10 ","Key.F11","Key.F12","Key.MINUS","Key.PLUS","Key.DELETE","Key.BACKSPACE","Key.LBRACKET","Key.RBRACKET","Key.BACKSLASH","Key.CAPSLOCK","Key.SEMICOLON","Key.QUOTE","Key.COMMA","Key.PERIOD","Key.SLASH","Key.ESCAPE","Key.ENTER","Key.SHIFT","Key.CONTROL","Key.ALT","Key.SPACE","Key.UP","Key.DOWN","Key.LEFT","Key.RIGHT","Input.justpressed","Input.pressed","Input.justreleased","Input.delaypressed","Mouse.x","Mouse.y","Mouse.leftclick","Mouse.leftheld","Mouse.leftreleased","Mouse.middleclick","Mouse.middleheld","Mouse.middlereleased","Mouse.rightclick","Mouse.rightheld","Mouse.rightreleased","Mouse.mousewheel","Convert.tostring","Convert.toint","Convert.tofloat","Random.int","Random.float","Random.string","Random.bool","Random.occasional","Random.rare","Random.pickstring","Random.pickint","Random.pickfloat","Debug.log"];
CodeMirror.registerHelper("hint", "haxe", 
	function(editor, options) {
		var cur = editor.getCursor();
		var line = editor.getLine(cur.line);
		var start = cur.ch;
		while (start>0){
			start--;
			if (isalnum(line.charCodeAt(start))||line[start]===".")  {
			} else {
				break;
			}
		}
		if (isalnum(line.charCodeAt(start))||line[start]===".")  {
		} else {
			start++;
		}
			
		var end=cur.ch-1;
		while (end<line.length-1){
			end++;
			if (isalnum(line.charCodeAt(end))||line[end]===".")  {
			} else {
				break;
			}
		}

		if (isalnum(line.charCodeAt(end))||line[end]===".")  {
		} else {
			end--;
		}
		end++;
		var token = line.substring(start,end);

		var matches=[];
		for (var i=0;i<haxeHintWords.length;i++){
			var w = haxeHintWords[i];
			if (w.length<token.length){
				continue;
			} 
			if (w.substring(0,token.length)==token){
				matches.push({text:w,displayText:haxeHintDescriptions[i],render:renderHint});
			}
		}
		return {list: matches, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
	}
);

// When an @ is typed, activate completion
editor.on("inputRead", function(editor, change) {
  if (change.text[0] == ".")
    editor.showHint();
});

editor.on('mousedown', function(cm, event) {
  if (event.target.className == 'cm-SOUND') {
    var seed = parseInt(event.target.innerHTML);
    playSound(seed);
  } else if (event.target.className == 'cm-LEVEL') {
    if (event.ctrlKey||event.metaKey) {
      document.activeElement.blur();  // unfocus code panel
      event.preventDefault();         // prevent refocus
      compile(["levelline",cm.posFromMouse(event).line]);
    }
  }
});

_editorCleanState = editor.getValue();

function checkEditorDirty() {
	var saveLink = document.getElementById('saveClickLink');

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
		var saveLink = document.getElementById('saveClickLink');
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

var mapObj = {
   parallel:"&#8741;",
   perpendicular:"&#8869;"
};

/*
editor.on("beforeChange", function(instance, change) {
    var startline = 
    for (var i = 0; i < change.text.length; ++i)
      text.push(change.text[i].replace(/parallel|perpendicular/gi, function(matched){ 
        return mapObj[matched];
      }));

    change.update(null, null, text);
});*/


code.editorreference = editor;
editor.setOption('theme', 'ambiance');

function getParameterByName(name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    return results === null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

function tryLoadGist(id) {
	var githubURL = 'https://api.github.com/gists/'+id;

	consolePrint("Contacting GitHub",true);
	var githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open('GET', githubURL);
	githubHTTPClient.onreadystatechange = function() {
	
		if(githubHTTPClient.readyState!=4) {
			return;
		}

		if (githubHTTPClient.responseText==="") {
			consoleError("GitHub request returned nothing.  A connection fault, maybe?");
		}

		var result = JSON.parse(githubHTTPClient.responseText);
		if (githubHTTPClient.status===403) {
			consoleError(result.message);
		} else if (githubHTTPClient.status!==200&&githubHTTPClient.status!==201) {
			consoleError("HTTP Error "+ githubHTTPClient.status + ' - ' + githubHTTPClient.statusText);
		} else {
			var code=result["files"]["script.txt"]["content"];
			editor.setValue(code);
			setEditorClean();
			unloadGame();
			compile(["restart"],code);
		}
	}
	githubHTTPClient.setRequestHeader("Content-type","application/x-www-form-urlencoded");
	githubHTTPClient.send();
}

function tryLoadFile(fileName) {
	var fileOpenClient = new XMLHttpRequest();
	fileOpenClient.open('GET', 'demo/'+fileName+".txt");
	fileOpenClient.onreadystatechange = function() {
		
  		if(fileOpenClient.readyState!=4) {
  			return;
  		}
  		
		editor.setValue(fileOpenClient.responseText);
		setEditorClean();
		unloadGame();
		compile(["restart"]);
	}
	fileOpenClient.send();
}

function dropdownChange() {
	tryLoadFile(this.value);
	this.selectedIndex=0;
}

