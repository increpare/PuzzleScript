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
    autoCloseBrackets: true,
    matchBrackets:true,
    tabSize:2,
    indentUnit:2,
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

function renderHint(elt,data,cur){
	var h = document.createElement("span")                // Create a <h1> element
	h.style.color="white";
	var t = document.createTextNode(cur.text);     // Create a text node
	h.appendChild(t);   
 


	elt.appendChild(h);//document.createTextNode(cur.displayText || getText(cur)));

	var h2 = document.createElement("span")                // Create a <h1> element
	h2.style.color="grey";
	var t2 = document.createTextNode(cur.displayText);     // Create a text node
	h2.appendChild(t2);  
	h2.style.color="#ff0000";
	elt.appendChild(t2);
}

var haxeHintArray = [
["Gfx.resizescreen","(width, height, scale)"],
["Gfx.clearscreen","(color)"],
["Gfx.clearscreeneachframe","(true or false)"],
["Gfx.drawbox","(x, y, width, height, col)"],
["Gfx.fillbox","(x, y, width, height, col, alpha)"],
["Gfx.drawtri","(x1, y1, x2, y2, x3, y3, col)"],
["Gfx.filltri","(x1, y1, x2, y2, x3, y3, col)"],
["Gfx.drawcircle","(x, y, radius, col)"],
["Gfx.fillcircle","(x, y, radius, col)"],
["Gfx.drawhexagon","(x, y, radius, angle, col)"],
["Gfx.fillhexagon","(x, y, radius, angle, col)"],
["Gfx.drawline","(x1, y1, x2, y2, col)"],
["Gfx.setlinethickness","(linethickness)"],
["Gfx.getpixel","(x, y):Int"],
["Gfx.RGB","(red (0-255), green (0-255), blue (0-255)):Int"],
["Gfx.HSL","(hue (0-360), saturation (0-1.0), lightness (0-1.0)):Int"],
["Gfx.getred","(col):Int"],
["Gfx.getgreen","(col):Int"],
["Gfx.getblue","(col):Int"],
["Gfx.screenwidth",":Int"],
["Gfx.screenheight",":Int"],
["Gfx.drawimage","(x, y, imagename, optional parameters)"],
["Gfx.changetileset","(newtileset)"],
["Gfx.drawtile","(x, y, tilenumber, optional parameters)"],
["Gfx.imagewidth","(imagename):Int"],
["Gfx.imageheight","(imagename):Int"],
["Gfx.tilewidth","():Int"],
["Gfx.tileheight","():Int"],
["Gfx.createimage","(imagename, width, height) "],
["Gfx.createtiles","(imagename, width, height, amount)"],
["Gfx.numberoftiles","():Int"],
["Gfx.drawtoscreen","()"],
["Gfx.drawtoimage","(imagename)"],
["Gfx.drawtotile","(tilenumber)"],
["Gfx.copytile","(to tile number, from tileset, from tile number)"],
["Gfx.grabtilefromscreen","(tilenumber, screen x, screen y)"],
["Gfx.grabtilefromimage","(tilenumber, imagename, image x, image y)"],
["Gfx.grabimagefromscreen","(imagename, screen x, screen y)"],
["Gfx.grabimagefromimage","(imagename, sourceimagename, image x, image y)"],
["Gfx.defineanimation","(animname, tileset, start frame, end frame, delay)"],
["Gfx.drawanimation","(x, y, animationname, optional parameters)"],
["Gfx.stopanimation","(animation name)"],
["Col.BLACK"],
["Col.GREY"],
["Col.WHITE"],
["Col.RED"],
["Col.PINK"],
["Col.DARKBROWN"],
["Col.BROWN"],
["Col.ORANGE"],
["Col.YELLOW"],
["Col.DARKGREEN"],
["Col.GREEN"],
["Col.LIGHTGREEN"],
["Col.NIGHTBLUE"],
["Col.DARKBLUE"],
["Col.BLUE"],
["Col.LIGHTBLUE"],
["Col.MAGENTA"],
["Font.ZERO4B11"],
["Font.APPLE"],
["Font.BOLD"],
["Font.C64"],
["Font.CASUAL"],
["Font.COMIC"],
["Font.CRYPT"],
["Font.DEFAULT"],
["Font.DOS"],
["Font.HANDY"],
["Font.GANON"],
["Font.NOKIA"],
["Font.OLDENGLISH"],
["Font.PIXEL"],
["Font.PRESSSTART"],
["Font.RETROFUTURE"],
["Font.ROMAN"],
["Font.SPECIAL"],
["Font.VISITOR"],
["Font.YOSTER"],
["Text.setfont","(fontname, size)"],
["Text.changesize","(fontsize)"],
["Text.display","(x, y, text, col, optional parameters)"],
["Text.input",'(x, y, "Question: ", Q colour, A colour):Bool'],
["Text.getinput","():String"],
["Music.playsound","(seed)"],
["Music.playnote","(seed,pitch,length (0-1),volume (0-1) )"],
["Key.A"],
["Key.B"],
["Key.C"],
["Key.D"],
["Key.E"],
["Key.F"],
["Key.G"],
["Key.H"],
["Key.I"],
["Key.J"],
["Key.K"],
["Key.L"],
["Key.M"],
["Key.N"],
["Key.O"],
["Key.P"],
["Key.Q"],
["Key.R"],
["Key.S"],
["Key.T"],
["Key.U"],
["Key.V"],
["Key.W"],
["Key.X"],
["Key.Y"],
["Key.Z"],
["Key.ZERO"],
["Key.ONE"],
["Key.TWO"],
["Key.THREE"],
["Key.FOUR"],
["Key.FIVE"],
["Key.SIX"],
["Key.SEVEN"],
["Key.EIGHT"],
["Key.NINE"],
["Key.F1"],
["Key.F2"],
["Key.F3"],
["Key.F4"],
["Key.F5"],
["Key.F6"],
["Key.F7"],
["Key.F8"],
["Key.F9"],
["Key.F10"],
["Key.F11"],
["Key.F12"],
["Key.MINUS"], 
["Key.PLUS"], 
["Key.DELETE"], 
["Key.BACKSPACE"], 
["Key.LBRACKET"],
["Key.RBRACKET"], 
["Key.BACKSLASH"],
["Key.CAPSLOCK"],
["Key.SEMICOLON"],
["Key.QUOTE"],
["Key.COMMA"],
["Key.PERIOD"],
["Key.SLASH"],
["Key.ESCAPE"],
["Key.ENTER"],
["Key.SHIFT"],
["Key.CONTROL"],
["Key.ALT"],
["Key.SPACE"],
["Key.UP"],
["Key.DOWN"],
["Key.LEFT"],
["Key.RIGHT"],
["Input.justpressed","(Key.ENTER)"],
["Input.pressed","(Key.LEFT)"],
["Input.justreleased","(Key.SPACE)"],
["Input.delaypressed","(Key.Z, 5)"],
["Mouse.x",":Int"],
["Mouse.y",":Int"],
["Mouse.leftclick","()"],
["Mouse.leftheld","()"],
["Mouse.leftreleased","()"],
["Mouse.middleclick","()"],
["Mouse.middleheld","()"],
["Mouse.middlereleased","()"],
["Mouse.rightclick","()"],
["Mouse.rightheld","()"],
["Mouse.rightreleased","()"],
["Mouse.mousewheel",":Int"],
["Convert.tostring","(1234):String"],
["Convert.toint",'("15"):Int'],
["Convert.tofloat",'("3.1417826"):Float'],
["Random.randint","(from, to_inclusive):Int"],
["Random.randfloat","(from, to_inclusive):Int"],
["Random.randstring","(length):String"],
["Random.randbool","():Bool"],
["Random.occasional","():Bool"],
["Random.rare","():Bool"],
["Random.pickstring",'("this one", "or this one?", "maybe this one?"):String'],
["Random.pickint","(5, 14, 72, 92, 1, -723, 8):Int"],
["Random.pickfloat","(5.1, 14.2, 72.3, 92.4, 1.5, -723.6, 8.7):Float"],
["Debug.log","(message)"],
["Game.title","(title)"],
["Game.homepage","(url)"],
["Game.background","(color)"],
["Game.foreground","(color)"],
/*,
["break"],
["case"],
["callback"],
["cast"],
["catch"],
["class"],
["continue"],
["default"],
["do"," expr-loop while( expr-cond );"],
["dynamic"],
["else"],
["enum"],
["extends"],
["extern"],
["false"],
["for","( variable in iterable ) expr-loop;"],
["function"],
["if","( expr-cond ) expr-1 [else expr-2]"],
["implements"],
["import"],
["in"],
["inline"],
["interface"],
["never"],
["new"],
["null"],
["override"],
["package"],
["private"],
["public"],
["return"],
["static"],
["super"],
["switch"],
["this"],
["throw"],
["trace"],
["true"],
["try"],
["typedef"],
["untyped"],
["using"],
["var"],
["while","( expr-cond ) expr-loop;"],
["Int"], 
["Float"], 
["String"], 
["Void"], 
["Std"], 
["Bool"], 
["Dynamic"] 
["Array"]*/
];

/*
if you want the hints to be sorted
function compareFn(a,b){
	return a[0].localeCompare(b[0]);
}
haxeHintArray.sort(compareFn);
*/

function CompletionsPick( p_oCompletion ) { 
 //  console.log( "==> Function entry: " + arguments.callee.name + "() <==" ) ; 
   //console.log( p_oCompletion ) ; 
   consolePrint(p_oCompletion.text+p_oCompletion.displayText,true);
} 


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
		for (var i=0;i<haxeHintArray.length;i++){
			var ar = haxeHintArray[i];
			var w = ar[0];
			if (w.length<token.length){
				continue;
			} 
			if (w.substring(0,token.length)==token){
				var w2 = ar.length>1?ar[1]:"";
				matches.push({text:w,displayText:w2,render:renderHint});
			}
		}
		var result={list: matches, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
  		CodeMirror.on( result, "pick",   CompletionsPick ) ; 
		return result;
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

