var code = document.getElementById('code');

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
	if (localStorage!==undefined && localStorage['saves']!==undefined) {
			var curSaveArray = JSON.parse(localStorage['saves']);
			var sd = curSaveArray[curSaveArray.length-1];
			code.value = sd.text;
			var loadDropdown = document.getElementById('loadDropDown');
			loadDropdown.selectedIndex=0;
	    }
	}
}


var editor = window.CodeMirror.fromTextArea(code, {
//	viewportMargin: Infinity,
	lineWrapping: true,
	lineNumbers: true,
      styleActiveLine: true
	});

editor.on('mousedown', function(cm, event) {
  if (event.target.className == 'cm-SOUND') {
    var seed = parseInt(event.target.innerHTML);
    playSound(seed);
  } else if (event.target.className == 'cm-LEVEL') {
    if (event.ctrlKey||event.metaKey) {
      compile(["levelline",cm.posFromMouse(event).line]);      
    }
  }
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
editor.setOption('theme', 'midnight');

function getParameterByName(name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    return results == null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

function tryLoadGist(id) {
	var githubURL = 'https://api.github.com/gists/'+id;

	consolePrint("Contacting GitHub");
	var githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open('GET', githubURL);
	githubHTTPClient.onreadystatechange = function() {
	
		if(githubHTTPClient.readyState!=4) {
			return;
		}

		if (clientStandaloneRequest.responseText==="") {
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
	}
	fileOpenClient.send();
}

function dropdownChange() {
	tryLoadFile(this.value);
	this.selectedIndex=0;
}
