
var code = document.getElementById('code');

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

var fileToOpen=getParameterByName("demo");
if (fileToOpen!==null&&fileToOpen.length>0) {
	var client = new XMLHttpRequest();
	client.open('GET', 'demo/'+fileToOpen+".txt");
	client.onreadystatechange = function() {
		
  		if(client.readyState!=4) {
  			return;
  		}
  		
		editor.setValue(client.responseText);
	}
	client.send();
}