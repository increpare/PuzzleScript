var get_blob = function() {
		return self.Blob;
}

var standalone_HTML_String="";

var clientStandaloneRequest = new XMLHttpRequest();

clientStandaloneRequest.open('GET', 'standalone_inlined.txt');
clientStandaloneRequest.onreadystatechange = function() {

		if(clientStandaloneRequest.readyState!=4) {
			return;
		}
		if (clientStandaloneRequest.responseText==="") {
			consolePrint("Couldn't find standalone template. Is there a connection problem to the internet?");
		}
		standalone_HTML_String=clientStandaloneRequest.responseText;
}
clientStandaloneRequest.send();

function buildStandalone(stateString) {
	if (standalone_HTML_String.length===0) {
		consolePrint("Can't export yet - still downloading html template.");
		return;
	}

	var htmlString = standalone_HTML_String.concat("");
	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title.toUpperCase();
	}
	var homepage = "www.puzzlescript.net";
	if (state.metadata.homepage!==undefined) {
		homepage=state.metadata.homepage.toLowerCase();
	}

	var bgcolor = "black";
	if ('background_color' in state.metadata) {
		bgcolor = state.bgcolor;
	}
	var fgcolor = "lightblue";
	if ('text_color' in state.metadata) {
		fgcolor = state.fgcolor;
	}

	htmlString = htmlString.replace(/\"__GAMETITLE__\"/g,title);
	htmlString = htmlString.replace(/\"__HOMEPAGE__\"/g,homepage);	
	htmlString = htmlString.replace(/\"__GAMEDAT__\"/g,stateString);
	htmlString = htmlString.replace(/\"__BGCOLOR__\"/g,bgcolor);
	htmlString = htmlString.replace(/\"__FGCOLOR__\"/g,fgcolor);

	var BB = get_blob();
	var blob = new BB([htmlString], {type: "text/plain;charset=utf-8"});
	saveAs(blob, title+".html");
}