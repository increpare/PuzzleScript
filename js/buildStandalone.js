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

var debug_HTML_String="";

var clientDebugRequest = new XMLHttpRequest();

clientDebugRequest.open('GET', 'debug_inlined.txt');
clientDebugRequest.onreadystatechange = function() {

		if(clientDebugRequest.readyState!=4) {
			return;
		}
		if (clientDebugRequest.responseText==="") {
			consolePrint("Couldn't find debug template. Is there a connection problem to the internet?");
		}
		debug_HTML_String=clientDebugRequest.responseText;
}
clientDebugRequest.send();

function buildFromHTML(stateString, str) {
	if (str.length===0) {
		consolePrint("Can't export yet - still downloading html template.");
		return;
	}

	var htmlString = str.concat("");
	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title.toUpperCase();
	}
	var homepage = "www.puzzlescript.net";
	if (state.metadata.homepage!==undefined) {
		homepage=state.metadata.homepage.toLowerCase();
	}

	if ('background_color' in state.metadata) {
		htmlString = htmlString.replace(/black;\/\*Don\'t/g,state.bgcolor+';\/\*Don\'t');	
	}
	if ('text_color' in state.metadata) {
		htmlString = htmlString.replace(/lightblue;\/\*Don\'t/g,state.fgcolor+';\/\*Don\'t');	
	}

	htmlString = htmlString.replace(/\"__GAMETITLE__\"/g,title);
	htmlString = htmlString.replace(/\"__HOMEPAGE__\"/g,homepage);	
	htmlString = htmlString.replace(/\"__GAMEDAT__\"/g,stateString);

	var BB = get_blob();
	var blob = new BB([htmlString], {type: "text/plain;charset=utf-8"});
	saveAs(blob, title+".html");
}

function buildStandalone(stateString) {
    buildFromHTML(stateString, standalone_HTML_String);
}

function buildDebug(stateString) {
    buildFromHTML(stateString, debug_HTML_String);
}