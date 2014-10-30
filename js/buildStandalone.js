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
			consolePrint("Couldn't find standalone template. Is there a connection problem to the internet?",true);
		}
		standalone_HTML_String=clientStandaloneRequest.responseText;
}
clientStandaloneRequest.send();

function buildStandalone(sourceCode) {
	if (standalone_HTML_String.length===0) {
		consolePrint("Can't export yet - still downloading html template.",true);
		return;
	}

	var htmlString = standalone_HTML_String.concat("");
	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
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

	htmlString = htmlString.replace(/__GAMETITLE__/g,title);
	htmlString = htmlString.replace(/__HOMEPAGE__/g,homepage);

	// $ has special meaning to JavaScript's String.replace ($0, $1, etc.) Escape $ as $$.
	sourceCode = sourceCode.replace(/\$/g, '$$$$');

	htmlString = htmlString.replace(/__GAMEDAT__/g,sourceCode);

	var BB = get_blob();
	var blob = new BB([htmlString], {type: "text/plain;charset=utf-8"});
	saveAs(blob, title+".html");
}
