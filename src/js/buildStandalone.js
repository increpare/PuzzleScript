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
			consolePrint("Couldn't find standalone template. Is there a connection problem to the internet?",true,null,null);
		}
		standalone_HTML_String=clientStandaloneRequest.responseText;
}
clientStandaloneRequest.send();

// replace bad characters in user-supplied text
function safeUser(value) {
	const regex = /[<>&"'\v]/g;
	var nv = value;
	if (nv.match(regex)) {
		consolePrint(`Unsafe characters found in script will be replaced by !`);
		nv = nv.replace(regex, '!');
	}
	// remove $ for now, they go back later
	return safeDollar(nv);
}

// replace $ because it's special in replace
function safeDollar(value) {
	return value.replace(/[$]/g, '\v');
}

// mainline function to build and save standalone version of script
function buildStandalone(sourceCode) {
	if (standalone_HTML_String.length===0) {
		consolePrint("Can't export yet - still downloading html template.",true,null,null);
		return;
	}

	var htmlString = standalone_HTML_String.concat("");
	var title = state.metadata.title ? state.metadata.title : "PuzzleScript Game";

	var homepage = state.metadata.homepage ? state.metadata.homepage : "https://www.puzzlescript.net";
	if (!homepage.match(/^https?:\/\//)) {
		homepage = "https://" + homepage;
	}
	var homepage_stripped = homepage.replace(/^https?:\/\//,'');

	var background_color = ('background_color' in state.metadata) ? state.bgcolor : "black";
	htmlString = htmlString.replace(/___BGCOLOR___/g, background_color);	

	var text_color = ('text_color' in state.metadata) ? state.fgcolor : "lightblue";
	htmlString = htmlString.replace(/___TEXTCOLOR___/g, text_color);	

	htmlString = htmlString.replace(/__GAMETITLE__/g, safeUser(title));
	htmlString = htmlString.replace(/__HOMEPAGE__/g, safeUser(homepage));
	htmlString = htmlString.replace(/__HOMEPAGE_STRIPPED_PROTOCOL__/g, safeUser(homepage_stripped));

	// $ has special meaning, so replace it by \v, then switch it back
	htmlString = htmlString.replace(/"__GAMEDAT__"/, safeDollar(sourceCode));
	htmlString = htmlString.replace(/\v/g, '$$');

	// remove bad Windows chars
	var fn = title.replace(/[<>:|*?]/g, '!').trim() + ".html";
	var BB = get_blob();
	var blob = new BB([htmlString], {type: "text/plain;charset=utf-8"});
	saveAs(blob, fn);
}
