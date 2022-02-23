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

function buildStandalone(sourceCode) {
	if (standalone_HTML_String.length===0) {
		consolePrint("Can't export yet - still downloading html template.",true,null,null);
		return;
	}

	var htmlString = standalone_HTML_String.concat("");
	var title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
	}
	var homepage = "https://www.puzzlescript.net";
	if (state.metadata.homepage!==undefined) {
		homepage=state.metadata.homepage;
		if (!homepage.match(/^https?:\/\//)) {
			homepage = "https://" + homepage;
		}
	}
	var homepage_stripped = homepage.replace(/^https?:\/\//,'');

	var background_color="black";
	if ('background_color' in state.metadata) {
		background_color=state.bgcolor;		
	}
	htmlString = htmlString.replace(/___BGCOLOR___/g,background_color);	

	var text_color="lightblue";
	if ('text_color' in state.metadata) {
		text_color = state.fgcolor;	
	}
	htmlString = htmlString.replace(/___TEXTCOLOR___/g,text_color);	

	htmlString = htmlString.replace(/__GAMETITLE__/g,title);


	htmlString = htmlString.replace(/__HOMEPAGE__/g,homepage);
	htmlString = htmlString.replace(/__HOMEPAGE_STRIPPED_PROTOCOL__/g,homepage_stripped);

	// $ has special meaning to JavaScript's String.replace ($0, $1, etc.) 
	// '$$'s are inserted as single '$'s.

	// First we double all strings - remember that replace interprets '$$' 
	// as a single'$', so we need to type four to double
	sourceCode = sourceCode.replace(/\$/g, '$$$$');

	// Then when we substitute them, the doubled $'s will be reduced to single ones.
	htmlString = htmlString.replace(/"__GAMEDAT__"/g,sourceCode);

	var BB = get_blob();
	var blob = new BB([htmlString], {type: "text/plain;charset=utf-8"});
	saveAs(blob, title+".html");
}
