'use strict';

let standalone_HTML_String="";

let clientStandaloneRequest = new XMLHttpRequest();

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

function escapeHtmlChars(unsafe)
{
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }
 
function buildStandalone(sourceCode) {
	if (standalone_HTML_String.length===0) {
		consolePrint("Can't export yet - still downloading html template.",true,null,null);
		return;
	}

	let htmlString = standalone_HTML_String.concat("");
	let title = "PuzzleScript Game";
	if (state.metadata.title!==undefined) {
		title=state.metadata.title;
	}
	let homepage = "https://www.puzzlescript.net";
	if (state.metadata.homepage!==undefined) {
		homepage=state.metadata.homepage;
		if (!homepage.match(/^https?:\/\//)) {
			homepage = "https://" + homepage;
		}
	}
	let homepage_stripped = homepage.replace(/^https?:\/\//,'');
	homepage_stripped = escapeHtmlChars(homepage_stripped);

	let background_color="black";
	if ('background_color' in state.metadata) {
		background_color=state.bgcolor;		
	}
	htmlString = htmlString.replace(/___BGCOLOR___/g,background_color);	

	let text_color="lightblue";
	if ('text_color' in state.metadata) {
		text_color = state.fgcolor;	
	}
	htmlString = htmlString.replace(/___TEXTCOLOR___/g,text_color);	

	htmlString = htmlString.replace(/__GAMETITLE__/g,escapeHtmlChars(title));


	htmlString = htmlString.replace(/__HOMEPAGE__/g,homepage);
	htmlString = htmlString.replace(/__HOMEPAGE_STRIPPED_PROTOCOL__/g,homepage_stripped);

	let hackLink = 'nohack' in state.metadata ? '' :
		'<span class="player-link-separator" aria-hidden="true">|</span><form method="post" action="https://www.puzzlescript.net/editor.html" accept-charset="UTF-8" target="_blank" rel="noopener" onsubmit="this.elements.source.value=sourceCode">' +
		'<input type="hidden" name="source"><button type="submit">hack</button></form>';
	htmlString = htmlString.replace(/__HACK_LINK__/g, () => hackLink);

	// sourceCode is already JSON. Keep it inside its script even when game text
	// contains HTML end tags. A callback also preserves literal $ replacements.
	sourceCode = sourceCode.replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
	htmlString = htmlString.replace(/"__GAMEDAT__"/g, () => sourceCode);

	saveAs(htmlString, 'text/html;charset=utf-8',title+".html");
}
