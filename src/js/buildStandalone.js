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

	// $ has special meaning to JavaScript's String.replace 
	// c.f.	https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/replace#specifying_a_string_as_a_parameter
	// basically: '$$'s are inserted as single '$'s.

	// First we double all strings - remember that replace interprets '$$' 
	// as a single'$', so we need to type four to double
	sourceCode = sourceCode.replace(/\$/g, '$$$$');

	// Then when we substitute them, the doubled $'s will be reduced to single ones.
	htmlString = htmlString.replace(/"__GAMEDAT__"/g,sourceCode);

	saveAs(htmlString, 'text/html;charset=utf-8',title+".html");
}
