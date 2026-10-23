'use strict';

function displayError(message) {
	var errorText = document.getElementById("errormessage");
	errorText.innerHTML="ERROR "+message+"<br>";

}
function getParameterByName(name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    return results == null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

// be careful about modifying this (cf  // https://github.com/increpare/PuzzleScript/issues/829 )
function strip_http(url) {
   url = url.replace(/^https?:\/\//,'');
   return url;
}

function getData(){
	var id = getParameterByName("p").replace(/[\\\/]/,"");
	if (id===null||id.length===0) {
		displayError("No ID specified in URL.")
		return;
	}
	github_load(id, function(code, e) {
		if (e!==null) {
			displayError(e);
			return;
		}
		compile(["restart"],code);

		if (state.metadata.homepage!==undefined) {
			var homepage=state.metadata.homepage;
			var homepageLink = document.getElementById("homepagelink");
            if (id==="2fe3172d2b9fe684977d184f1b6226d5"){
                // https://github.com/increpare/PuzzleScript/issues/829
                // Supporting vulnerability on legacy grounds to enable Pushing It:
                // https://www.puzzlescript.net/play.html?p=2fe3172d2b9fe684977d184f1b6226d5
			    homepageLink.innerHTML=strip_http(homepage);
            } else {
                //sanitize tags from title
                homepageLink.textContent=strip_http(homepage);
            }

 			if (!homepage.match(/^https?:\/\//)) {
 				homepage = "https://" + homepage;
 			}
 			homepageLink.href = homepage;
		}
		if (state.metadata.title!==undefined) {
			var title=state.metadata.title;
			var gametitle = document.getElementById("gametitle");
            gametitle.textContent=title;
			window.document.title=title+" - PuzzleScript Game";
		}

		var hacklink = document.getElementById("hacklink");

		var url = "editor.html?hack="+id;
		url=qualifyURL(url);

		hacklink.href=url;
	});
}

getData();
