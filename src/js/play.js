'use strict';

function displayError(message) {
	var errorText = document.getElementById("errormessage");
	errorText.textContent="ERROR "+message;
	errorText.appendChild(document.createElement("br"));

}
function getParameterByName(name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    if (results == null) {
        return "";
    }
    try {
        return decodeURIComponent(results[1].replace(/\+/g, " "));
    } catch (error) {
        return "";
    }
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
	var isPreservedPushingIt = id === "2fe3172d2b9fe684977d184f1b6226d5";
	function gameLoaded(code, e) {
		if (e!==null) {
			displayError(e);
			return;
		}
		compile(["restart"],code);

		if (state.metadata.homepage!==undefined) {
			var homepage=state.metadata.homepage;
			var homepageLink = document.getElementById("homepagelink");
			homepageLink.textContent=strip_http(homepage);

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

		if (isPreservedPushingIt) {
			var runtime = document.createElement("script");
			runtime.src = "legacy/pushing-it/runtime.js";
			runtime.onerror = function() {
				displayError("Unable to load preserved Pushing It runtime.");
			};
			document.body.appendChild(runtime);
		}
	}

	if (isPreservedPushingIt) {
		var request = new XMLHttpRequest();
		request.open("GET", "legacy/pushing-it/script.txt");
		request.onload = function() {
			if (request.status === 200) {
				gameLoaded(request.responseText, null);
			} else {
				gameLoaded(null, "Unable to load preserved Pushing It source.");
			}
		};
		request.onerror = function() {
			gameLoaded(null, "Unable to load preserved Pushing It source.");
		};
		request.send();
	} else {
		github_load(id, gameLoaded);
	}
}

getData();
