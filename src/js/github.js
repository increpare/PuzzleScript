github = {};

// The client ID of a GitHub OAuth app registered at https://github.com/settings/developers.
// The “callback URL” of that app points to https://www.puzzlescript.net/auth.html.
// If you’re running from another host name, sharing might not work.
OAUTH_CLIENT_ID = "211570277eb588cddf44";

github.authURL = function() {
	// Generates 32 letters of random data, like "liVsr/e+luK9tC02fUob75zEKaL4VpQn".
	var randomState = window.btoa(Array.prototype.map.call(
		window.crypto.getRandomValues(new Uint8Array(24)),
		function(x) { return String.fromCharCode(x); }).join(""));

	return "https://github.com/login/oauth/authorize"
		+ "?client_id=" + OAUTH_CLIENT_ID
		+ "&scope=gist"
		+ "&state=" + randomState
		+ "&allow_signup=true";
};

github.signOut = function() {
	storage_remove("oauth_access_token");
};

github.isSignedIn = function() {
	var token = storage_get("oauth_access_token");
	return typeof token === "string";
};

github.load = function(id, done) { 
	var githubURL = "https://api.github.com/gists/"+id;

	var githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open("GET", githubURL);
	githubHTTPClient.onreadystatechange = function() {
		if (githubHTTPClient.readyState!=4) {
			return;
		} else if (githubHTTPClient.responseText==="") {
			done(null, "GitHub request returned nothing.  A connection fault, maybe?");
			return;
		}

		var result = JSON.parse(githubHTTPClient.responseText);
		if (githubHTTPClient.status===403) {
			done(null, result.message);
			return;
		} else if (githubHTTPClient.status!==200 && githubHTTPClient.status!==201) {
			done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText);
			return;
		}
		var result = JSON.parse(githubHTTPClient.responseText);
		var code=result["files"]["script.txt"]["content"];
		done(code, null);
	}

	// if (storage_has("oauth_access_token")) {
	//     var oauthAccessToken = storage_get("oauth_access_token");
	//     if (typeof oauthAccessToken === "string") {
	//         githubHTTPClient.setRequestHeader("Authorization","token "+oauthAccessToken);
	//     }
	// }
	githubHTTPClient.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	githubHTTPClient.send();
};

github.save = function(title, code, done) {
	var oauthAccessToken = storage_get("oauth_access_token");
	if (typeof oauthAccessToken !== "string") {
		printUnauthorized();
		return;
	}

	var gistToCreate = {
		"description" : title,
		"public" : true,
		"files": {
			"readme.txt" : {
				"content": "Play this game by pasting the script in http://www.puzzlescript.net/editor.html"
			},
			"script.txt" : {
				"content": code
			}
		}
	};

	var githubURL = "https://api.github.com/gists";
	var githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open("POST", githubURL);
	githubHTTPClient.onreadystatechange = function() {		
		if(githubHTTPClient.readyState!=4) {
			return;
		}		
		var result = JSON.parse(githubHTTPClient.responseText);
		if (githubHTTPClient.status===403) {
			done(null, result.message);
		} else if (githubHTTPClient.status!==200&&githubHTTPClient.status!==201) {
			github.signOut();
			if (githubHTTPClient.statusText==="Unauthorized"){
				done(null, "Authorization check failed.  You have to log back into GitHub (or give it permission again or something).");
			} else {
				done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText + ".  Try giving puzzlescript permission again, that might fix things...");
			}
		} else {
			done(result.id, null);
		}
	}
	githubHTTPClient.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	githubHTTPClient.setRequestHeader("Authorization", "Token "+oauthAccessToken);
	var stringifiedGist = JSON.stringify(gistToCreate);
	githubHTTPClient.send(stringifiedGist);
};
