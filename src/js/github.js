// The client ID of a GitHub OAuth app registered at https://github.com/settings/developers.
// The “callback URL” of that app points to https://www.puzzlescript.net/auth.html.
// If you’re running from another host name, sharing might not work.
OAUTH_CLIENT_ID = "211570277eb588cddf44";

function github_authURL() {
	// Generates 32 letters of random data, like "liVsr/e+luK9tC02fUob75zEKaL4VpQn".
	let randomState = window.btoa(Array.prototype.map.call(
		window.crypto.getRandomValues(new Uint8Array(24)),
		function(x) { return String.fromCharCode(x); }).join(""));

	return "https://github.com/login/oauth/authorize"
		+ "?client_id=" + OAUTH_CLIENT_ID
		+ "&scope=gist"
		+ "&state=" + randomState
		+ "&allow_signup=true";
}

function github_signOut(){
	storage_remove("oauth_access_token");
}

function github_isSignedIn() {
	let token = storage_get("oauth_access_token");
	return typeof token === "string";
}

function github_load(id, done) { 
	let githubURL = "https://api.github.com/gists/"+id;

	let githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open("GET", githubURL);
	githubHTTPClient.onreadystatechange = function() {
		if (githubHTTPClient.readyState!=4) {
			return;
		} else if (githubHTTPClient.responseText==="") {
			done(null, "GitHub request returned nothing.  A connection fault, maybe?");
			return;
		}

		let limit = window.parseInt(githubHTTPClient.getResponseHeader("x-ratelimit-limit"));
		let used = window.parseInt(githubHTTPClient.getResponseHeader("x-ratelimit-used"));
		let reset = new Date(1000 * window.parseInt(githubHTTPClient.getResponseHeader("x-ratelimit-reset")));
		console.log("Rate limit used " + used + "/" + limit + " (resets " + reset.toISOString() + ")");

		let result = JSON.parse(githubHTTPClient.responseText);
		if (githubHTTPClient.status===403) {
			if (!github_isSignedIn() && (result.message.indexOf("rate limit") !== -1)) {
				done(null, "Exceeded GitHub rate limits. Try signing in from the editor.");
			} else {
				done(null, result.message);
			}
		} else if (githubHTTPClient.status===401) {
			github_signOut();
			done(null, "Authorization check failed.  Try reloading or signing back in from the editor.");
		} else if (githubHTTPClient.status>=500) {
			done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText + ".");
		} else if (githubHTTPClient.status!==200 && githubHTTPClient.status!==201) {
			done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText);
		} else {
			let result = JSON.parse(githubHTTPClient.responseText);
			let code=result["files"]["script.txt"]["content"];
			done(code, null);
		}
	}

	if (github_isSignedIn()) {
		let oauthAccessToken = storage_get("oauth_access_token");
		githubHTTPClient.setRequestHeader("Authorization", "Token "+oauthAccessToken);
	}
	githubHTTPClient.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	githubHTTPClient.send();
}

function github_save(title, code, done) {
	let oauthAccessToken = storage_get("oauth_access_token");
	if (typeof oauthAccessToken !== "string") {
		printUnauthorized();
		return;
	}

	let gistToCreate = {
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

	let githubURL = "https://api.github.com/gists";
	let githubHTTPClient = new XMLHttpRequest();
	githubHTTPClient.open("POST", githubURL);
	githubHTTPClient.onreadystatechange = function() {		
		if(githubHTTPClient.readyState!=4) {
			return;
		}		
		let result = JSON.parse(githubHTTPClient.responseText);
		if (githubHTTPClient.status===403) {
			done(null, result.message);
		} else if (githubHTTPClient.status===401) {
			github_signOut();
			done(null, "Authorization check failed.  You have to log back into GitHub (or give it permission again or something).");
		} else if (githubHTTPClient.status>=500) {
			done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText + ".");
		} else if (githubHTTPClient.status!==200 && githubHTTPClient.status!==201) {
			github_signOut();
			done(null, "HTTP Error "+ githubHTTPClient.status + " - " + githubHTTPClient.statusText + ".  Try giving puzzlescript permission again, that might fix things...");
		} else {
			done(result.id, null);
		}
	}
	githubHTTPClient.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	githubHTTPClient.setRequestHeader("Authorization", "Token "+oauthAccessToken);
	let stringifiedGist = JSON.stringify(gistToCreate);
	githubHTTPClient.send(stringifiedGist);
}