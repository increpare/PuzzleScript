function runClick() {
	compile(["restart"]);
}

function rebuildClick() {
	compile(["rebuild"]);
}

function post_to_url(path, params, method) {
    method = method || "post"; // Set method to post by default if not specified.

    // The rest of this code assumes you are not using a library.
    // It can be made less wordy if you use one.
    var form = document.createElement("form");
    form.setAttribute("method", method);
    form.setAttribute("action", path);

    for(var key in params) {
        if(params.hasOwnProperty(key)) {
            var hiddenField = document.createElement("input");
            hiddenField.setAttribute("type", "hidden");
            hiddenField.setAttribute("name", key);
            hiddenField.setAttribute("value", params[key]);

            form.appendChild(hiddenField);
         }
    }

    document.body.appendChild(form);
    form.submit();
}

function exportClick() {
	var sourceCode = editor.getValue();

	compile("restart");

	var stateString = JSON.stringify(state);
	
	buildStandalone(stateString);
}

function compile(command,text) {
	forceRegenImages=true;
	if (command===undefined) {
		command = ["restart"];
	}
	lastDownTarget=canvas;	

	if (text===undefined){
		var code = window.form1.code;

		var editor = code.editorreference;

		text = editor.getValue();
	}
	if (canDump===true) {
		compiledText=text;
	}

	errorCount = 0;
	compiling = true;
	errorStrings = [];
	consolePrint('=================================');
	try
	{
		var state = loadFile(text);
//		consolePrint(JSON.stringify(state));
	} finally {
		compiling = false;
	}
	if (errorCount>MAX_ERRORS) {
		return;
	}
	/*catch(err)
	{
		if (anyErrors===false) {
			logErrorNoLine(err.toString());
		}
	}*/

	if (errorCount>0) {
		consolePrint('<span class="systemMessage">Errors detected during compilation, the game will not work correctly.</span>');
	}
	else {
		var ruleCount=0;
		for (var i=0;i<state.rules.length;i++) {
			ruleCount+=state.rules[i].length;
		}
		for (var i=0;i<state.lateRules.length;i++) {
			ruleCount+=state.lateRules[i].length;
		}
		if (command[0]=="restart") {
			consolePrint('<span class="systemMessage">Successful Compilation, generated ' + ruleCount + ' instructions.</span>');
		} else {
			consolePrint('<span class="systemMessage">Successful live recompilation, generated ' + ruleCount + ' instructions.</span>');

		}
	}
	setGameState(state,command);

	if (canDump===true) {
		inputHistory=[];
	}
}


