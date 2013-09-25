function runClick() {
	compile(["restart"]);
}

function rebuildClick() {
	compile(["rebuild"]);
}


function compile(command,text) {
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
		if (anyErrors==false) {
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


