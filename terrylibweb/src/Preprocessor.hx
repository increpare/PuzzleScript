class Preprocessor {
	//Public functions
	public static function getlinenum(originalline:Int):Int {
		if (originalline >= 0 && originalline < linenumbers.length) {
			//trace(script[originalline]);
			return linenumbers[originalline];
		}
		return -1;
	}
	
	public static function loadscript(myscript:String) {
		var j:Int = 0;
		var k:Int = 0;
		
		originalscript = myscript;
		script = myscript.split("\n");
		linenumbers = [];
		for (i in 0 ... script.length) {
			script[i] = S.replacechar(script[i], "\r", "");
			script[i] = S.replacechar(script[i], "\t", "  ");
			linenumbers[i] = i + 1;
		}
		
		//Do some preperation on the script to make it easier to handle
		for (i in 0 ... script.length) {
			//Remove spaces
			script[i] = S.trimspaces(script[i]);
			
			//Remove comments
			script[i] = removecomments(script[i]);
			
			//Remove spacing around symbols
			script[i] = removesymbolspacing(script[i]);
		}
		
		//Split multiple statements on single lines into multiple lines
		j = 0;
		while (j < script.length) {
			if (S.isinstring(S.left(script[j], script[j].length - 1), ";")) {
				//Ok, do a detailed check
				var linesplit:Array<String> = splitbysymbol(script[j], ";", 1);
				
				if (linesplit.length > 1) {
					k = linenumbers[j];
					script.splice(j, 1);
					linenumbers.splice(j, 1);
				  for (i in 0 ... linesplit.length) {
						script.insert(j, linesplit[i]);
						linenumbers.insert(j, k);
					}
				}
			}
			j++;
		}
		
		//Split { and } symbols into unique lines
		j = 0;
		while (j < script.length) {
			if (S.isinstring(excludestrings(script[j]), "{")) {
				if (script[j] != "{") {
					var linesplit:Array<String> = splitbysymbol(script[j], "{", 2);
					
					if (linesplit.length > 1) {
						k = linenumbers[j];
						script.splice(j, 1);
						linenumbers.splice(j, 1);
						for (i in 0 ... linesplit.length) {
							script.insert(j, linesplit[i]);
							linenumbers.insert(j, k);
						}
					}
				}
			}
			
			if (S.isinstring(excludestrings(script[j]), "}")) {
				if (script[j] != "}") {
					var linesplit:Array<String> = splitbysymbol(script[j], "}", 2);
					
					if (linesplit.length > 1) {
						k = linenumbers[j];
						script.splice(j, 1);
						linenumbers.splice(j, 1);
						for (i in 0 ... linesplit.length) {
							script.insert(j, linesplit[i]);
							linenumbers.insert(j, k);
						}
					}
			  }
			}
			j++;
		}
		
		//Remove blank lines
		j = 0;
		while (j < script.length) {
			if (script[j] == "") {
				script.splice(j, 1);
				linenumbers.splice(j, 1);
			}else {
				j++;
			}
		}
	}
	
	public static function sortbyscope():Bool {
		var scriptpart1:Array<String> = [];
		var scriptpart2:Array<String> = [];
		var linenumpart1:Array<Int> = [];
		var linenumpart2:Array<Int> = [];
		var scriptnewfunction:Array<String> = [];
		var scriptupdatefunction:Array<String> = [];
		var linenumnewfunction:Array<Int> = [];
		var linenumupdatefunction:Array<Int> = [];
		var linescope:Array<Int> = [];
		var scopecount:Int = 0;
		var functionmode:String = "normal";
		
		var i:Int = 0;
		while (i < script.length) {
			if (S.isinstring(excludestrings(script[i]), "{")) {
				scopecount++;
				
				if (i > 0) {
					linescope[i - 1]++;
				}
			}
			
			linescope.push(scopecount);
			
			if (S.isinstring(excludestrings(script[i]), "}")) {
				scopecount--;
			}
			i++;
		}
		
		if (scopecount != 0) {
			return false;
		}
		
		i = 0;
		while (i < script.length) {
			scopecount = linescope[i];
			if (S.isinstring(excludestrings(script[i]), "function new()")) {
				functionmode = "new";
			}
			if (S.isinstring(excludestrings(script[i]), "function update()")) {
				functionmode = "update";
			}
			
			if (scopecount == 0) {
				scriptpart1.push(script[i]);
				linenumpart1.push(linenumbers[i]);
			}else {
				if (functionmode == "new") {
					scriptnewfunction.push(script[i]);
					linenumnewfunction.push(linenumbers[i]);
				}else if (functionmode == "update") {
					scriptupdatefunction.push(script[i]);
					linenumupdatefunction.push(linenumbers[i]);
				}else {
					scriptpart2.push(script[i]);
					linenumpart2.push(linenumbers[i]);
				}
			}
			
			if (functionmode == "new" || functionmode == "update") {
				if (scopecount == 1) {
				  if (script[i] == "}") {
						functionmode = "normal";
					}
				}
			}
			
			i++;
		}
		
		script = scriptpart1.concat(scriptpart2.concat(scriptnewfunction.concat(scriptupdatefunction)));
		linenumbers = linenumpart1.concat(linenumpart2.concat(linenumnewfunction.concat(linenumupdatefunction)));
		
		return true;
	}
	
	public static function excludestrings(t:String):String {
		if (S.isinstring(t, "\"")) {
			var newstring:String = "";
			commacount = 0;
			for (i in 0 ... t.length) {
				currentchar = t.substr(i, 1);
				if (currentchar == "\"") {
					commacount = (commacount + 1) % 2;
				}
				if(commacount == 0){
					newstring = newstring + currentchar;
				}
			}
			return newstring;
		}
		return t;
	}
	
	public static function checkforerrors() {
		
	}
	
	public static function getfinalscript():String {
		return script.join("\n");
	}
	
	public static function debug() {
		for (i in 0 ... script.length) {
			trace("LINE " + linenumbers[i] + ": " + script[i] + "");
		}
	}
	
	//Internal stuff:
	private static var originalscript:String;
	private static var script:Array<String>;
	private static var linenumbers:Array<Int>;
	private	static var currentchar:String = "";
	private	static var currentline:String = "";
	private	static var commacount:Int = 0;
	
	private static function splitbysymbol(fullstring:String, s:String, splitmode:Int = 0):Array<String> {
		//Split modes:
		//0 - Strip out symbol
		//1 - Append symbol to end of line
		//2 - Put symbol on it's own line
		var linesplit:Array<String> = [];
		
		commacount = 0;
		currentline = "";
		for (i in 0 ... fullstring.length) {
			currentchar = S.mid(fullstring, i, 1);
			if (currentchar == "\"") commacount = (commacount + 1) % 2;
			if (currentchar == s && commacount == 0) {
				if (splitmode == 2) {
					linesplit.push(currentline);
					linesplit.push(s);
				}else if (splitmode == 1) {
					linesplit.push(currentline + s);
				}else {
					linesplit.push(currentline);
				}
				currentline = "";
			}else{
				currentline = currentline + currentchar;
			}
		}
		
		if (currentline != "") {
			linesplit.push(currentline);
		}
		linesplit.reverse();
		
		return linesplit;
	}
	
	private static function removecomments(t:String):String {
		if (S.isinstring(t, "//")) {
			if (S.positioninstring(t, "//") == 0) {
				t = "";
			}else {
				//ok, check to see that the comment isn't in a string
				var numcommas:Int = 0;
				for (i in 0 ... t.length) {
					if (S.mid(t, i, 1) == "\"") numcommas = (numcommas + 1) % 2;
					if (S.mid(t, i, 1) == "/") {
						if(i + 1 < t.length){
							if (S.mid(t, i + 1, 1) == "/") {
								if (numcommas == 0) {
									t = S.left(t, S.positioninstring(t, "//"));
								}
							}
						}
					}
				}
			}
		}
		
		return t;
	}
	
	private static function removesymbolspacing(t:String):String {
		t = S.trimspaces(t);
			
		var commacheck:Int = 0;
		var checkspaces:Bool = true;
		while (checkspaces) {
			checkspaces = false;
			for (k in 0 ... t.length) {
				if (S.mid(t, k) == "\"") {
					commacheck = (commacheck + 1) % 2;
				}
				if (S.mid(t, k) == " " && commacheck == 0) {
					var char1:String = S.mid(t, k - 1);
					var char2:String = S.mid(t, k + 1);
					if (issymbol(char1) || issymbol(char2)) {
						t = S.left(t, k) + S.right(t, t.length - (k + 1));
						checkspaces = true;
						break;
					}
				}
			}
		}
		
		return t;
	}
	
	private static function issymbol(t:String):Bool {
		if (t == "=") return true;
		if (t == ",") return true;
		if (t == "/") return true;
		if (t == "*") return true;
		if (t == "+") return true;
		if (t == "-") return true;
		if (t == "<") return true;
		if (t == ">") return true;
		if (t == "%") return true;
		if (t == "(") return true;
		if (t == ")") return true;
		if (t == "[") return true;
		if (t == "]") return true;
		if (t == "{") return true;
		if (t == "}") return true;
		if (t == "!") return true;
		if (t == "~") return true;
		if (t == "#") return true;
		if (t == ";") return true;
		return false;
	}
}