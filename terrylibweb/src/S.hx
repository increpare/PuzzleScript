class S {
	/** Returns an uppercase version of the string. */
	public static function uppercase(currentstring:String):String {
		return currentstring.toUpperCase();
	}
	
	/** Returns an lowercase version of the string. */
	public static function lowercase(currentstring:String):String {
		return currentstring.toLowerCase();
	}
	
	/** Splits a string into an array, divided by a given delimiter character (e.g. ",")*/
	public static function split(currentstring:String, delimiter:String):Array<String> {
		return currentstring.split(delimiter);
	}
	
	/** Removes substring from the fullstring. */
	public static function removefromstring(fullstring:String, substring:String):String {
		var t:Int = positioninstring(fullstring, substring);
		if (t == -1) {
			return fullstring;
		}else {
			return removefromstring(getroot(fullstring, substring) + getbranch(fullstring, substring), substring);
		}
	}
	
	public static function isalpha(t:String):Bool {
		t = t.toLowerCase();
		if (t.charCodeAt(0) >= "a".charCodeAt(0)) {
			if (t.charCodeAt(0) <= "z".charCodeAt(0)) {
				return true;
			}
		}
		if (t == "_") return true;
		return false;
	}
	
	/** Returns true if the given stringtocheck is in the given fullstring. */
	public static function isinstring(fullstring:String, stringtocheck:String, wholeword:Bool = false):Bool {
		//trace("isinstring(\"" + fullstring + "\",\"" + stringtocheck + "\");");
		var p:Int = positioninstring(fullstring, stringtocheck);
		//trace("position in string: " + p);
		if (p != -1) {
			if(wholeword){
				if (p == 0 || !isalpha(mid(fullstring, p - 1))) {
					if (mid(fullstring, p - 1) == ".") {
						if (mid(fullstring, p - 2) == ".") {
							return true;
						}else{
							return false;
						}
					}
					if (mid(fullstring, p - 1) == "\"") {
						return false;
					}
					if (p + stringtocheck.length >= fullstring.length) {
						return true;
					}
					//trace("mid(fullstring, p + stringtocheck.length) = " + mid(fullstring, p + stringtocheck.length));
					if (!isalpha(mid(fullstring, p + stringtocheck.length))) {
						if(!isnumber(mid(fullstring, p + stringtocheck.length))){
						  return true;
						}
					}
				}
			}else {
				return true;
			}
		}
		return false;
	}
	
	/** Return the position of a substring in a given string. -1 if not found. */
	public static function positioninstring(fullstring:String, substring:String, start:Int = 0):Int {
		return (fullstring.indexOf(substring, start));
	}
	
	/** Return character at given position */
	public static function letterat(currentstring:String, position:Int = 0):String {
		return currentstring.substr(position, 1);
	}
	
	/** Return characters from the middle of a string. */
	public static function mid(currentstring:String, start:Int = 0, length:Int = 1):String {
		if (start < 0) return "";
		return currentstring.substr(start,length);
	}
	
	/** Return characters from the left of a string. */
	public static function left(currentstring:String, length:Int = 1):String {
		return currentstring.substr(0,length);
	}
	
	/** Return characters from the right of a string. */
	public static function right(currentstring:String, length:Int = 1):String {
		return currentstring.substr(currentstring.length - length, length);
	}
	
	public static function regexpin(fullstring:String, substring:String):Bool {
		var r = new EReg("(?:^|[^a-zA-Z])" + substring + "(?:$|[^a-zA-Z])", "i");
		if (r.match(fullstring)) {
		  return true;
		}
		return false;
	}
	
	/** Reverse a string. */
	public static function reversetext(currentstring:String):String {
		var t2:String = "";
		
		for (i in 0 ... currentstring.length) {
			t2 += mid(currentstring, currentstring.length-i-1, 1);
		}
		return t2;
	}
	
	/** Given a string currentstring, replace all occurances of string ch with ch2. Useful to remove characters. */
	public static function replacechar(currentstring:String, ch:String = "|", ch2:String = ""):String {
		var fixedstring:String = "";
		for (i in 0 ... currentstring.length) {
			if (mid(currentstring, i) == ch) {
				fixedstring += ch2;
			}else {
				fixedstring += mid(currentstring, i);
			}
		}
		return fixedstring;
	}
	
	/** Given a string currentstring, return everything after the LAST occurance of the "ch" character */
	public static function getlastbranch(currentstring:String, ch:String):String {
		var i:Int = currentstring.length - 1;
		while (i >= 0) {
			if (mid(currentstring, i, 1) == ch) {
				return mid(currentstring, i + 1, currentstring.length - i - 1);
			}
			i--;
		}
		return currentstring;
	}

/** Given a string currentstring, return everything before the LAST occurance of the "ch" character */
	public static function getlastroot(currentstring:String, ch:String):String {
		var i:Int = currentstring.length - 1;
		while (i >= 0) {
			if (mid(currentstring, i, 1) == ch) {
				return mid(currentstring, 0, i);
			}
			i--;
		}
		return currentstring;
	}
	
	/** Given a string currentstring, return everything before the first occurance of the "ch" character */
	public static function getroot(currentstring:String, ch:String):String {
		for (i in 0 ... currentstring.length) {
			if (mid(currentstring, i, 1) == ch) {
				return mid(currentstring, 0, i);
			}
		}
		return currentstring;
	}
	
	/** Given a string currentstring, return everything after the first occurance of the "ch" character */
	public static function getbranch(currentstring:String, ch:String):String {
		for (i in 0 ... currentstring.length) {
			if (mid(currentstring, i, 1) == ch) {
				return mid(currentstring, i + 1, currentstring.length - i - 1);
			}
		}
		return currentstring;
	}
	
	/** Given a string currentstring, return everything between the first and the last bracket (). */
	public static function getbetweenbrackets(currentstring:String):String {
		while (mid(currentstring, 0, 1) != "(" && currentstring.length > 0)	currentstring = mid(currentstring, 1, currentstring.length - 1);
		while (mid(currentstring, currentstring.length-1, 1) != ")" && currentstring.length > 0) currentstring = mid(currentstring, 0, currentstring.length - 1);
		
		if (currentstring.length <= 0) return "";
		return mid(currentstring, 1, currentstring.length - 2);
	}
	
	/** Given a string currentstring, return a string without spaces around it. */
	public static function trimspaces(currentstring:String):String {
		while (mid(currentstring, 0, 1) == " " && currentstring.length > 0)	currentstring = mid(currentstring, 1, currentstring.length - 1);
		while (mid(currentstring, currentstring.length - 1, 1) == " " && currentstring.length > 0) currentstring = mid(currentstring, 0, currentstring.length - 1);
		
		while (mid(currentstring, 0, 1) == "\t" && currentstring.length > 0) currentstring = mid(currentstring, 1, currentstring.length - 1);
		while (mid(currentstring, currentstring.length - 1, 1) == "\t" && currentstring.length > 0) currentstring = mid(currentstring, 0, currentstring.length - 1);
		
		if (currentstring.length <= 0) return "";
		return currentstring;
	}
	
	/** True if string currentstring is some kind of number; false if it's something else. */
	public static function isnumber(currentstring:String):Bool {
		if (Math.isNaN(Std.parseFloat(currentstring))) {
			return false;
		}else{
			return true;
		}	
		return false;
	}
}