import terrylib.*;
import hscript.Expr.Error;

class Err {
	//Central place to report errors.
	public static var PRE_BRACKETMISMATCH:Int = 0;
	public static var PRE_MISSINGUPDATE:Int = 1;
	public static var PARSER_INIT:Int = 2;
	public static var PARSER_NEW:Int = 3;
	public static var RUNTIME_INIT:Int = 4;
	public static var RUNTIME_UPDATE:Int = 5;
	public static var RUNTIME_FUNCTION:Int = 6;
	
	public static function log(errorcode:Int, ?linenum:Int, ?details:Array<String>) {
		Webscript.runscript = false;
		Webscript.errorinscript = true;
		
		Gfx.resizescreen(192, 120, 4);
		Text.setfont("default", 1);
		
		if (errorcode == PRE_BRACKETMISMATCH) {
			Webdebug.error("Error: Bracket mismatch.");
			Webdebug.error("(Missing a { or } bracket somewhere.)");
		}else if (errorcode == PRE_MISSINGUPDATE){
			Webdebug.error("Error: An \"update()\" function is required.");
		}else if (errorcode == PARSER_INIT) {
			Webdebug.error("Parser error in processing script file.");
			outputdetails(details);
		}else if (errorcode == PARSER_NEW){
			Webdebug.error("Runtime error in function new().", linenum);
			outputdetails(details);
		}else if (errorcode == RUNTIME_INIT) {
			Webdebug.error("Runtime error in initial run.", linenum);
			outputdetails(details);
		}else if (errorcode == RUNTIME_UPDATE){
			Webdebug.error("Runtime error.", linenum);
			outputdetails(details);
		}else if (errorcode == RUNTIME_FUNCTION){
			Webdebug.error("Runtime error in function:");
			outputdetails(details);
		}
	}
	
	private static function outputdetails(details:Array<String>) {
		if (details != null) {	
			if (details.length > 0) {
				for (i in 0 ... details.length) {
					if(details[i] != null){
					  Webdebug.error(details[i]);
					}
				}	
			}
		}
	}
	
	public static function process(e:Dynamic):Array<String> {
		/*
			e looks like
			{
				e { error name , error code id, ? possibly data associated with that error}
			}
		*/
		if (Std.is(e, hscript.Expr.Error) ) {	
			var errstr:String;
			try{
			  errstr = e.e[0];
				if (e.e[0] == "EUnknownVariable") {
					errorstart = e.pmin;
					errorend = e.pmax;
					geterrorline();
					return ["Unknown variable \"" + e.e[2] + "\" in line " + errorline, S.Webscript.loadedscript[errorline]];
				}else{
					for (i in 2 ... e.e.length){
						errstr = errstr + " " + e.e[i];
					}
					return [errstr];
				}
			}catch (err:Dynamic) {
				return [e.toString()];
			}
		}
		
		if (e.name == "TypeError") {
			#if flash
			//trace(CallStack.toString(callStack));
			return ["TypeError"];
			#else
			return [e.stack];
			#end
		}
		return [e.toString()];
	}
	
	public static function geterrorline() {
		var charcount:Int = 0;
		
		var i:Int = 0;
		while (i < Webscript.loadedscript.length) {
			charcount += Webscript.loadedscript[i].length;
			if (charcount > errorstart) {
				errorline = i;
				break;
			}
			i++;
		}
	}
	
	public static var errorline:Int;
	public static var errorstart:Int;
	public static var errorend:Int;
}