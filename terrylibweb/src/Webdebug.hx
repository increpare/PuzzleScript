import terrylib.*;
import openfl.external.ExternalInterface;

class Webdebug{
	public static function log(msg:String){
		ExternalInterface.call("consoleError", msg, true);
	}
	
	public static function warn(msg:String, linenum:Int){
		ExternalInterface.call("logWarning", msg, linenum, true);
	}
	
	public static function warn_noline(msg:String){
		ExternalInterface.call("logWarningNoLine", msg, true);
	}
}