import terrylibweb.*;
import openfl.external.ExternalInterface;

class Webdebug{
	public static function log(msg:String){
		ExternalInterface.call("consolePrint", msg, true);
	}
}