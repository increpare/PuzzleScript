#if terrylibweb
package terrylib;

import openfl.Assets;

class Data {
	@:generic
	public static function blank2darray<T>(width:Int, height:Int):Array<Array<T>> {
		var returnedarray2d:Array<Array<T>> = [for (x in 0 ... width) [for (y in 0 ... height) cast ""]];
		return returnedarray2d;
	}
}
#else
package terrylib;

import openfl.Assets;

class Data {
	public static var width:Int = 0;
	public static var height:Int = 0;
	
	public static function loadtext(textfile:String):Array<String> {
		tempstring = Assets.getText("data/text/" + textfile + ".txt");
		tempstring = replacechar(tempstring, "\r", "");
		
		return tempstring.split("\n");
	}
	
	@:generic
	public static function loadcsv<T>(csvfile:String, delimiter:String = ","):Array<T> {
		tempstring = Assets.getText("data/text/" + csvfile + ".csv");
		
		//figure out width
		width = 1;
		var i:Int = 0;
		while (i < tempstring.length) {
			if (mid(tempstring, i) == delimiter) width++;
			if (mid(tempstring, i) == "\n") {
				break;
			}
			i++;
		}
		
		tempstring = replacechar(tempstring, "\r", "");
		tempstring = replacechar(tempstring, "\n", delimiter);
		
		var returnedarray:Array<T> = new Array<T>();
		var stringarray:Array<String> = tempstring.split(delimiter);
		
		for (i in 0 ... stringarray.length) {
			returnedarray.push(cast stringarray[i]);
		}
		
		height = Std.int(returnedarray.length / width);
		return returnedarray;
	}
	
	@:generic
	public static function blank2darray<T>(width:Int, height:Int):Array<Array<T>> {
		var returnedarray2d:Array<Array<T>> = [for (x in 0 ... width) [for (y in 0 ... height) cast ""]];
		return returnedarray2d;
	}
	
	@:generic
	public static function loadcsv_2d<T>(csvfile:String, delimiter:String = ","):Array<Array<T>> {
		tempstring = Assets.getText("data/text/" + csvfile + ".csv");
		
		//figure out width
		width = 1;
		var i:Int = 0;
		while (i < tempstring.length) {
			if (mid(tempstring, i) == delimiter) width++;
			if (mid(tempstring, i) == "\n") {
				break;
			}
			i++;
		}
		
		tempstring = replacechar(tempstring, "\r", "");
		tempstring = replacechar(tempstring, "\n", delimiter);
		
		var returnedarray:Array<T> = new Array<T>();
		var stringarray:Array<String> = tempstring.split(delimiter);
		
		for (i in 0 ... stringarray.length) {
			returnedarray.push(cast stringarray[i]);
		}
		
		height = Std.int(returnedarray.length / width);
		
		var returnedarray2d:Array<Array<T>> = [for (x in 0 ... width) [for (y in 0 ... height) returnedarray[x + (y * width)]]];
		return returnedarray2d;
	}
	
	/** Return characters from the middle of a string. */
	private static function mid(currentstring:String, start:Int = 0, length:Int = 1):String {
		return currentstring.substr(start,length);
	}
	
	private static function replacechar(currentstring:String, ch:String = "|", ch2:String = ""):String {
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
	
	private static var tempstring:String;
}
#end