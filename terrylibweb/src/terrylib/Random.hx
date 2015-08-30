package terrylib;

class Random{
	/** Return a random boolean value (true or false) */
	#if terrylibweb
	public static function bool():Bool{
	#else
	public static inline function bool():Bool{
	#end
		return random() < 0.5;
	}
	
	/** True 1/5th of the time */
	#if terrylibweb
	public static function occasional():Bool{
	#else
	public static inline function occasional():Bool{
	#end
		return random() < 0.2;
	}

	/** True 5% or 1/20th of the time */
	#if terrylibweb
	public static function rare():Bool{
	#else
	public static inline function rare():Bool{
	#end
		return random() < 0.05;
	}
	
	/** Return a random integer between 'from' and 'to', inclusive. */
	#if terrylibweb
	public static function int(from:Int, to:Int):Int {
	#else
	public static inline function int(from:Int, to:Int):Int {
	#end
		return from + Math.floor(((to - from + 1) * random()));
	}
	
	/** Return a random float between 'from' and 'to', inclusive. */
	#if terrylibweb
	public static function float(from:Float, to:Float):Float{
	#else
	public static inline function float(from:Float, to:Float):Float{
	#end
		return from + ((to - from) * random());
	}

	/** Return a random string of a certain length.  You can optionally specify 
	    which characters to use, otherwise the default is (a-zA-Z0-9) */
	public static function string(length:Int, ?charactersToUse = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):String{
		var str = "";
		for (i in 0...length){
			str += charactersToUse.charAt(int(0, charactersToUse.length - 1));
		}
		return str;
	}

	//These functions are pretty ugly, but useful!
	
	/** Return a random string from a list of up to 12 strings. */
	public static function pickstring(s1:String, s2:String, s3:String = "", s4:String = "",
																					 s5:String = "", s6:String = "", s7:String = "", s8:String = "",
																					 s9:String = "", s10:String = "", s11:String = "",s12:String = ""):String{
	  temp = 2;
		if (s3 != "") temp = 3;
	  if (s4 != "") temp = 4;
	  if (s5 != "") temp = 5;
	  if (s6 != "") temp = 6;
	  if (s7 != "") temp = 7;
	  if (s8 != "") temp = 8;
	  if (s9 != "") temp = 9;
	  if (s10 != "") temp = 10;
	  if (s11 != "") temp = 11;
	  if (s12 != "") temp = 12;
		
		switch(int(1, temp)) {
			case 1: return s1;
			case 2: return s2;
			case 3: return s3;
			case 4: return s4;
			case 5: return s5;
			case 6: return s6;
			case 7: return s7;
			case 8: return s8;
			case 9: return s9;
			case 10: return s10;
			case 11: return s11;
			case 12: return s12;
		}
		
		return s1;
	}
	
	/** Return a random Int from a list of up to 12 Ints. */
	public static function pickint(s1:Int, s2:Int, s3:Int = -10000, s4:Int = -10000,
																					 s5:Int = -10000, s6:Int = -10000, s7:Int = -10000, s8:Int = -10000,
																					 s9:Int = -10000, s10:Int = -10000, s11:Int = -10000,s12:Int = -10000):Int{
	  temp = 2;
    if (s3 != -10000) temp = 3;
	  if (s4 != -10000) temp = 4;
	  if (s5 != -10000) temp = 5;
	  if (s6 != -10000) temp = 6;
	  if (s7 != -10000) temp = 7;
	  if (s8 != -10000) temp = 8;
	  if (s9 != -10000) temp = 9;
	  if (s10 != -10000) temp = 10;
	  if (s11 != -10000) temp = 11;
	  if (s12 != -10000) temp = 12;
		
		switch(int(1, temp)) {
			case 1: return s1;
			case 2: return s2;
			case 3: return s3;
			case 4: return s4;
			case 5: return s5;
			case 6: return s6;
			case 7: return s7;
			case 8: return s8;
			case 9: return s9;
			case 10: return s10;
			case 11: return s11;
			case 12: return s12;
		}
		
		return s1;
	}
	
	/** Return a random Float from a list of up to 12 Floats. */
	public static function pickfloat(s1:Float, s2:Float, s3:Float = -10000, s4:Float = -10000,
																					 s5:Float = -10000, s6:Float = -10000, s7:Float = -10000, s8:Float = -10000,
																					 s9:Float = -10000, s10:Float = -10000, s11:Float = -10000,s12:Float = -10000):Float{
	  temp = 2;
    if (s3 != -10000) temp = 3;
	  if (s4 != -10000) temp = 4;
	  if (s5 != -10000) temp = 5;
	  if (s6 != -10000) temp = 6;
	  if (s7 != -10000) temp = 7;
	  if (s8 != -10000) temp = 8;
	  if (s9 != -10000) temp = 9;
	  if (s10 != -10000) temp = 10;
	  if (s11 != -10000) temp = 11;
	  if (s12 != -10000) temp = 12;
		
		switch(int(1, temp)) {
			case 1: return s1;
			case 2: return s2;
			case 3: return s3;
			case 4: return s4;
			case 5: return s5;
			case 6: return s6;
			case 7: return s7;
			case 8: return s8;
			case 9: return s9;
			case 10: return s10;
			case 11: return s11;
			case 12: return s12;
		}
		
		return s1;
	}
	
	public static function random():Float {
		seed = (seed * 9301 + 49297) % 233280; 
		return Math.abs(seed/(233280));
	}
	
	public static function setseed(s:Int) {
		seed = Std.int(Math.abs(s % 233280));
	}
	
	private static var temp:Int;
	public static var seed:Int = 0;
}