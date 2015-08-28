import terrylib.*;
import hscript.*;
import openfl.Assets;
import openfl.external.ExternalInterface;

#if flash
	import openfl.events.*;
	import openfl.net.*;
#end

class Webscript {
	public static var myscript:String;
	public static var parsedscript:Expr;
	public static var parser:Parser;
	public static var interpreter:Interp;
	
	public static var scriptloaded:Bool;
	public static var runscript:Bool;
	public static var errorinscript:Bool;
	
	public static var initfunction:Dynamic;
	public static var updatefunction:Dynamic;
	
	public static var title:String;
	public static var homepage:String;
	public static var background_color:Int;
	
	public static function init() {
		scriptloaded = false;
		runscript = false;
		errorinscript = false;
		
		Text.setfont(Webfont.ZERO4B11, 1);
		Text.setfont(Webfont.APPLE, 1);
		Text.setfont(Webfont.BOLD, 1);
		Text.setfont(Webfont.C64, 1);
		Text.setfont(Webfont.CASUAL, 1);
		Text.setfont(Webfont.COMIC, 1);
		Text.setfont(Webfont.CRYPT, 1);
		Text.setfont(Webfont.DOS, 1);
		Text.setfont(Webfont.GANON, 1);
		Text.setfont(Webfont.HANDY, 1);
		Text.setfont(Webfont.NOKIA, 1);
		Text.setfont(Webfont.OLDENGLISH, 1);
		Text.setfont(Webfont.PIXEL, 1);
		Text.setfont(Webfont.PRESSSTART, 1);
		Text.setfont(Webfont.RETROFUTURE, 1);
		Text.setfont(Webfont.ROMAN, 1);
		Text.setfont(Webfont.SPECIAL, 1);
		Text.setfont(Webfont.VISITOR, 1);
		Text.setfont(Webfont.YOSTER, 1);
		
		Text.setfont(Webfont.DEFAULT, 1);
		
		try {
			ExternalInterface.addCallback("loadscript", loadscript);
		}catch (e:Dynamic) {
			//Ok, try loading this locally for testing
			#if flash
			loadfile();
			#end
		}
	}
	
	#if flash
	
	public static var myLoader:URLLoader;
	public static function loadfile():Void {
		//make a new loader
    myLoader = new URLLoader();
    //new request - for a file in the same folder called 'someTextFile.txt'
    var myRequest:URLRequest = new URLRequest("script.txt");
		
		//wait for the load
    myLoader.addEventListener(Event.COMPLETE, onLoadComplete);
		myLoader.addEventListener(IOErrorEvent.IO_ERROR, onIOError);
		
    //load!
    myLoader.load(myRequest);
	}
	
	public static function onIOError(e:Event):Void {
		trace("\"script.txt\" not found.");
	}
	
	public static function onLoadComplete(e:Event):Void {
		myscript = Convert.tostring(myLoader.data);
		
		scriptfound();
	}
	
	public static var	reloaddelay:Int = 0;
	#end
	
	public static function printError(e:Dynamic):String{
		
		if ( Std.is(e,hscript.Expr.Error) ) {	
			var errstr : String = e.e[0];
			for (i in 2 ... e.e.length){
				errstr = errstr + " " + e.e[i];
			}
			return errstr;
		}
		if (e.name == "TypeError"){
			return e.stack;
		}
		return e.toString();
	}

	public static function update() {
		#if flash
		  if (Input.justpressed(Key.R)) {
				reloaddelay = 5;
			}
		#end
		#if flash
		if (reloaddelay > 0) {
			Gfx.clearscreen(Col.BLACK);
			reloaddelay--;
			if (reloaddelay <= 0) loadfile();
		}else	if (errorinscript) {
		#else
		if (errorinscript) {
		#end
			Text.setfont("default", 1);
			Gfx.clearscreen(Gfx.RGB(32, 0, 0));
			Text.display(Text.CENTER, Text.CENTER, "ERROR! ERROR! ERROR!", Col.RED);
		}else if (scriptloaded) {
			if (runscript) {
				try {
					updatefunction();
				}catch (e:Dynamic) {
					var errorMessage = printError(e);
					Webdebug.error("RUNTIME ERROR: " + errorMessage);
					Gfx.resizescreen(192, 120, 4);
					errorinscript = true;
					runscript = false;
				}
			}	
		}else {
			counter+=10;
			Gfx.clearscreen(Col.BLUE);
			Gfx.fillbox(4, 4, Gfx.screenwidth - 8, Gfx.screenheight - 8, Col.NIGHTBLUE);
			
			Text.display(Gfx.screenwidth - 6, Gfx.screenheight - Text.height()-4, "terrylib alpha v0.1", Col.WHITE, { align:Text.RIGHT } );
			
			var msg:String = "WAITING FOR SCRIPTFILE...";
			var startpos:Float = Gfx.screenwidthmid - Text.len(msg) / 2;
			var currentpos:Float = 0;
			for (i in 0 ... msg.length) {
				if (S.mid(msg, i, 1) != "_") {
					Text.display(startpos + currentpos, Gfx.screenheightmid - 10 + Math.sin((((i*5)+counter)%360) * Math.PI * 2 / 360)*5, S.mid(msg, i, 1), Col.WHITE);
				}
				currentpos += Text.len(S.mid(msg, i, 1));
			}
		}
		
		if (Gfx.showfps) {
			oldfont = Text.currentfont;
			oldfontsize = Text.currentsize;
			Text.setfont("pixel", 1);
			Text.display(Gfx.screenwidth - 4, 1, "FPS: " + Gfx.fps(), Col.YELLOW, { align: Text.RIGHT});
			Text.setfont(oldfont, oldfontsize);
		}
	}
	private static var counter:Int = 0;
	private static var oldfont:String = "";
	private static var oldfontsize:Int = 0;
	
	public static function loadscript(script:String) {
		myscript = script;
		scriptfound();
	}
	
	public static function scriptfound(){
		scriptloaded = true;
		errorinscript = false;
    parser = new hscript.Parser();
		parser.allowTypes = true;
    interpreter = new hscript.Interp();
		
		interpreter.variables.set("Random", Random);
		interpreter.variables.set("Math", Math);
		interpreter.variables.set("Col", Col);
		interpreter.variables.set("Convert", Convert);
		interpreter.variables.set("Debug", Webdebug);
		interpreter.variables.set("Gfx", Gfx);
		interpreter.variables.set("Input", Input);
		interpreter.variables.set("Key", Key);
		interpreter.variables.set("Game", Game);
		interpreter.variables.set("Mouse", Mouse);
		interpreter.variables.set("Music", Webmusic);
		interpreter.variables.set("Text", Text);
		interpreter.variables.set("Font", Webfont);
		interpreter.variables.set("Std", Std);
		
		runscript = true;
		try{
			parsedscript = parser.parseString(myscript);
		}catch (e:Dynamic) {
			/*
				e looks like
				{
					e { error name , error code id, ? possibly data associated with that error}
				}
			*/
			var errstr : String = printError(e);
			Webdebug.error("Error:"+errstr);
			runscript = false;
			errorinscript = true;
			Gfx.resizescreen(192, 120, 4);
		}
		
		if (runscript) {
			try{
				interpreter.execute(parsedscript);
			}catch (e:hscript.Expr.Error) {
				var errstr : String = printError(e);
				Webdebug.error("Initilisation error:"+errstr, parser.line);
				runscript = false;
				errorinscript = true;
				Gfx.resizescreen(192, 120, 4);
			}
			
			title = interpreter.variables.get("title");
			if (title == null) title = "Untitled";
			homepage = interpreter.variables.get("homepage");
			if (homepage == null) homepage = "";
			var bg_col:Dynamic = interpreter.variables.get("background_color");
			if (bg_col == null) {
				background_color = Col.BLACK;
			}else{
			  background_color = Convert.toint(bg_col);
			}
			
			initfunction = interpreter.variables.get("new");
			updatefunction = interpreter.variables.get("update");
			
			//Set default font
			Text.setfont("default", 1);
			if (initfunction != null) {
				try{
					initfunction();	
				}catch (e:Dynamic) {
					var errstr : String = printError(e);
					Webdebug.error("Error in new(): " + errstr, parser.line);
					runscript = false;
					errorinscript = true;
					Gfx.resizescreen(192, 120, 4);
				}
			}
			
			if (updatefunction == null) {
				Webdebug.error("Error: An \"update\" function is required. e.g.");
				Webdebug.error("function update(){");
				Webdebug.error("}");
				runscript = false;
				errorinscript = true;
				
				Gfx.resizescreen(192, 120, 4);
			}
		}
	}	
}