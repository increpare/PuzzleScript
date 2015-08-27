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
    var myRequest:URLRequest = new URLRequest("schript.txt");
		
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
	#end
	
	public static function update() {
		if (errorinscript) {
			Gfx.clearscreen(Gfx.RGB(32, 0, 0));
			Text.display(Text.CENTER, Text.CENTER, "ERROR! ERROR! ERROR!", Col.RED);
		}else if (scriptloaded) {
			if (runscript) {
				try {
					updatefunction();
				}catch (e:Dynamic) {
					Webdebug.log("RUNTIME ERROR:");
					Webdebug.log(Convert.tostring(e));
					errorinscript = true;
					runscript = false;
				}
			}	
		}else {
			counter+=5;
			Gfx.clearscreen(Col.BLUE);
			Gfx.fillbox(4, 4, Gfx.screenwidth - 8, Gfx.screenheight - 8, Col.NIGHTBLUE);
			
			Text.display(Gfx.screenwidth - 6, Gfx.screenheight - Text.height()-4, "terrylib alpha v0.1", Col.WHITE, { align:Text.RIGHT } );
			
			var msg:String = "WAITING_FOR_SCRIPTFILE...";
			var startpos:Float = Gfx.screenwidthmid - Text.len(msg) / 2;
			var currentpos:Float = 0;
			for (i in 0 ... msg.length) {
				if (S.mid(msg, i, 1) != "_") {
					Text.display(startpos + currentpos, Gfx.screenheightmid - 10 + Math.sin((((i*5)+counter)%360) * Math.PI * 2 / 360)*5, S.mid(msg, i, 1), Col.WHITE);
				}
				currentpos += Text.len(S.mid(msg, i, 1));
			}
			
		}
	}
	private static var counter:Int = 0;
	
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
		
		interpreter.variables.set("Math", Math);
		interpreter.variables.set("Col", Col);
		interpreter.variables.set("Convert", Convert);
		interpreter.variables.set("Debug", Webdebug);
		interpreter.variables.set("Gfx", Gfx);
		interpreter.variables.set("Input", Input);
		interpreter.variables.set("Key", Key);
		interpreter.variables.set("Game", Game);
		//interpreter.variables.set("Load", Load);
		interpreter.variables.set("Mouse", Mouse);
		interpreter.variables.set("Music", Webmusic);
		interpreter.variables.set("Random", Random);
		//interpreter.variables.set("Scene", Scene);
		interpreter.variables.set("Text", Text);
    
		//myscript = loadscriptfile("test");
		runscript = true;
		try{
			parsedscript = parser.parseString(myscript);
		}catch (e:hscript.Expr.Error) {
			Webdebug.log("Error in line " + parser.line);
			runscript = false;
			errorinscript = true;
		}
		
		if (runscript) {
			interpreter.execute(parsedscript);
			
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
			
			if (initfunction != null) {
				initfunction();	
			}
			
			if (updatefunction == null) {
				Webdebug.log("Error: An \"update\" function is required. e.g.");
				Webdebug.log("function update(){");
				Webdebug.log("}");
				runscript = false;
				errorinscript = true;
			}
		}
	}	
}