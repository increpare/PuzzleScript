import terrylib.*;
import hscript.*;
import openfl.Assets;
import openfl.external.ExternalInterface;

@:expose
class MyClass {
  
  public function runScript(s:String) {
  	Scene.get(Main).loadscript(s);
  }
}

class Main {
	public function loadscriptfile(scriptname:String):String {
		var tempstring:String = Assets.getText("data/" + scriptname + ".txt");
		//Remove class bit from string eventually
		
		var bracketcount:Int;
		var classposition:Int;
		
		return tempstring;
	}
	
	public var myscript:String;
	public var parsedscript:Expr;
	public var parser:Parser;
	public var interpreter:Interp;
	
	public var scriptloaded:Bool;
	public var runscript:Bool;
	
	public var initfunction:Dynamic;
	public var updatefunction:Dynamic;
	
	public function new() {
		ExternalInterface.addCallback("loadscript", loadscript);
		
		scriptloaded = false;
		runscript = false;
	}
	
	
	public function loadscript(script:String):Void {
		myscript = script;
		scriptfound();
	}
	
	public function scriptfound(){
		trace(myscript);
		scriptloaded=true;
    parser = new hscript.Parser();
		parser.allowTypes = true;
    interpreter = new hscript.Interp();
		
		//interpreter.variables.set("Lib", Webfunctions);
		interpreter.variables.set("Col", Col);
		interpreter.variables.set("Convert", Convert);
		//interpreter.variables.set("Core", Core);
		interpreter.variables.set("Debug", Debug);
		interpreter.variables.set("Gfx", Gfx);
		interpreter.variables.set("Input", Input);
		interpreter.variables.set("Key", Key);
		//interpreter.variables.set("Load", Load);
		interpreter.variables.set("Mouse", Mouse);
		interpreter.variables.set("Music", Music);
		interpreter.variables.set("Random", Random);
		//interpreter.variables.set("Scene", Scene);
		interpreter.variables.set("Text", Text);
    
		//myscript = loadscriptfile("test");
		runscript = true;
		try{
			parsedscript = parser.parseString(myscript);
		}catch (e:hscript.Expr.Error) {
			Debug.log("Error in line " + parser.line);
			runscript = false;
		}
		
		if (runscript) {
			interpreter.execute(parsedscript);
			trace(interpreter.variables.get("new"));
			initfunction = interpreter.variables.get("new");
			updatefunction = interpreter.variables.get("update");
			
			initfunction();
		}
	}
	
	public function update() {
		if (scriptloaded) {
			if (runscript) {
				try {
					updatefunction();
				}catch (e:Dynamic) {
					Debug.log(e);
					Debug.log("RUNTIME ERROR:");
					runscript = false;
				}
			}	
		}else {
			Gfx.clearscreen(Col.NIGHTBLUE);
			Text.display(Text.CENTER, Text.CENTER, "WAITING FOR SCRIPTFILE...");
		}
  }
}