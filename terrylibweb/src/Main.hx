import terrylib.*;

@:expose
class MyClass {
  public function runScript(s:String) {
		Webscript.loadscript(s);
  }
}

class Main {
	public function new() {
		Webscript.init();
	}
	
	public function update() {
		Webscript.update();
  }
}