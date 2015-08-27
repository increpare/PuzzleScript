import terrylib.*;

@:expose
class Webbridge {
	public function runScript(s:String) {
		Webscript.loadscript(s);
  }
	
	public function get_background_colour():Int {
		return Webscript.background_color;
	}
	
	public function get_title():String {
		return Webscript.title;
	}
	
	public function get_homepage():String {
		return Webscript.homepage;
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