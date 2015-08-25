// Huge huge thanks to Ruari O'Sullivan (@randomnine) for figuring this stuff out for me!
#if !terrylibweb
package terrylib;

import openfl.errors.ArgumentError;

class Scene {
	private static function init() {
		scenelist = new Array<Dynamic>();
		#if neko
		  try{
		    scenelist.push(Type.createInstance(Main, []));
			}catch (e:Dynamic) {
				throw("ERROR: Neko builds require that Main.hx has a \"new()\" function.");
			}
		#else
		  scenelist.push(Type.createInstance(Main, []));
		#end
		currentscene = 0;
	}
	
	private static function update() {
		callscenemethod(scenelist[currentscene], "update");
	}
	
	private static function callscenemethod(scene:Dynamic, method:String) {
		var instanceFunc:Dynamic = Reflect.field(scenelist[currentscene], method);
		if (instanceFunc != null && Reflect.isFunction(instanceFunc)) {
			try {
				Reflect.callMethod(scenelist[currentscene], instanceFunc, []);
			} catch ( e:ArgumentError ) {
				throw( "ERROR: Couldn't call " + Type.getClassName(scene) + "." + method + "() without any arguments.");
			}
			return;
		}
		
		// Now try the static method
		var classFunc:Dynamic = Reflect.field(Type.getClass(scenelist[currentscene]), method);
		if (classFunc != null && Reflect.isFunction(classFunc)) {
			try {
				Reflect.callMethod(scenelist[currentscene], classFunc, []);
			} catch ( e:ArgumentError ) {
				throw( "ERROR: Couldn't call " + Type.getClassName(scene) + "." + method + "() without any arguments.");
			}
			return;
		}
		
		// method didn't exist; complain if necessary
	}
	
	public static function change(newscene:Class<Dynamic>) {
		for (i in 0 ... scenelist.length) {
			if (newscene == Type.getClass(scenelist[i])) {
				currentscene = i;
				callscenemethod(scenelist[currentscene], "reset");
				return;
			}
		}
		
		#if neko
		  try{
		    scenelist.push(Type.createInstance(newscene, []));
				currentscene = scenelist.length - 1;
			}catch (e:Dynamic) {
				throw("ERROR: Neko builds require all classes to have a \"new()\" function.");
			}
		#else
		  scenelist.push(Type.createInstance(newscene, []));
			currentscene = scenelist.length - 1;
		#end
	}
	
	public static function get<T>(newscene:Class<T>):T {
	  for (i in 0 ... scenelist.length) {
			if (newscene == Type.getClass(scenelist[i])) {
				return scenelist[i];
			}
		}
		
		throw("ERROR: Scene has not been created yet!");
	}
	
	private static var scenelist:Array<Dynamic>;
	private static var currentscene:Int;
}

#end