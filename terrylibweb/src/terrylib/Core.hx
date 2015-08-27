package terrylib;

import openfl.utils.Timer;
import openfl.display.*;
import openfl.events.*;
import openfl.Lib;

@:access(Main)
@:access(terrylib.Gfx)
@:access(terrylib.Music)
@:access(terrylib.Mouse)
@:access(terrylib.Input)
@:access(terrylib.Scene)
class Core extends Sprite {
	public function new() {
		super();
		
		init();
	}
	
	public function init() {
		//Init library classes
		Random.setseed(Std.int(Math.random() * 233280));
		Input.init(this.stage);
		Mouse.init(this.stage);
		Gfx.init(this.stage);
		#if terrylibweb
		#else
		Music.init();
		#end
		
		//Default setup
		#if terrylibweb
			Gfx.resizescreen(192, 120, 4);
			Text.addfont("retrofuture", 1);
		#else
			Gfx.resizescreen(768, 480);
			Text.addfont("opensans", 24);
		#end
		
		#if terrylibweb
		terrylibmain = new Main();
		#else
		Scene.init();
		#end
		
		_rate = 1000 / TARGET_FPS;
	  _skip = _rate * 10;
		_timer.addEventListener(TimerEvent.TIMER, update);
		_timer.start();
	}
	
	public function update(e:TimerEvent) {
		Gfx.skiprender = false;
		_current = Lib.getTimer();
		if (_last < 0) _last = _current;
		_delta += _current - _last;
		_last = _current;
		if (_delta >= _rate){
			_delta %= _skip;
			if (_delta >= _rate) {
				_delta -= _rate;
				while (_delta >= _rate) {
				  _delta -= _rate;
					Gfx.skiprender = true;
					doupdate();
				}
			}
			Gfx.skiprender = false;
			doupdate();
			e.updateAfterEvent();
		}
	}
	
	public function doupdate() {
		Mouse.update(Std.int(Lib.current.mouseX / Gfx.screenscale), Std.int(Lib.current.mouseY / Gfx.screenscale));
		Input.update();
		
		if(!Gfx.skiprender) Gfx.backbuffer.lock();
		
		Gfx.clearscreen();
		#if terrylibweb
		terrylibmain.update();
		#else
		Scene.update();
		#end
		Text.drawstringinput();
		Debug.showlog();
		
		if(!Gfx.skiprender) Gfx.backbuffer.unlock();
	}
	
	#if terrylibweb
		public var terrylibmain:Main;
	#end
	private var TARGET_FPS:Int = 60;
	private var _rate:Float;
	private var _skip:Float;
	private var _last:Float = -1;
	private var _current:Float = 0;
	private var _delta:Float = 0;
	private var _timer:Timer = new Timer(4);
}