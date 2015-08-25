package terrylibweb;

import openfl.display.DisplayObject;
import openfl.events.MouseEvent;
import openfl.ui.Mouse;
import openfl.events.Event;
import openfl.net.*;
import openfl.Lib;
	
class Mouse{		
	public static var x:Int;
	public static var y:Int;
	
	public static var mousewheel:Int = 0;
	
	public static var mouseoffstage:Bool;
	public static var isdragging:Bool;
	
	public static function leftheld():Bool { return _current > 0; }
	public static function leftclick():Bool { return _current == 2; }
	public static function leftreleased():Bool { return _current == -1; }
	
	public static function rightheld():Bool { return _rightcurrent > 0; }
	public static function rightclick():Bool { return _rightcurrent == 2; }	
	public static function rightreleased():Bool { return _rightcurrent == -1; }
	
	public static function middleheld():Bool { return _middlecurrent > 0; }
	public static function middleclick():Bool { return _middlecurrent == 2; }	
	public static function middlereleased():Bool { return _middlecurrent == -1; }
	
	private static function init(stage:DisplayObject) {
		//Right mouse stuff
		#if !flash
		stage.addEventListener(MouseEvent.RIGHT_MOUSE_DOWN, handleRightMouseDown);
		stage.addEventListener(MouseEvent.RIGHT_MOUSE_UP, handleRightMouseUp );
		#end
		
		stage.addEventListener(MouseEvent.MOUSE_DOWN, handleMouseDown);
		stage.addEventListener(MouseEvent.MOUSE_UP, handleMouseUp);
		stage.addEventListener(MouseEvent.MIDDLE_MOUSE_DOWN, handleMiddleMouseDown);
		stage.addEventListener(MouseEvent.MIDDLE_MOUSE_UP, handleMiddleMouseUp);
		stage.addEventListener(MouseEvent.MOUSE_WHEEL, mousewheelHandler);
		stage.addEventListener(MouseEvent.MOUSE_MOVE, mouseOver);
		stage.addEventListener(Event.MOUSE_LEAVE, mouseLeave);
		x = 0;
		y = 0;
		_rightcurrent = 0;
		_rightlast = 0;
		_middlecurrent = 0;
		_middlelast = 0;
		_current = 0;
		_last = 0;
	}		
	
	private static function mouseLeave(e:Event) {
		mouseoffstage = true;
		_current = 0;
		_last = 0;
		isdragging = false;
		_rightcurrent = 0;
		_rightlast = 0;
		_middlecurrent = 0;
		_middlelast = 0;
	}
	
	private static function mouseOver(e:MouseEvent) {
		mouseoffstage = false;
	}
	
	private static function mousewheelHandler( e:MouseEvent ) {
		mousewheel = e.delta;
	}
	
	public static function visitsite(t:String) {
		gotosite = t;
	}
	
	public static function update(X:Int,Y:Int){
		x = X;
		y = Y;
		
		if((_last == -1) && (_current == -1))
			_current = 0;
		else if((_last == 2) && (_current == 2))
			_current = 1;
		_last = _current;
		
		if((_rightlast == -1) && (_rightcurrent == -1))
			_rightcurrent = 0;
		else if((_rightlast == 2) && (_rightcurrent == 2))
			_rightcurrent = 1;
		_rightlast = _rightcurrent;
		
		if((_middlelast == -1) && (_middlecurrent == -1))
			_middlecurrent = 0;
		else if((_middlelast == 2) && (_middlecurrent == 2))
			_middlecurrent = 1;
		_middlelast = _middlecurrent;
	}
	
	private static function reset(){
		_current = 0;
		_last = 0;
		_rightcurrent = 0;
		_rightlast = 0;
		_middlecurrent = 0;
		_middlelast = 0;
	}
	
		
	#if !flash
		private static function handleRightMouseDown(event:MouseEvent) {	if (_rightcurrent > 0) { _rightcurrent = 1; } else { _rightcurrent = 2; } }
		private static function handleRightMouseUp(event:MouseEvent) {	if (_rightcurrent > 0) { _rightcurrent = -1; } else { _rightcurrent = 0; }	}
  #end
	
	private static function handleMiddleMouseDown(event:MouseEvent) {	if (_middlecurrent > 0) { _middlecurrent = 1; } else { _middlecurrent = 2; } }
	private static function handleMiddleMouseUp(event:MouseEvent) {	if (_middlecurrent > 0) { _middlecurrent = -1; } else { _middlecurrent = 0; }	}
	
	private static function handleMouseDown(event:MouseEvent) {
		if (Input.pressed(Key.CONTROL)) {
			if(_rightcurrent > 0) _rightcurrent = 1;
			else _rightcurrent = 2;
		}else{
			if(_current > 0) _current = 1;
			else _current = 2;
			
			if (_current == 2) {
				if (gotosite != "") {
					var link:URLRequest = new URLRequest(gotosite);
					Lib.getURL(link);
					gotosite = "";
				}
			}
		}
	}
	
	private static function handleMouseUp(event:MouseEvent) {		
		if(_rightcurrent > 0) _rightcurrent = -1;
		else _rightcurrent = 0;
		
		if(_current > 0) _current = -1;
		else _current = 0;
	}
	
	private static var _current:Int;
	private static var _last:Int;
	
	private static var _middlecurrent:Int;
	private static var _middlelast:Int;
	private static var _rightcurrent:Int;
	private static var _rightlast:Int;
	private static var gotosite:String = "";
}