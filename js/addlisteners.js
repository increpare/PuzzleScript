var canvas = document.getElementById("gameCanvas");
var attach = function(elt, evt, hnd) {
	if(elt.addEventListener) {
		elt.addEventListener(evt,hnd,false);
	} else if(elt.attachEvent) {
		elt.attachEvent(evt,hnd);
	}
}
attach(canvas, "contextmenu", rightClickCanvas);
attach(canvas, "mousemove", mouseMove);
attach(canvas, "mouseout", mouseOut);
attach(document, "mousedown", onMouseDown);
attach(document, "mouseup", onMouseUp);
attach(document, "keydown", onKeyDown);
attach(document, "keyup", onKeyUp);
attach(window, 'focus', onMyFocus);
attach(window, 'blur', onMyBlur);

// Lights, camera…function!
setInterval(function() {
    update();
}, deltatime);
