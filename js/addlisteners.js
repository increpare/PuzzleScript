

    onmousemove="mouseMove(event)" 
    onmouseout="mouseOut()"

var el = document.getElementById("gameCanvas");
if (el.addEventListener) {
    el.addEventListener("contextmenu", rightClickCanvas, false);
    el.addEventListener("mousemove", mouseMove, false);
    el.addEventListener("mouseout", mouseOut, false);
} else {
    el.attachEvent('oncontextmenu', rightClickCanvas);
    el.attachEvent('onmousemove', mouseMove);
    el.attachEvent('onmouseout', mouseOut);
}  

window.onbeforeunload = function (e) {
	var e = e || window.event;
	var msg = 'You have unsaved changes!';

	if(_editorDirty) {			

		// For IE and Firefox prior to version 4
		if (e) {
			e.returnValue = msg;
		}

		// For Safari
		return msg;
	}
};