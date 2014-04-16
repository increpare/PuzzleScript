var soundbarwidth = 440;
var lowerbarheight = 18;
var upperbarheight = 30;

function resize_widths(verticaldragbarX){
	document.getElementById("leftpanel").style.width = verticaldragbarX + "px";
	document.getElementById("righttophalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("rightbottomhalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("horizontaldragbar").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("verticaldragbar").style.left = verticaldragbarX + "px";
	canvasResize();
}

function resize_heights(horizontaldragbarY){
	document.getElementById("leftpanel").style.height = (window.innerHeight - 30) + "px";
	document.getElementById("verticaldragbar").style.height = (window.innerHeight - 30) + "px";
	document.getElementById("righttophalf").style.height = horizontaldragbarY - upperbarheight + "px";
	document.getElementById("rightbottomhalf").style.top = horizontaldragbarY + 3 + "px";
	document.getElementById("horizontaldragbar").style.top = horizontaldragbarY + "px";
	canvasResize();
}

function resize_all(){
	verticaldragbarX = parseInt(document.getElementById("verticaldragbar").style.left.replace("px",""));
	if ((window.innerWidth - verticaldragbarX) < soundbarwidth){
		verticaldragbarX = window.innerWidth - soundbarwidth;
	} else if ((verticaldragbarX < window.innerWidth/2)){
		verticaldragbarX = window.innerWidth/2;
	}
	resize_widths(verticaldragbarX);
	
	horizontaldragbarY = parseInt(document.getElementById("horizontaldragbar").style.top.replace("px",""));
	if ((window.innerHeight - horizontaldragbarY) < (lowerbarheight)){
		horizontaldragbarY = window.innerHeight - lowerbarheight - 7;
	} else if ((horizontaldragbarY < window.innerHeight/2)){
		horizontaldragbarY = window.innerHeight/2;
	};
	resize_heights(horizontaldragbarY);
};

function verticalDragbarMouseDown(e) {
    e.preventDefault();
    window.addEventListener("mousemove", verticalDragbarMouseMove, false);
	window.addEventListener("mouseup", verticalDragbarMouseUp, false);
};

function verticalDragbarMouseMove(e) {
	if ((window.innerWidth - e.pageX) > soundbarwidth){
		resize_widths(e.pageX + 2);
	} else {
		resize_widths(window.innerWidth - soundbarwidth);
	};
};

function verticalDragbarMouseUp(e) {
    window.removeEventListener("mousemove", verticalDragbarMouseMove, false);
};

function horizontalDragbarMouseDown(e) {
	e.preventDefault();
    window.addEventListener("mousemove", horizontalDragbarMouseMove, false);
	window.addEventListener("mouseup", horizontalDragbarMouseUp, false);
};

function horizontalDragbarMouseMove(e) {
	if ((window.innerHeight - e.pageY) > (lowerbarheight + 7)){
		resize_heights(e.pageY + 2);
	} else {
		resize_heights(window.innerHeight - lowerbarheight - 7);
	}
};

function horizontalDragbarMouseUp(e) {
    window.removeEventListener("mousemove", horizontalDragbarMouseMove, false);
};

function reset_panels(){
	resize_widths(window.innerWidth/2);
	resize_heights(window.innerHeight/2);
};