var soundbarwidth = 440;
var lowerbarheight = 18;
var upperbarheight = 30;
var winwidth = 0;
var winheight = 0;

function resize_widths(verticaldragbarX){
	document.getElementById("leftpanel").style.width = verticaldragbarX + "px";
	document.getElementById("righttophalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("rightbottomhalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("horizontaldragbar").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("verticaldragbar").style.left = verticaldragbarX + "px";
	canvasResize();
	vbarX = verticaldragbarX;
}

function resize_heights(horizontaldragbarY){
	document.getElementById("leftpanel").style.height = (window.innerHeight - upperbarheight) + "px";
	document.getElementById("verticaldragbar").style.height = (window.innerHeight - upperbarheight) + "px";
	document.getElementById("righttophalf").style.height = horizontaldragbarY - upperbarheight + "px";
	document.getElementById("rightbottomhalf").style.top = horizontaldragbarY + 2 + "px";
	document.getElementById("horizontaldragbar").style.top = horizontaldragbarY + "px";
	canvasResize();
	hbarY = horizontaldragbarY;
}

function resize_all(e){
	hdiff = window.innerWidth - winwidth;
	verticaldragbarX = hdiff + parseInt(document.getElementById("verticaldragbar").style.left.replace("px",""));
	if ((verticaldragbarX <= 0)){
		verticaldragbarX = 0;
	} else if ((window.innerWidth - verticaldragbarX) < soundbarwidth){
		verticaldragbarX = window.innerWidth - soundbarwidth;
	};
	resize_widths(verticaldragbarX);
	
	vdiff = window.innerHeight - winheight;
	horizontaldragbarY = vdiff + parseInt(document.getElementById("horizontaldragbar").style.top.replace("px",""));
	if ((horizontaldragbarY <= upperbarheight)){
		horizontaldragbarY = upperbarheight;
	} else if ((window.innerHeight - horizontaldragbarY) < (lowerbarheight)){
		horizontaldragbarY = window.innerHeight - lowerbarheight - 7;
	};
	resize_heights(horizontaldragbarY);
	
	winwidth = window.innerWidth;
	winheight = window.innerHeight;
};

function verticalDragbarMouseDown(e) {
    e.preventDefault();
    window.addEventListener("mousemove", verticalDragbarMouseMove, false);
	window.addEventListener("mouseup", verticalDragbarMouseUp, false);
};

function verticalDragbarMouseMove(e) {
	if (e.pageX <= 0){
		resize_widths(0);
	} else if ((window.innerWidth - e.pageX) > soundbarwidth){
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
	if (e.pageY <= upperbarheight) {
		resize_heights(upperbarheight);
	} else if ((window.innerHeight - e.pageY) > (lowerbarheight + 7)){
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
	winwidth = window.innerWidth;
	winheight = window.innerHeight;
};