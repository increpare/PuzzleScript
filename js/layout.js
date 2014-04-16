var soundbarwidth = 440;
var lowerbarheight = 18;
var upperbarheight = 30;

function verticalDragbarMouseDown(e) {
    e.preventDefault();
    window.addEventListener("mousemove", verticalDragbarMouseMove, false);
	window.addEventListener("mouseup", verticalDragbarMouseUp, false);
};

function verticalDragbarMouseMove(e) {
	if ((window.innerWidth - e.pageX) > soundbarwidth){
		document.getElementById("leftpanel").style.width = e.pageX + 2 + "px";
		document.getElementById("righttophalf").style.left = e.pageX + 5 + "px";
		document.getElementById("rightbottomhalf").style.left = e.pageX + 5 + "px";
		document.getElementById("horizontaldragbar").style.left = e.pageX + 5 + "px";
		document.getElementById("verticaldragbar").style.left = e.pageX + 2 + "px";
		canvasResize();
	};
};

function verticalDragbarMouseUp(e) {
    //document.getElementById("clickevent").innerHTML = 'in another mouseUp event' + i++;
    window.removeEventListener("mousemove", verticalDragbarMouseMove, false);
};

function horizontalDragbarMouseDown(e) {
	e.preventDefault();
    window.addEventListener("mousemove", horizontalDragbarMouseMove, false);
	window.addEventListener("mouseup", horizontalDragbarMouseUp, false);
};

function horizontalDragbarMouseMove(e) {
	if ((window.innerHeight - e.pageY) > (lowerbarheight + 8)){
		document.getElementById("righttophalf").style.height = e.pageY + 2 - upperbarheight + "px";
		document.getElementById("rightbottomhalf").style.top = e.pageY + 5 + "px";
		document.getElementById("horizontaldragbar").style.top = e.pageY + 2 + "px";
		canvasResize();
	}
};

function horizontalDragbarMouseUp(e) {
    //document.getElementById("clickevent").innerHTML = 'in another mouseUp event' + i++;
    window.removeEventListener("mousemove", horizontalDragbarMouseMove, false);
};

function resize_all(){
	document.getElementById("leftpanel").style.height = (window.innerHeight - 30) + "px";
	document.getElementById("verticaldragbar").style.height = (window.innerHeight - 30) + "px";

	verticaldragbarX = parseInt(document.getElementById("verticaldragbar").style.left.replace("px",""));
	
	if ((window.innerWidth - verticaldragbarX) < soundbarwidth){
		verticaldragbarX = window.innerWidth - soundbarwidth - 30;
	};
	
	document.getElementById("leftpanel").style.width = verticaldragbarX + "px";
	document.getElementById("righttophalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("rightbottomhalf").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("horizontaldragbar").style.left = verticaldragbarX + 3 + "px";
	document.getElementById("verticaldragbar").style.left = verticaldragbarX + "px";
	
	horizontaldragbarY = parseInt(document.getElementById("horizontaldragbar").style.top.replace("px",""));
	if ((window.innerHeight - horizontaldragbarY) < (lowerbarheight)){
		horizontaldragbarY = window.innerHeight - lowerbarheight - 8;
	};
	
	document.getElementById("righttophalf").style.height = horizontaldragbarY - upperbarheight + "px";
	document.getElementById("rightbottomhalf").style.top = horizontaldragbarY + 3 + "px";
	document.getElementById("horizontaldragbar").style.top = horizontaldragbarY + "px";
	canvasResize();
};

function adjust_panels(){
	document.getElementById("leftpanel").style.height = (window.innerHeight - 30) + "px";
	document.getElementById("verticaldragbar").style.height = (window.innerHeight - 30) + "px";
	document.getElementById("righttophalf").style.left = (window.innerWidth/2 + 3) + "px";
	document.getElementById("rightbottomhalf").style.left = (window.innerWidth/2 + 3) + "px";
	document.getElementById("rightbottomhalf").style.top = (window.innerHeight/2 + 3) + "px";
	document.getElementById("horizontaldragbar").style.left = (window.innerWidth/2 + 3) + "px";
	
	//need to have these in px in case resize_all is called
	document.getElementById("verticaldragbar").style.left = (window.innerWidth/2) + "px";
	document.getElementById("horizontaldragbar").style.top = (window.innerHeight/2) + "px";
};