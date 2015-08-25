var soundbarwidth = 100;
var lowerbarheight = document.getElementById("soundbar").clientHeight;
var upperbarheight = document.getElementById("uppertoolbar").clientHeight;
var winwidth = window.innerWidth;
var winheight = window.innerHeight;
var verticaldragbarWidth = document.getElementById("verticaldragbar").clientWidth;
var horizontaldragbarHeight = document.getElementById("horizontaldragbar").clientHeight;
var minimumDimension = 100;

function resize_widths(verticaldragbarX){
	document.getElementById("leftpanel").style.width = verticaldragbarX + "px";
	document.getElementById("righttophalf").style.left = verticaldragbarX + verticaldragbarWidth + "px";
	document.getElementById("rightbottomhalf").style.left = verticaldragbarX + verticaldragbarWidth + "px";
	document.getElementById("horizontaldragbar").style.left = verticaldragbarX + verticaldragbarWidth + "px";
	document.getElementById("verticaldragbar").style.left = verticaldragbarX + "px";
	if (window.onresize!=null){
		window.onresize();
	}
}

function resize_heights(horizontaldragbarY){
	document.getElementById("leftpanel").style.height = (window.innerHeight - upperbarheight) + "px";
	document.getElementById("verticaldragbar").style.height = (window.innerHeight - upperbarheight) + "px";
	
	document.getElementById("righttophalf").style.height = horizontaldragbarY - upperbarheight + "px";
	document.getElementById("rightbottomhalf").style.top = horizontaldragbarY + horizontaldragbarHeight + "px";
	document.getElementById("horizontaldragbar").style.top = horizontaldragbarY + "px";
	if (window.onresize!=null){
		window.onresize();
	}
}

function resize_all(e){
	smallmovelimit = 100;
	
	hdiff = window.innerWidth - winwidth;
	verticaldragbarX = parseInt(document.getElementById("verticaldragbar").style.left.replace("px",""));
	
	if(hdiff > -smallmovelimit && hdiff < smallmovelimit){
		verticaldragbarX += hdiff;
	} else {
		verticaldragbarX *= window.innerWidth/winwidth;
	};
	
	if ((verticaldragbarX <= minimumDimension)){
		verticaldragbarX = minimumDimension;
	} else if ((window.innerWidth - verticaldragbarX) < soundbarwidth){
		verticaldragbarX = window.innerWidth - soundbarwidth;
	};
	resize_widths(verticaldragbarX);
	
	
	
	horizontaldragbarY = parseInt(document.getElementById("horizontaldragbar").style.top.replace("px",""));
	vdiff = window.innerHeight - winheight;
	
	if(vdiff > -smallmovelimit && vdiff < smallmovelimit){
		horizontaldragbarY += vdiff;
	} else {
		horizontaldragbarY *= window.innerHeight/winheight;
	};
	
	if ((horizontaldragbarY <= upperbarheight + minimumDimension)){
		horizontaldragbarY = upperbarheight + minimumDimension;
	} else if ((window.innerHeight - horizontaldragbarY) < (lowerbarheight + minimumDimension)){
		horizontaldragbarY = window.innerHeight - (lowerbarheight + minimumDimension + 5);
	};
	resize_heights(horizontaldragbarY);
	
	winwidth = window.innerWidth;
	winheight = window.innerHeight;
};

function verticalDragbarMouseDown(e) {
	e.preventDefault();
	document.body.style.cursor = "col-resize";
	window.addEventListener("mousemove", verticalDragbarMouseMove, false);
	window.addEventListener("mouseup", verticalDragbarMouseUp, false);
};

function verticalDragbarMouseMove(e) {
	if (e.pageX <= minimumDimension){
		resize_widths(minimumDimension);
	} else if ((window.innerWidth - e.pageX) > soundbarwidth){
		resize_widths(e.pageX - 1);
	} else {
		resize_widths(window.innerWidth - soundbarwidth);
	};
};

function verticalDragbarMouseUp(e) {
	document.body.style.cursor = "";
	window.removeEventListener("mousemove", verticalDragbarMouseMove, false);
};

function horizontalDragbarMouseDown(e) {
	e.preventDefault();
	document.body.style.cursor = "row-resize";
	window.addEventListener("mousemove", horizontalDragbarMouseMove, false);
	window.addEventListener("mouseup", horizontalDragbarMouseUp, false);
};

function horizontalDragbarMouseMove(e) {
	if (e.pageY <= (upperbarheight + minimumDimension)) {
		resize_heights(upperbarheight + minimumDimension);
	} else if ((window.innerHeight - e.pageY) > (lowerbarheight + minimumDimension)){
		resize_heights(e.pageY - 1);
	} else {
		resize_heights(window.innerHeight - lowerbarheight - minimumDimension);
	}
};

function horizontalDragbarMouseUp(e) {
	document.body.style.cursor = "";
	window.removeEventListener("mousemove", horizontalDragbarMouseMove, false);
};

function reset_panels(){
	resize_widths(Math.floor(window.innerWidth/2));
	resize_heights(Math.floor(window.innerHeight/2));
	winwidth = window.innerWidth;
	winheight = window.innerHeight;
};