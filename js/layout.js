
function resize_righthalf(event, ui){
	newwidth = $("BODY").width() - ui.size["width"];
	bottomhalfheight = $("BODY").height() - (ui.position["top"] + ui.size["height"]);
	$( ".righttophalf").width(newwidth);
	$( ".rightbottomhalf" ).width(newwidth);
	canvasResize();
};

function resize_rightbottom(event, ui){
	bottomhalfheight = $("BODY").height() - (ui.position["top"] + ui.size["height"]);
	$( ".rightbottomhalf" ).height(bottomhalfheight);
	$( ".leftpanel" ).width( $("BODY").width() - ui.size["width"]);
	canvasResize();
}

function resize_all(){
	newwidth = $("BODY").width() - $(".leftpanel").width();
	$( ".righttophalf").width(newwidth);
	$( ".rightbottomhalf" ).width(newwidth);
	canvasResize();
	
	bottomhalfheight = $("BODY").height() - ($(".righttophalf").position()["top"] + $(".righttophalf").height());
	$( ".rightbottomhalf" ).height(bottomhalfheight);
};

$( ".leftpanel" ).resizable({
	handles: "e",
	resize: resize_righthalf,
	stop: resize_righthalf
});

$( ".righttophalf" ).resizable({
	handles: "s",
	resize: resize_rightbottom,
	stop: resize_rightbottom
} );

window.onresize = resize_all;

window.onload = function(event){
	$(".leftpanel").height($("BODY").height() - $(".uppertoolbar").height());
	$(".righttophalf").height($("BODY").height()/2 - $(".uppertoolbar").height());
}