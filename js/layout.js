var soundbarwidth = 440;

function resize_righthalf(event, ui){
	if ($("BODY").width() - ui.size["width"] <= soundbarwidth){
		return;
	};
	
	newwidth = $("BODY").width() - ui.size["width"];
	$( ".righttophalf").width(newwidth);
	$( ".rightbottomhalf" ).width(newwidth);
	canvasResize();
};

function resize_rightbottom(event, ui){
	bottomhalfheight = $("BODY").height() - (ui.position["top"] + ui.size["height"]);
	$( ".rightbottomhalf" ).height(bottomhalfheight);
	
	canvasResize();
}

function resize_all(){
	maxtophalfheight = ($("BODY").height()-($(".uppertoolbar").height() + $(".lowertoolbar").height() + 2));

	//reset maximums drag extents
	$( ".righttophalf" ).resizable( "option", "maxHeight", maxtophalfheight);
	$(".leftpanel").resizable( "option", "maxWidth", ($("BODY").width() - soundbarwidth));
	
	//resize panel heights
	if ($(".righttophalf").height() > maxtophalfheight){
		$(".righttophalf").height(maxtophalfheight);
		bottomhalfheight = $("BODY").height() - (maxtophalfheight);
	} else {
		bottomhalfheight = $("BODY").height() - ($(".uppertoolbar").height() + $(".righttophalf").height());
	};
	$( ".rightbottomhalf" ).height(bottomhalfheight);
	$(".leftpanel").height($("BODY").height() - $(".uppertoolbar").height());
	
	//resize panel widths
	if ($("BODY").width() - $(".leftpanel").width() < soundbarwidth){
		$(".leftpanel").width($("BODY").width() - soundbarwidth);
	};
	RHSwidth = $("BODY").width() - $(".leftpanel").width();
	$( ".righttophalf").width(RHSwidth);
	$( ".rightbottomhalf" ).width(RHSwidth);
	
	canvasResize();
};

$( ".leftpanel" ).resizable({
	handles: "e",
	resize: resize_righthalf,
	stop: resize_righthalf,
	maxWidth: ($("BODY").width() - soundbarwidth)
});

$( ".righttophalf" ).resizable({
	handles: "s",
	resize: resize_rightbottom,
	stop: resize_rightbottom,
	maxHeight: ($("BODY").height()-($(".uppertoolbar").height() + $(".lowertoolbar").height() + 2))
} );

window.onresize = resize_all;

window.onload = function(event){
	$(".leftpanel").height($("BODY").height() - $(".uppertoolbar").height());
	$(".righttophalf").height($("BODY").height()/2 - $(".uppertoolbar").height());
}