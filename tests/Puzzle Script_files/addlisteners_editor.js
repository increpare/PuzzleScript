
for (var i=0;i<10;i++) {
	var idname = "newsound"+i;
	var el = document.getElementById(idname);
    el.addEventListener("click", (function(n){return function(){return newSound(n);};})(i), false);
}

var soundButtonPress = document.getElementById("soundButtonPress");
soundButtonPress.addEventListener("click", buttonPress, false);


var runClickLink = document.getElementById("runClickLink");
runClickLink.addEventListener("click", runClick, false);
var rebuildClickLink = document.getElementById("rebuildClickLink");
rebuildClickLink.addEventListener("click", rebuildClick, false);
