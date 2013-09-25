

    onmousemove="mouseMove(event)" 
    onmouseout="mouseOut()"

var el = document.getElementById("gameCanvas");
if (el.addEventListener) {
    el.addEventListener("mousemove", mouseMove, false);
    el.addEventListener("mouseout", mouseOut, false);
} else {
    el.attachEvent('onmousemove', mouseMove);
    el.attachEvent('onmouseout', mouseOut);
}  