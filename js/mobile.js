/*
 * Add gesture support for mobile devices.
 */

window.Mobile = {};

Mobile.hasTouch = function() {
    return document.documentElement &&
        document.documentElement.hasOwnProperty('ontouchstart');
};

Mobile.enable = function () {
    var handler;
    if (Mobile.hasTouch() && !Mobile.gesturesEnabled) {
        handler = new Mobile.
            GestureHandler();
        handler.bindEvents();

        Mobile.gesturesEnabled = true;
    }
};

window.Mobile.GestureHandler = function () {
    this.initialize.apply(this, arguments);
};

(function (proto) {
    var SWIPE_DISTANCE = 50;
    var CODE = {
        action: 88 // x
    }

    proto.initialize = function () {
        this.firstPos = { x: 0, y: 0 };
        this.lastPos = { x: 0, y: 0 };
    };

    proto.bindEvents = function () {
        window.addEventListener('touchstart', this.onTouchStart.bind(this));
        window.addEventListener('touchend', this.onTouchEnd.bind(this));
        window.addEventListener('touchmove', this.onTouchMove.bind(this));
    };

    proto.onTouchStart = function (event) {
        this.mayBeSwiping = true;
        this.gestured = false;

        this.swipeDirection = undefined;
        this.swipeDistance = 0;
        this.startTime = new Date().getTime();

        this.lastPos.x = event.touches[0].clientX;
        this.lastPos.y = event.touches[0].clientY;
    };

    proto.onTouchEnd = function (event) {
        if (!this.gestured) {
            // Was this a single finger tap?
            if (event.touches.length === 0) {
                this.handleTap();
            }
        }
    };

    proto.onTouchMove = function (event) {
        if (this.isSuccessfulSwipe()) {
            this.handleSwipe(this.swipeDirection);
            this.gestured = true;
            this.mayBeSwiping = false;
        } else if (this.mayBeSwiping) {
            this.swipeStep(event);
        }

        event.preventDefault();
    };

    proto.isSuccessfulSwipe = function () {
        var isSuccessful;

        if (this.mayBeSwiping &&
            this.swipeDirection !== undefined &&
            this.swipeDistance > SWIPE_DISTANCE) {
            isSuccessful = true;
        }

        return isSuccessful;
    };

    proto.swipeStep = function (event) {
        var pos, distance;

        pos = {
            x: event.touches[0].clientX,
            y: event.touches[0].clientY
        };
    };

    proto.cardinalDistance = function (origin, target) {
        var xDist, yDist;

        xDist = origin.x - target.x;
        yDist = origin.y - target.y;

        return Math.min(xDist, yDist);
    };

    proto.handleSwipe = function (direction) {

    };

    proto.handleTap = function () {
        this.emitKeydown('action');
    };

    proto.emitKeydown = function (input) {
        var event;

        event = { keyCode: CODE[input] };

        // Press, then release key.
        onKeyDown(event);
        onKeyUp(event);
    };
}(window.Mobile.GestureHandler.prototype));

