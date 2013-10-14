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
        handler = new Mobile.GestureHandler();
        handler.bindEvents();

        Mobile.gesturesEnabled = true;
    }
};

window.Mobile.GestureHandler = function () {
    this.initialize.apply(this, arguments);
};

(function (proto) {
    'use strict';

    var SWIPE_THRESHOLD = 10;
    var SWIPE_DISTANCE = 50;
    var SWIPE_TIMEOUT = 1000;
    var CODE = {
        action: 88, // x
        left:   37, // left arrow
        right:  39, // right arrow
        up:     38, // up arrow
        down:   40  // down arrow
    }

    proto.initialize = function () {
        this.firstPos = { x: 0, y: 0 };
    };

    proto.bindEvents = function () {
        window.addEventListener('touchstart', this.onTouchStart.bind(this));
        window.addEventListener('touchend', this.onTouchEnd.bind(this));
        window.addEventListener('touchmove', this.onTouchMove.bind(this));
    };

    proto.onTouchStart = function (event) {
        if (this.isTouching) {
            return;
        }

        this.isTouching = true;

        this.mayBeSwiping = true;
        this.gestured = false;

        this.swipeDirection = undefined;
        this.swipeDistance = 0;
        this.startTime = new Date().getTime();

        this.firstPos.x = event.touches[0].clientX;
        this.firstPos.y = event.touches[0].clientY;
    };

    proto.onTouchEnd = function (event) {
        if (!this.gestured) {
            // Was this a single finger tap?
            if (event.touches.length === 0) {
                this.handleTap();
            }
        }

        // Last finger to be removed from the screen.
        if (event.touches.length === 0) {
            this.isTouching = false;
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
            this.swipeDistance >= SWIPE_DISTANCE) {
            isSuccessful = true;
        }

        return isSuccessful;
    };

    // Examine the current state to see what direction they're swiping and
    // if the gesture can still be considered a swipe.
    proto.swipeStep = function (event) {
        var currentPos, distance, currentTime;

        if (!this.mayBeSwiping) {
            return;
        }

        currentPos = {
            x: event.touches[0].clientX,
            y: event.touches[0].clientY
        };
        currentTime = new Date().getTime();

        this.swipeDistance = this.cardinalDistance(this.firstPos, currentPos);
        if (!this.swipeDirection) {
            // We've swiped far enough to decide what direction we're swiping in.
            if (this.swipeDistance > SWIPE_THRESHOLD) {
                this.swipeDirection = this.dominantDirection(this.firstPos, currentPos);
            }
        } else if (distance < SWIPE_DISTANCE) {
            direction = this.dominantDirection(this.firstPos, currentPos);
            // Cancel the swipe if the direction changes.
            if (direction !== this.swipeDirection) {
                this.mayBeSwiping = false;
            }
        } else if (currentTime - this.startTime > SWIPE_TIMEOUT) {
            // Cancel the swipe if they took too long to finish.
            this.mayBeSwiping = false;
        }
    };

    proto.cardinalDistance = function (firstPos, currentPos) {
        var xDist, yDist;

        xDist = Math.abs(firstPos.x - currentPos.x);
        yDist = Math.abs(firstPos.y - currentPos.y);

        return Math.max(xDist, yDist);
    };

    proto.dominantDirection = function (firstPos, currentPos) {
        var dx, dy;
        var dominantAxis, dominantDirection;

        dx = currentPos.x - firstPos.x;
        dy = currentPos.y - firstPos.y;

        dominantAxis = 'x';
        if (Math.abs(dy) > Math.abs(dx)) {
            dominantAxis = 'y';
        }

        if (dominantAxis === 'x') {
            if (dx > 0) {
                dominantDirection = 'right';
            } else {
                dominantDirection = 'left';
            }
        } else {
            if (dy > 0) {
                dominantDirection = 'down';
            } else {
                dominantDirection = 'up';
            }
        }

        return dominantDirection;
    };

    proto.handleSwipe = function (direction) {
        this.emitKeydown(this.swipeDirection);
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

