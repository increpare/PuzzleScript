/*
Hello, World! example
June 11, 2015
Copyright (C) 2015 David Martinez
All rights reserved.
This code is the most basic barebones code for writing a program for Arduboy.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
*/

#include <Arduboy2.h>

// make an instance of arduboy used for many functions
Arduboy2 arduboy;


// This function runs once in your game.
// use it for anything that needs to be set only once in your game.
void setup() {
  // initiate arduboy instance
  arduboy.boot();

  // here we set the framerate to 15, we do not need to run at
  // default 60 and it saves us battery life
  arduboy.setFrameRate(15);
}


//0 title screen
//1 message
//2 level
enum State {
	LEVEL,
	TITLE,
	MESSAGE
};

const char CHAR_WIDTH=6;
const char SCREEN_WIDTH=128;
const char SCREEN_HEIGHT=64;

State state=TITLE;

void titleLoop(){
  if (!(arduboy.nextFrame()))
    return;
  arduboy.setCursor(0, 0);
  arduboy.print(F("123456789012345678901234567890"));
}

void messageLoop(){
  if (!(arduboy.nextFrame()))
    return;

  arduboy.display(true);
}

void levelLoop(){
  if (!(arduboy.nextFrame()))
    return;


  arduboy.display(true);
}

// our main game loop, this runs once every cycle/frame.
// this is where our game logic goes.
void loop() {
  // pause render until it's time for the next frame
  switch(state){
  	case LEVEL:
  		levelLoop();
  		break;
  	case TITLE:
  		titleLoop();
  		break;
  	case MESSAGE:
  		messageLoop();
  		break;
  }
}