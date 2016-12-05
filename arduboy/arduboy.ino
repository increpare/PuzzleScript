#include "Arduboy2.h"
#include <ArduboyTones.h>

Arduboy2 arduboy;
ArduboyTones sound(arduboy.audio.enabled);

#include "generated.h"

#define var byte

byte curLevel=0;

void resetState(){
  for (byte i=0;i<128;i++){
    level[i]=pgm_read_byte_near(levels[curLevel]+i);
  }
  memset(rowCellContents,0, 8*sizeof(byte));
  memset(colCellContents,0, 16*sizeof(byte));
  mapCellContents=0;
  //generate row/cell contents properly
}

void drawLevel(){
  for (byte j=0;j<8;j++){
    for (byte i=0;i<16;i++){
      byte idx = i+16*j;
      byte dat = level[idx];
      for (char g=0;g<GLYPH_COUNT;g++){
        char m = 1<<g;
        if (dat&m){
            arduboy.drawBitmap(i*8,j*8, tiles_b[g], 8,8, 0);
            arduboy.drawBitmap(i*8,j*8, tiles_w[g], 8,8, 1);
        }
      }
    }
  }

  arduboy.display(true);
}

void setup() {
  Serial.begin(9600);
  arduboy.boot();
  arduboy.systemButtons();
  arduboy.audio.begin();
  arduboy.setFrameRate(15);
  resetState();
  drawTitle();
}

void processRules(){
  applyRule0_0();
  applyRule0_1();
  applyRule0_2();
  applyRule0_3();
}

void processLateRules(){
}


void preserveUndoState(){
  memcpy(undoState,level,128);
}

void doUndo(){
  memcpy(level,undoState,128);
}

void doReset(){
  resetState();
}

void DoCompute(){
  preserveUndoState();
  processRules();
  processMovements();  
  processLateRules();
  checkWin();
}

void moveTick(word mvmt){
  memset(movementMask,0, 128*sizeof(word));
  for (byte j=0;j<8;j++){
    for (byte i=0;i<16;i++){
      byte idx = i+16*j;
      byte p = level[idx];
      if (p&PLAYER_MASK){
        movementMask[idx]= PLAYER_LAYERMASK & mvmt;
      }
    }
  }
  DoCompute();  
}
byte sfx_movedObjects;

bool repositionEntitiesOnLayer(byte positionIndex,byte layer,byte dirMask) 
{
  byte px = positionIndex%16;
  byte py = positionIndex/16;

  const byte maxx = 16-1;
  const byte maxy = 8-1;

  char dx=0;
  char dy=0;

  switch(dirMask){
    case 1:
      if (py==0){
        return false;
      }
      dy=-1;
      break;
    case 2:
      if (py==7){
        return false;
      }
      dy=1;
      break;
    case 4:
      if (px==0){
        return false;
      }
      dx=-1;
      break;
    case 8:
      if (px==15){
        return false;
      }
      dx=1;
      break;
    default:
      return false;
  }

  byte targetIndex = positionIndex + dx + dy*16;


  word layerMask = LAYERMASK[layer];
  word targetMask = level[targetIndex];
  word sourceMask = level[positionIndex];

  if (targetMask&layerMask){
    return false;
  }

  word movingEntities = sourceMask & layerMask;

  byte targetbefore = level[targetIndex];

  level[targetIndex] |= movingEntities;
  level[positionIndex] &= ~layerMask;
  
  byte targetafter = level[targetIndex];


  byte colIndex = targetIndex%16;
  byte rowIndex = targetIndex/16;

  colCellContents[colIndex] |= movingEntities;
  rowCellContents[rowIndex] |= movingEntities;
  sfx_movedObjects|=movingEntities;

  mapCellContents |= layerMask;
  return true;
}

byte LAYERCOUNT_MAX=3;
byte repositionEntitiesAtCell(byte positionIndex) {
    word movMask = movementMask[positionIndex];


    if (movMask==0){
        return false;
    }


    bool moved=false;
    for (var layer=0;layer<LAYERCOUNT_MAX;layer++){
      word layerMovement = (movMask>>(5*layer))&0b11111;
      
      if (layerMovement!=0){
        bool thismoved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement);
        if(thismoved){
          movMask = movMask & (~(layerMovement));
          moved=true;
        }
      }
    }

    movementMask[positionIndex]=movMask;    
    return moved;
}


void processMovements(){
  sfx_movedObjects=0;

  var moved=true;
  while(moved){
    moved=false;
    for (var i=0;i<128;i++) {
      moved = repositionEntitiesAtCell(i) || moved;
    }
  } 

  if (sfx_movedObjects&0b1000){
      sound.tone(100, 100);
  } 
}


bool nothingHappened(){
  return !(
          arduboy.justPressed(UP_BUTTON) ||
          arduboy.justPressed(DOWN_BUTTON) ||
          arduboy.justPressed(LEFT_BUTTON) ||
          arduboy.justPressed(RIGHT_BUTTON) ||
          arduboy.justPressed(A_BUTTON) ||
          arduboy.justPressed(B_BUTTON) //||
          // arduboy.justReleased(UP_BUTTON) ||
          // arduboy.justReleased(DOWN_BUTTON) ||
          // arduboy.justReleased(LEFT_BUTTON) ||
          // arduboy.justReleased(RIGHT_BUTTON) ||
          // arduboy.justReleased(A_BUTTON) ||
          // arduboy.justReleased(B_BUTTON)
          );
}






void titleLoop(){
  if (nothingHappened()){
    return;
  }

  if (titleSelection<2){
    if (arduboy.justPressed(UP_BUTTON)){
      titleSelection=0;
    }
    if (arduboy.justPressed(DOWN_BUTTON)){
      titleSelection=1;
    }
  }

  //FOR TESTING PURPOSES
  if (arduboy.justPressed(A_BUTTON)){
    if (titleSelection<2){
      titleSelection=2;
    } else {
      titleSelection=0;
    }
  }
  if (arduboy.justPressed(B_BUTTON)){
    state=LEVEL;
    curLevel=0;
    resetState();
    drawLevel();
    return;
  }
 
  drawTitle();
  undoState[0]=0;

}

void messageLoop(){
  if (nothingHappened()){
    return;
  }

  arduboy.display(true);
}

void levelLoop(){
  if (waiting){
    unsigned long now = millis();
    unsigned long diff = now-waitfrom;
    if (diff>500){
      waiting=false;
      if (curLevel<1){
        curLevel++;
        resetState();
        drawLevel();
      } else {
        state=TITLE;
        drawTitle();
      }
    }
    return;
  }

  if (nothingHappened()){
    return;
  }
  //resetState();

  if (arduboy.justPressed(UP_BUTTON)){              
      moveTick(ALL_UP);
  }
  if (arduboy.justPressed(DOWN_BUTTON)){   
      moveTick(ALL_DOWN);
  } 
  if (arduboy.justPressed(RIGHT_BUTTON)){ 
      moveTick(ALL_RIGHT);
  }
  if (arduboy.justPressed(LEFT_BUTTON)){
      moveTick(ALL_LEFT);
  }
  if (arduboy.justPressed(A_BUTTON)){         
    if (arduboy.pressed(B_BUTTON)){
      doReset();
    } else {
      doUndo();
    }
  }
  if (arduboy.justPressed(B_BUTTON)){
    if (arduboy.pressed(A_BUTTON)){
      doReset();
    } else {
      moveTick(ALL_ACTION);
    }
  }
  drawLevel();
}

// our main game loop, this runs once every cycle/frame.
// this is where our game logic goes.
void loop() {
  // pause render until it's time for the next frame
  if (!(arduboy.nextFrame()))
    return;
  arduboy.pollButtons();

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