#include "Arduboy2.h"
#include "generated.h"

Arduboy2 arduboy;

#define var byte

const byte DIR_UP     = 0b00001;
const byte DIR_DOWN   = 0b00010;
const byte DIR_LEFT   = 0b00100;
const byte DIR_RIGHT  = 0b01000;
const byte DIR_ACTION = 0b10000;

const word ALL_UP = DIR_UP+(DIR_UP<<5)+(DIR_UP<<10);
const word ALL_DOWN = DIR_DOWN+(DIR_DOWN<<5)+(DIR_DOWN<<10);
const word ALL_LEFT = DIR_LEFT+(DIR_LEFT<<5)+(DIR_LEFT<<10);
const word ALL_RIGHT = DIR_RIGHT+(DIR_RIGHT<<5)+(DIR_RIGHT<<10);

byte level[128];
word movementArray[128];
byte rowCellContents[8];
byte colCellContents[16];
byte mapCellContents=0;


void resetState(){
  memcpy(level,levels[0],128*sizeof(byte));
  memset(rowCellContents,0, 8*sizeof(byte));
  memset(colCellContents,0, 16*sizeof(byte));
  mapCellContents=0;
}

void setup() {
  Serial.begin(9600);
  arduboy.boot();
  arduboy.setFrameRate(15);

  resetState();
  render();
}



void moveTick(word mvmt){
  memset(movementArray,0, 128*sizeof(word));
  for (byte j=0;j<8;j++){
    for (byte i=0;i<16;i++){
      byte idx = i+16*j;
      byte p = level[idx];
      if (p&PLAYER_MASK){
        movementArray[idx]= PLAYER_LAYERMASK & mvmt;
      }
    }
  }

  processMovements();  
}

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
    Serial.println(F("collision"));
    return false;
  }

  Serial.println(F("no collision"));
  Serial.println(sourceMask,BIN);
  Serial.println(targetMask,BIN);
  Serial.println(layerMask,BIN);

  word movingEntities = sourceMask & layerMask;
  Serial.print(F("moving entities = "));
  Serial.println(movingEntities,BIN);

  byte targetbefore = level[targetIndex];

  level[targetIndex] |= movingEntities;
  level[positionIndex] &= ~layerMask;
  
  byte targetafter = level[targetIndex];

  Serial.println(F("before vs after "));
  Serial.println(targetbefore,BIN);
  Serial.println(targetafter,BIN);

  byte colIndex = targetIndex%16;
  byte rowIndex = targetIndex/16;

  colCellContents[colIndex] |= movingEntities;
  rowCellContents[rowIndex] |= movingEntities;
  mapCellContents |= layerMask;
  Serial.println(F("all good"));
  return true;
}

byte LAYERCOUNT_MAX=3;
byte repositionEntitiesAtCell(byte positionIndex) {
    word movementMask = movementArray[positionIndex];


    if (movementMask==0){
        return false;
    }

    Serial.print(F("found movement at "));
    Serial.println(positionIndex);

    bool moved=false;
    for (var layer=0;layer<LAYERCOUNT_MAX;layer++){
      word layerMovement = (movementMask>>(5*layer))&0b11111;
      Serial.print(movementMask,BIN);
      Serial.print(F(" >> "));
      Serial.print(layer);
      Serial.print(F("& 0b11111 = "));
      Serial.println(layerMovement,BIN);
      
      if (layerMovement!=0){
        Serial.println(F("trying to reposition"));
        bool thismoved = repositionEntitiesOnLayer(positionIndex,layer,layerMovement);
        if(thismoved){
          Serial.print(F("movement on "));
          Serial.println(movementMask,BIN);
          movementMask = movementMask & (~(layerMovement));
          moved=true;
        }
      }
    }

    movementArray[positionIndex]=movementMask;    
    return moved;
}

void processMovements(){
  Serial.println(F("processMovements"));
  var moved=true;
  while(moved){
    moved=false;
    for (var i=0;i<128;i++) {
      moved = repositionEntitiesAtCell(i) || moved;
    }
  } 
}


void render(){
  arduboy.clear();

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

  arduboy.display();
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

// our main game loop, this runs once every cycle/frame.
// this is where our game logic goes.
void loop() {
  // pause render until it's time for the next frame
  if (!(arduboy.nextFrame()))
    return;


  arduboy.pollButtons();
  
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
  render();
}
