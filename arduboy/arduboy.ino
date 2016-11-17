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
word movementMask[128];
byte rowCellContents[8];
byte colCellContents[16];
byte mapCellContents=0;


void resetState(){
  memcpy(level,levels[0],128*sizeof(byte));
  memset(rowCellContents,0, 8*sizeof(byte));
  memset(colCellContents,0, 16*sizeof(byte));
  mapCellContents=0;
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

void setup() {
  Serial.begin(9600);
  arduboy.boot();
  arduboy.setFrameRate(15);

  resetState();
  render();
}

bool doesMatch1(byte i){
  byte d = 1;
  byte cellObjects0 = level[i];
  word cellMovements0 = movementMask[i];
  byte cellObjects1 = level[i+1*d];
  word cellMovements1 = movementMask[i+1*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&4096);
}
bool doesMatch2(byte i){
  byte d = 1;
  byte cellObjects1 = level[i];
  word cellMovements1 = movementMask[i];
  byte cellObjects0 = level[i+1*d];
  word cellMovements0 = movementMask[i+1*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&0b10000000000000);
}
bool doesMatch3(byte i){
  byte d = 1;
  byte cellObjects0 = level[i];
  word cellMovements0 = movementMask[i];
  byte cellObjects1 = level[i+16*d];
  word cellMovements1 = movementMask[i+16*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&0b10000000000);
}
bool doesMatch4(byte i){
  byte d = 1;
  byte cellObjects1 = level[i];
  word cellMovements1 = movementMask[i];
  byte cellObjects0 = level[i+16*d];
  word cellMovements0 = movementMask[i+16*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&0b100000000000);
}

//horizontal rule
bool applyRule1(byte r){ 
  //match code
  for (byte j=0;j<8;j++){
    //check if row has relevant parts
    for (byte i=0;i<16-1;i++){  
      byte idx = i+16*j;
      if (doesMatch1(idx)){
        movementMask[idx]|=DIR_LEFT<<(5*2);
      }
    }
  }
}

bool applyRule2(byte r){ 
  //match code
  for (byte j=0;j<8;j++){
    //check if row has relevant parts
    for (byte i=0;i<16-1;i++){  
      byte idx = i+16*j;
      if (doesMatch2(idx)){
        movementMask[idx+1]|=DIR_RIGHT<<(5*2);
      }
    }
  }
}


bool applyRule3(byte r){ 
  //match code
  for (byte j=0;j<7;j++){
    //check if row has relevant parts
    for (byte i=0;i<16;i++){  
      byte idx = i+16*j;
      if (doesMatch3(idx)){
        movementMask[idx]|=DIR_UP<<(5*2);
      }
    }
  }
}

bool applyRule4(byte r){ 
  //match code
  for (byte j=0;j<7;j++){
    //check if row has relevant parts
    for (byte i=0;i<16;i++){  
      byte idx = i+16*j;
      if (doesMatch4(idx)){
        movementMask[idx+16]|=DIR_DOWN<<(5*2);
      }
    }
  }
}

void processRules(){
  Serial.println(F("Applying rules"));
  applyRule1(0);
  applyRule2(0);
  applyRule3(0);
  applyRule4(0);

  // for (byte i=0;i<RULE_GROUP_COUNT;i++){
  //   int** rg = RULE_GROUP[i];
  //   bool applied=true;
  //   while (appleid){
  //     applied=false;      
  //     for (byte j=0;j<RULE_GROUP_LENGTH[i];j++){
  //       //int* r = rg[j];
  //       //applied |= applyRule(r)
  //       applyRule(0)
  //     }
  //   }
  // }
}

void processLateRules(){

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

  processRules();
  processMovements();  
  processLateRules();
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
  Serial.println(F("processMovements"));
  var moved=true;
  while(moved){
    moved=false;
    for (var i=0;i<128;i++) {
      moved = repositionEntitiesAtCell(i) || moved;
    }
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
