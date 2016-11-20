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

const byte PLAYER_MASK = 0b00000100;
const long PLAYER_LAYERMASK = 0b00000000000000000111110000000000;

const long LAYERMASK[] = {
	0b00000000000000000000000000000001,
	0b00000000000000000000000000000010,
	0b00000000000000000000000000011100,
};
const int GLYPH_COUNT = 5;

PROGMEM const byte tiles_b[][8] = {
	{
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b00000000,
		0b00011000,
		0b00111100,
		0b00111100,
		0b00011000,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b00010000,
		0b11111111,
		0b00111111,
		0b00111111,
		0b11111111,
		0b00010000,
		0b00000000,
	},
	{
		0b11111111,
		0b11111111,
		0b11011011,
		0b11111111,
		0b11111111,
		0b11011011,
		0b11111111,
		0b11111111,
	},
	{
		0b11111111,
		0b10000001,
		0b10000001,
		0b10000001,
		0b10000001,
		0b10000001,
		0b10000001,
		0b11111111,
	},
};

PROGMEM const byte tiles_w[][8] = {
	{
		0b11111111,
		0b11111111,
		0b11111111,
		0b11111111,
		0b11111111,
		0b11111111,
		0b11111111,
		0b11111111,
	},
	{
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b00000000,
		0b00100100,
		0b00000000,
		0b00000000,
		0b00100100,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
};

const byte levels[][128] {
	{
		9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		9,9,9,9,1,1,1,9,9,9,9,9,1,1,1,1,
		9,1,3,9,1,1,9,1,1,1,1,1,9,1,1,1,
		9,1,1,9,9,9,1,1,1,1,1,1,1,9,1,1,
		9,19,5,1,1,1,1,19,1,9,1,1,1,1,9,1,
		9,1,1,17,1,1,1,1,1,1,17,1,1,9,1,1,
		9,1,1,9,9,9,1,1,1,1,1,1,1,9,1,1,
		9,9,9,9,1,1,9,9,9,9,9,9,9,1,1,1,
	},
	{
		9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
		1,9,9,9,9,9,9,9,9,9,9,9,1,1,1,1,
		9,1,3,1,1,1,1,1,1,1,1,1,9,1,1,1,
		9,1,1,1,1,1,1,1,1,1,1,1,1,9,1,1,
		9,19,5,1,1,1,1,19,1,9,1,1,1,1,9,1,
		9,1,1,17,1,1,1,1,1,1,17,1,1,9,1,1,
		9,1,1,9,9,9,1,1,1,1,1,1,1,9,1,1,
		9,9,9,9,1,1,9,9,9,9,9,9,9,1,1,1,
	},
};

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
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&8192);
}

bool doesMatch3(byte i){
  byte d = 1;
  byte cellObjects0 = level[i];
  word cellMovements0 = movementMask[i];
  byte cellObjects1 = level[i+16*d];
  word cellMovements1 = movementMask[i+16*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&1024);
}

bool doesMatch4(byte i){
  byte d = 1;
  byte cellObjects1 = level[i];
  word cellMovements1 = movementMask[i];
  byte cellObjects0 = level[i+16*d];
  word cellMovements0 = movementMask[i+16*d];
  return (cellObjects0 & 16) && (cellObjects1&4) && (cellMovements1&2048);
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
  applyRule1(0);
  applyRule2(0);
  applyRule3(0);
  applyRule4(0);
}

void processLateRules(){

}
