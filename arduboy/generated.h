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

bool applyRule0_0_0(){ 
  for (byte y=0;y<7;y++){
    for (byte x=0;x<16;x++){  
      byte i = x+16*y;
      if ((level[i] & 16) && (level[i+16] & 4) && (movementMask[i+16] & 1024)){
		level[i] = (level[i]&4294967267)|16;
		movementMask[i] = (movementMask[i]&4294967295)|1024;
		level[i+16] = (level[i+16]&4294967267)|4;
		movementMask[i+16] = (movementMask[i+16]&4294967295)|1024;

      }
    }
  }
}
bool applyRule0_1_0(){ 
  for (byte y=0;y<7;y++){
    for (byte x=0;x<16;x++){  
      byte i = x+16*y;
      if ((level[i] & 4) && (movementMask[i] & 2048) && (level[i+16] & 16)){
		level[i] = (level[i]&4294967267)|4;
		movementMask[i] = (movementMask[i]&4294967295)|2048;
		level[i+16] = (level[i+16]&4294967267)|16;
		movementMask[i+16] = (movementMask[i+16]&4294967295)|2048;

      }
    }
  }
}
bool applyRule0_2_0(){ 
  for (byte y=0;y<8;y++){
    for (byte x=0;x<15;x++){  
      byte i = x+16*y;
      if ((level[i] & 16) && (level[i+1] & 4) && (movementMask[i+1] & 4096)){
		level[i] = (level[i]&4294967267)|16;
		movementMask[i] = (movementMask[i]&4294967295)|4096;
		level[i+1] = (level[i+1]&4294967267)|4;
		movementMask[i+1] = (movementMask[i+1]&4294967295)|4096;

      }
    }
  }
}
bool applyRule0_3_0(){ 
  for (byte y=0;y<8;y++){
    for (byte x=0;x<15;x++){  
      byte i = x+16*y;
      if ((level[i] & 4) && (movementMask[i] & 8192) && (level[i+1] & 16)){
		level[i] = (level[i]&4294967267)|4;
		movementMask[i] = (movementMask[i]&4294967295)|8192;
		level[i+1] = (level[i+1]&4294967267)|16;
		movementMask[i+1] = (movementMask[i+1]&4294967295)|8192;

      }
    }
  }
}
bool applyRule1_0_0(){ 
  for (byte y=0;y<6;y++){
    for (byte x=0;x<16;x++){  
      byte i = x+16*y;
      if ((level[i] & 16) && (level[i+16] & 16) && (level[i+32] & 16)){
		level[i] = (level[i]&4294967267)|0;
		movementMask[i] = (movementMask[i]&4294967295)|0;
		level[i+16] = (level[i+16]&4294967267)|0;
		movementMask[i+16] = (movementMask[i+16]&4294967295)|0;
		level[i+32] = (level[i+32]&4294967267)|0;
		movementMask[i+32] = (movementMask[i+32]&4294967295)|0;

      }
    }
  }
}
bool applyRule1_1_0(){ 
  for (byte y=0;y<8;y++){
    for (byte x=0;x<14;x++){  
      byte i = x+16*y;
      if ((level[i] & 16) && (level[i+1] & 16) && (level[i+2] & 16)){
		level[i] = (level[i]&4294967267)|0;
		movementMask[i] = (movementMask[i]&4294967295)|0;
		level[i+1] = (level[i+1]&4294967267)|0;
		movementMask[i+1] = (movementMask[i+1]&4294967295)|0;
		level[i+2] = (level[i+2]&4294967267)|0;
		movementMask[i+2] = (movementMask[i+2]&4294967295)|0;

      }
    }
  }
}

