enum State {
  LEVEL,
  TITLE,
  MESSAGE
};
State state=TITLE;
const byte DIR_UP     = 0b00001;
const byte DIR_DOWN   = 0b00010;
const byte DIR_LEFT   = 0b00100;
const byte DIR_RIGHT  = 0b01000;
const byte DIR_ACTION = 0b10000;

const word ALL_UP = DIR_UP+(DIR_UP<<5)+(DIR_UP<<10);
const word ALL_DOWN = DIR_DOWN+(DIR_DOWN<<5)+(DIR_DOWN<<10);
const word ALL_LEFT = DIR_LEFT+(DIR_LEFT<<5)+(DIR_LEFT<<10);
const word ALL_RIGHT = DIR_RIGHT+(DIR_RIGHT<<5)+(DIR_RIGHT<<10);
const word ALL_ACTION = DIR_ACTION+(DIR_ACTION<<5)+(DIR_ACTION<<10);

byte undoState[128];
byte level[128];
word movementMask[128];
byte rowCellContents[8];
byte colCellContents[16];
byte mapCellContents=0;
unsigned long waitfrom;
bool waiting=false;
const byte PLAYER_MASK = 0b00000010;
const word PLAYER_LAYERMASK = 0b00000000000000000000001111100000;

const word LAYERMASK[] = {
	0b00000000000000000000000000000001,
	0b00000000000000000000000000001110,
};

        byte titleSelection = 2;

        void drawTitle(){

          arduboy.setCursor(10, 0);
          arduboy.print(F("Block Pushing Game"));
          

          arduboy.setCursor(10, 10);
          arduboy.print(F("by Stephen Lavelle"));

          switch (titleSelection){
            case 2:{
              arduboy.setCursor(22, 30);
              arduboy.print(F("> start game <"));
              break;
            }
            case 0:{
              arduboy.setCursor(28, 26);
              arduboy.print(F("> new game <"));
              arduboy.setCursor(25, 34);
              arduboy.print(F("continue game"));
              break;
            }
            case 1:{
              arduboy.setCursor(40, 26);
              arduboy.print(F("new game"));
              arduboy.setCursor(13, 34);
              arduboy.print(F("> continue game <"));
              break;
            }
          }

          arduboy.setCursor(0,64-15);
          arduboy.print(F("A:reset, B:action\nA+B:restart")); 
          arduboy.display(true);
        }
        
const int GLYPH_COUNT = 4;

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
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
};

PROGMEM const byte tiles_w[][8] = {
	{
		0b00000000,
		0b00000000,
		0b00000000,
		0b00001000,
		0b00010000,
		0b00000000,
		0b00000000,
		0b00000000,
	},
	{
		0b00000000,
		0b01001000,
		0b01111110,
		0b00011110,
		0b00011110,
		0b01111110,
		0b01001000,
		0b00000000,
	},
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
		0b01111110,
		0b01000010,
		0b01000010,
		0b01000010,
		0b01000010,
		0b01111110,
		0b00000000,
	},
};

PROGMEM const byte levels[][128] {
	{
		0,0,0,0,0,4,4,4,4,0,0,0,0,0,0,0,
		0,0,0,0,0,4,0,1,4,0,0,0,0,0,0,0,
		0,0,0,0,0,4,0,0,4,4,4,0,0,0,0,0,
		0,0,0,0,0,4,9,2,0,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,0,8,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,0,4,4,4,0,0,0,0,0,
		0,0,0,0,0,4,4,4,4,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	},
	{
		0,0,0,0,0,4,4,4,4,4,4,0,0,0,0,0,
		0,0,0,0,0,4,0,0,0,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,4,2,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,8,9,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,1,9,0,4,0,0,0,0,0,
		0,0,0,0,0,4,0,0,0,0,4,0,0,0,0,0,
		0,0,0,0,0,4,4,4,4,4,4,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	},
};

bool applyRule0_0_0(){ 
  for (byte y=0;y<7;y++){
    for (byte x=0;x<16;x++){  
      byte i = x+16*y;
      if (( level[i] & 8 ) && ( level[i+16] & 2 ) && ( movementMask[i+16] & 32)){
		level[i] = (level[i]&4294967281)|8;
		movementMask[i] = (movementMask[i]&4294967295)|32;
		level[i+16] = (level[i+16]&4294967281)|2;
		movementMask[i+16] = (movementMask[i+16]&4294967295)|32;

      }
    }
  }
}
bool applyRule0_1_0(){ 
  for (byte y=0;y<7;y++){
    for (byte x=0;x<16;x++){  
      byte i = x+16*y;
      if (( level[i] & 2 ) && ( movementMask[i] & 64) && ( level[i+16] & 8 )){
		level[i] = (level[i]&4294967281)|2;
		movementMask[i] = (movementMask[i]&4294967295)|64;
		level[i+16] = (level[i+16]&4294967281)|8;
		movementMask[i+16] = (movementMask[i+16]&4294967295)|64;

      }
    }
  }
}
bool applyRule0_2_0(){ 
  for (byte y=0;y<8;y++){
    for (byte x=0;x<15;x++){  
      byte i = x+16*y;
      if (( level[i] & 8 ) && ( level[i+1] & 2 ) && ( movementMask[i+1] & 128)){
		level[i] = (level[i]&4294967281)|8;
		movementMask[i] = (movementMask[i]&4294967295)|128;
		level[i+1] = (level[i+1]&4294967281)|2;
		movementMask[i+1] = (movementMask[i+1]&4294967295)|128;

      }
    }
  }
}
bool applyRule0_3_0(){ 
  for (byte y=0;y<8;y++){
    for (byte x=0;x<15;x++){  
      byte i = x+16*y;
      if (( level[i] & 2 ) && ( movementMask[i] & 256) && ( level[i+1] & 8 )){
		level[i] = (level[i]&4294967281)|2;
		movementMask[i] = (movementMask[i]&4294967295)|256;
		level[i+1] = (level[i+1]&4294967281)|8;
		movementMask[i+1] = (movementMask[i+1]&4294967295)|256;

      }
    }
  }
}

void checkWin(){
	{

        for (byte i=0;i<128;i++){
            if ( !(level[i]&1) && (level[i]&8) ){
                return;
            }
        }
	}

  waiting=true;
  waitfrom=millis();
}
