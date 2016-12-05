
enum State {
  LEVEL,
  TITLE,
  MESSAGE
};

const byte LAYER_COUNT = 3;

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
const word PLAYER_LAYERMASK = 0b0000001111100000;

const word LAYERMASK[] = {
	0b0000000000000001,
	0b0000000000011110,
};

        byte titleSelection = 2;

        void drawTitle(){

          arduboy.setCursor(25, 0);
          arduboy.print(F("Robot Spawner"));
          

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
		0b00000000,
		0b01111110,
		0b01000010,
		0b01000010,
		0b01000010,
		0b01000010,
		0b01111110,
		0b00000000,
	},
	{
		0b00000000,
		0b01010100,
		0b00000010,
		0b01000000,
		0b00000010,
		0b01000000,
		0b00101010,
		0b00000000,
	},
	{
		0b00000000,
		0b01111110,
		0b01100110,
		0b01000010,
		0b01000010,
		0b01100110,
		0b01111110,
		0b00000000,
	},
};

PROGMEM const byte levels[][128] {
	{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	},
};

bool applyRule0_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 0b11100 ) && ( !( movementMask[i0+0]&992 ) )){
                level[i0+0] = (level[i0+0]&4294967295)|0;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 0;
                movementMask[i0+0] |= (1<<random(0,4))<<5;
            }
        }
    }
}
bool applyRule1_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 1 ) && !( level[i0+0] & 30 )){
                level[i0+0] = (level[i0+0]&4294967294)|1;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 0;
                switch (random(0,3)){
                    case 0:
                    {
                        level[i0+0] |= 0b100;
                        break;
                    }
                    case 1:
                    {
                        level[i0+0] |= 0b1000;
                        break;
                    }
                    case 2:
                    {
                        level[i0+0] |= 0b10000;
                        break;
                    }
                }
            }
        }
    }
}

void processRules(){
	applyRule0_0();
	applyRule1_0();
}
void processLateRules(){
}

void checkWin(){}