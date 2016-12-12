
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
const byte PLAYER_MASK = 0b00011110;
const word PLAYER_LAYERMASK = 0b0000111110000000;

const word LAYERMASK[] = {
    0b0000000000000001,
    0b0000000001111110,
};

        byte titleSelection = 2;

        void drawTitle(){

          arduboy.setCursor(46, 0);
          arduboy.print(F("Kettle"));
          

          arduboy.setCursor(10, 10);
          arduboy.print(F("by stephen lavelle"));

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
        
const int GLYPH_COUNT = 7;

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
        0b11111111,
        0b11111111,
        0b11111111,
        0b11111111,
        0b11111111,
        0b11111111,
        0b11111111,
        0b11111111,
    },
};

PROGMEM const byte tiles_w[][8] = {
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
        0b01011110,
        0b01111110,
        0b01011110,
        0b01011110,
        0b01111110,
        0b01011110,
        0b00000000,
    },
    {
        0b00000000,
        0b01111010,
        0b01111110,
        0b01111010,
        0b01111010,
        0b01111110,
        0b01111010,
        0b00000000,
    },
    {
        0b00000000,
        0b01111110,
        0b01111110,
        0b01111110,
        0b01111110,
        0b00001000,
        0b01111110,
        0b00000000,
    },
    {
        0b00000000,
        0b01111110,
        0b00001000,
        0b01111110,
        0b01111110,
        0b01111110,
        0b01111110,
        0b00000000,
    },
    {
        0b00000000,
        0b00001000,
        0b01111110,
        0b00011110,
        0b00011110,
        0b01111110,
        0b00001000,
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

PROGMEM const byte levels[][128] {
    {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,4,4,4,4,0,0,0,0,0,0,
        0,0,0,0,0,16,0,0,0,32,8,0,0,0,0,0,
        0,0,0,0,0,16,0,1,1,0,8,0,0,0,0,0,
        0,0,0,0,0,16,32,1,1,0,8,0,0,0,0,0,
        0,0,0,0,0,16,32,0,0,32,8,0,0,0,0,0,
        0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    },
    {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,4,4,4,4,0,0,0,0,0,0,
        0,0,0,0,0,16,33,33,1,0,8,0,0,0,0,0,
        0,0,0,0,0,16,1,1,1,32,8,0,0,0,0,0,
        0,0,0,0,0,16,33,1,33,32,8,0,0,0,0,0,
        0,0,0,0,0,16,32,0,32,32,8,0,0,0,0,0,
        0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    },
};

bool applyRule0_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 16 ) && ( movementMask[i0+0] & 32)){
                Serial.println(F("0_0"));
                level[i0+0] = (level[i0+0]&4294967169)|16;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967263 ) | 0;
            }
        }
    }
}
bool applyRule1_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 16 ) && ( movementMask[i0+0] & 64)){
                Serial.println(F("1_0"));
                level[i0+0] = (level[i0+0]&4294967169)|16;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967231 ) | 0;
            }
        }
    }
}
bool applyRule2_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 16 ) && ( movementMask[i0+0] & 128)){
                Serial.println(F("2_0"));
                level[i0+0] = (level[i0+0]&4294967169)|16;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967167 ) | 0;
            }
        }
    }
}
bool applyRule3_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 2 ) && ( movementMask[i0+0] & 64)){
                Serial.println(F("3_0"));
                level[i0+0] = (level[i0+0]&4294967169)|2;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967231 ) | 0;
            }
        }
    }
}
bool applyRule4_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 2 ) && ( movementMask[i0+0] & 128)){
                Serial.println(F("4_0"));
                level[i0+0] = (level[i0+0]&4294967169)|2;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967167 ) | 0;
            }
        }
    }
}
bool applyRule5_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 2 ) && ( movementMask[i0+0] & 256)){
                Serial.println(F("5_0"));
                level[i0+0] = (level[i0+0]&4294967169)|2;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967039 ) | 0;
            }
        }
    }
}
bool applyRule6_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 8 ) && ( movementMask[i0+0] & 32)){
                Serial.println(F("6_0"));
                level[i0+0] = (level[i0+0]&4294967169)|8;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967263 ) | 0;
            }
        }
    }
}
bool applyRule7_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 8 ) && ( movementMask[i0+0] & 64)){
                Serial.println(F("7_0"));
                level[i0+0] = (level[i0+0]&4294967169)|8;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967231 ) | 0;
            }
        }
    }
}
bool applyRule8_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 8 ) && ( movementMask[i0+0] & 256)){
                Serial.println(F("8_0"));
                level[i0+0] = (level[i0+0]&4294967169)|8;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967039 ) | 0;
            }
        }
    }
}
bool applyRule9_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 4 ) && ( movementMask[i0+0] & 32)){
                Serial.println(F("9_0"));
                level[i0+0] = (level[i0+0]&4294967169)|4;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967263 ) | 0;
            }
        }
    }
}
bool applyRule10_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 4 ) && ( movementMask[i0+0] & 128)){
                Serial.println(F("10_0"));
                level[i0+0] = (level[i0+0]&4294967169)|4;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967167 ) | 0;
            }
        }
    }
}
bool applyRule11_0(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 4 ) && ( movementMask[i0+0] & 256)){
                Serial.println(F("11_0"));
                level[i0+0] = (level[i0+0]&4294967169)|4;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967039 ) | 0;
            }
        }
    }
}
bool applyRule12_0(){
    for (byte y0=0;y0<7;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( level[i0+16] & 0b11110 ) && ( movementMask[i0+16] & 32)){
                Serial.println(F("12_0"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 32;
                level[i0+16] = (level[i0+16]&4294967295)|0;
                movementMask[i0+16] = ( movementMask[i0+16]&4294967295 ) | 32;
            }
        }
    }
}
bool applyRule12_1(){
    for (byte y0=0;y0<7;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 0b11110 ) && ( movementMask[i0+0] & 64) && ( level[i0+16] & 32 )){
                Serial.println(F("12_1"));
                level[i0+0] = (level[i0+0]&4294967295)|0;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 64;
                level[i0+16] = (level[i0+16]&4294967169)|32;
                movementMask[i0+16] = ( movementMask[i0+16]&4294967295 ) | 64;
            }
        }
    }
}
bool applyRule12_2(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<15;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( level[i0+1] & 0b11110 ) && ( movementMask[i0+1] & 128)){
                Serial.println(F("12_2"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 128;
                level[i0+1] = (level[i0+1]&4294967295)|0;
                movementMask[i0+1] = ( movementMask[i0+1]&4294967295 ) | 128;
            }
        }
    }
}
bool applyRule12_3(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<15;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 0b11110 ) && ( movementMask[i0+0] & 256) && ( level[i0+1] & 32 )){
                Serial.println(F("12_3"));
                level[i0+0] = (level[i0+0]&4294967295)|0;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 256;
                level[i0+1] = (level[i0+1]&4294967169)|32;
                movementMask[i0+1] = ( movementMask[i0+1]&4294967295 ) | 256;
            }
        }
    }
}
bool applyRule13_0(){
    for (byte y0=0;y0<7;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( level[i0+16] & 32 ) && ( movementMask[i0+16] & 32)){
                Serial.println(F("13_0"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 32;
                level[i0+16] = (level[i0+16]&4294967169)|32;
                movementMask[i0+16] = ( movementMask[i0+16]&4294967295 ) | 32;
            }
        }
    }
}
bool applyRule13_1(){
    for (byte y0=0;y0<7;y0++){
        for (byte x0=0;x0<16;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( movementMask[i0+0] & 64) && ( level[i0+16] & 32 )){
                Serial.println(F("13_1"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 64;
                level[i0+16] = (level[i0+16]&4294967169)|32;
                movementMask[i0+16] = ( movementMask[i0+16]&4294967295 ) | 64;
            }
        }
    }
}
bool applyRule13_2(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<15;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( level[i0+1] & 32 ) && ( movementMask[i0+1] & 128)){
                Serial.println(F("13_2"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 128;
                level[i0+1] = (level[i0+1]&4294967169)|32;
                movementMask[i0+1] = ( movementMask[i0+1]&4294967295 ) | 128;
            }
        }
    }
}
bool applyRule13_3(){
    for (byte y0=0;y0<8;y0++){
        for (byte x0=0;x0<15;x0++){
            byte i0 = x0+16*y0;
            if (( level[i0+0] & 32 ) && ( movementMask[i0+0] & 256) && ( level[i0+1] & 32 )){
                Serial.println(F("13_3"));
                level[i0+0] = (level[i0+0]&4294967169)|32;
                movementMask[i0+0] = ( movementMask[i0+0]&4294967295 ) | 256;
                level[i0+1] = (level[i0+1]&4294967169)|32;
                movementMask[i0+1] = ( movementMask[i0+1]&4294967295 ) | 256;
            }
        }
    }
}

void processRules(){
    applyRule0_0();
    applyRule1_0();
    applyRule2_0();
    applyRule3_0();
    applyRule4_0();
    applyRule5_0();
    applyRule6_0();
    applyRule7_0();
    applyRule8_0();
    applyRule9_0();
    applyRule10_0();
    applyRule11_0();
    applyRule12_0();
    applyRule12_1();
    applyRule12_2();
    applyRule12_3();
    applyRule13_0();
    applyRule13_1();
    applyRule13_2();
    applyRule13_3();
}
void processLateRules(){
}

void checkWin(){
    {
        for (byte i=0;i<128;i++){
            if ( !(level[i]&1) && (level[i]&32) ){
                return;
            }
        }
    }

    waiting=true;
    waitfrom=millis();
}