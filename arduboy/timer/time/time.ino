void setup() {
  Serial.begin(9600);
  // put your setup code here, to run once:

}


long fn(long x){
	if (x<1){
		return 1;
	} else {
		return fn(x-1)+fn(x-2);
	}
}


void l3(){
	unsigned long t1 = micros();
	byte a = 0b10110101;
	byte b = 0b11011010;
	byte c = 0b11110011;
	byte d = 0b10110101;
	byte e = 0b11011010;
	byte f = 0b11110011;
	fn(10);
	unsigned long t2 = micros();
	Serial.print("F3 ");
	Serial.println(t2-t1);

}

void l1(){
  // put your main code here, to run repeatedly:
	byte a = random()&0b11111111;
	byte b = random()&0b11111111;
	byte c = random()&0b11111111;
	byte d = random()&0b11111111;
	byte e = random()&0b11111111;
	byte f = random()&0b11111111;
	unsigned long t1 = micros();
	for (volatile long i=0;i<1000000;i++){
		a=(a&b+b&c)|(b*c);
		d=(d&e+e&f)|(e*f);
	}
	unsigned long t2 = micros();
	Serial.print("F1 ");
	Serial.println(t2-t1);
}


void l2(){
  // put your main code here, to run repeatedly:
	word a = random()&0b1111111111111111;
	word b = random()&0b1111111111111111;
	word c = random()&0b1111111111111111;
	unsigned long t1 = micros();
	for (volatile long i=0;i<1000000;i++){
			a=(a*b+b*c)|(b*c);
	}
	unsigned long t2 = micros();
	Serial.print("F2 ");
	Serial.println(t2-t1);
}

void loop() {
	l1();
	l2();
	l3();
	delay(1000);
}