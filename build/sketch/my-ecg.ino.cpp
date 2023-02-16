#include <Arduino.h>
#line 1 "/Users/antonioglenn/Desktop/code-and-projects/my-ecg/my-ecg.ino"
#line 1 "/Users/antonioglenn/Desktop/code-and-projects/my-ecg/my-ecg.ino"
void setup();
#line 13 "/Users/antonioglenn/Desktop/code-and-projects/my-ecg/my-ecg.ino"
void loop();
#line 1 "/Users/antonioglenn/Desktop/code-and-projects/my-ecg/my-ecg.ino"
void setup()
{
    // configure built in LED 
    pinMode(LED_BUILTIN, OUTPUT);
    // read from analog pi A16
    pinMode(A16, INPUT);
    // configure serial port
    Serial.begin(115200);

	
}

void loop()
{
    // toggle built in led 
    float analogValue = analogRead(A16);
    Serial.print(">A16:");
    Serial.println(analogValue);
}
 
