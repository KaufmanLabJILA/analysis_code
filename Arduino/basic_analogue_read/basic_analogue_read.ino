byte rx_byte = 0;

void setup() {
  // set ADC resolution to 12 bit (only on arduino due)
  analogReadResolution(12);
  // change voltage reference to 1.1V for more resolution on photodiodes
  // which saturate at 600mV (not available on Due...)
  // analogReference(EXTERNAL)
  // initialize both serial ports:
  Serial.begin(115200);
  Serial.println("DAQ ready.");
}
void loop() {
  if (Serial.available() > 0) {
    rx_byte = Serial.read();       
    if (rx_byte = 1) {
      Serial.println(micros()); 
      Serial.println(analogRead(0));
    }
    rx_byte = 0;
  }
}
