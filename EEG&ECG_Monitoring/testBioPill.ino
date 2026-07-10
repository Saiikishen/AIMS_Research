//
// Spike Recorder code — ported for Seeed XIAO ESP32C6
//

#define BUFFER_SIZE 100                 // sampling circular buffer size
#define SIZE_OF_COMMAND_BUFFER 30       // command buffer size
#define LENGTH_OF_MESSAGE_IMPULS 100    // length of message impulse in ms
#define MAX_CHANNELS 3                  // XIAO ESP32C6 only breaks out A0-A3

const uint8_t channelPins[MAX_CHANNELS] = {A0, A1, A2};

int head = 0;                    
int tail = 0;                    
byte reading[BUFFER_SIZE];       
char commandBuffer[SIZE_OF_COMMAND_BUFFER];

const int messageImpulsPin = D5; 
volatile int messageImpulseTimer = 0;

int numberOfChannels = 1;        
int commandMode = 0;             // flag: don't sample/output while parsing a command

hw_timer_t *timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t sampleTick = 0; // ISR increments; loop() consumes

const uint32_t BASE_PERIOD_US = 100; // 100us period = 10kHz base rate (1 channel)

// ---- Timer ISR: kept deliberately minimal ----
// analogRead() is NOT safe to call from inside a hardware timer ISR on
// ESP32 -- the ADC driver takes a FreeRTOS lock internally, which causes
// an "Interrupt wdt timeout" crash. So the ISR only flags that a sample
// is due; the actual analogRead() happens in loop().
void ARDUINO_ISR_ATTR onTimer() {
  if (messageImpulseTimer > 0) {
    messageImpulseTimer--;
    if (messageImpulseTimer == 0) {
      digitalWrite(messageImpulsPin, LOW);   // digitalWrite IS ISR-safe
    }
  }
  portENTER_CRITICAL_ISR(&timerMux);
  sampleTick++;
  portEXIT_CRITICAL_ISR(&timerMux);
}

void setup() {
  Serial.begin(230400);
  while (!Serial) { delay(10); }   // native USB CDC -- wait for host to connect
  delay(300);
  Serial.println("StartUp!");
  Serial.setTimeout(2);

  pinMode(messageImpulsPin, OUTPUT);
  analogReadResolution(10);  // 0-1023, matches the 10-bit packing scheme below

  timer = timerBegin(1000000);                // 1 MHz tick rate (1 tick = 1us)
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, BASE_PERIOD_US, true, 0); // fire every 100us = 10kHz
}

// Samples all active channels and packs them into the circular buffer,
// matching the original bit layout: 7 data bits per byte, MSB of only the
// very first byte in a cycle set to 1 (frame marker), 0 everywhere else.
void takeSample() {
  for (int ch = 0; ch < numberOfChannels; ch++) {
    int tempSample = analogRead(channelPins[ch]);
    byte hi = (tempSample >> 7) & 0x7F;
    if (ch == 0) hi |= 0x80;
    reading[head] = hi;
    head = (head + 1) % BUFFER_SIZE;
    reading[head] = tempSample & 0x7F;
    head = (head + 1) % BUFFER_SIZE;
  }
}

// Replaces serialEvent(), which the ESP32 Arduino core doesn't reliably
// invoke automatically -- polled explicitly from loop() instead.
void handleSerialCommands() {
  if (Serial.available() == 0) return;
  commandMode = 1;

  String inString = Serial.readStringUntil('\n');
  inString.toCharArray(commandBuffer, SIZE_OF_COMMAND_BUFFER);
  commandBuffer[inString.length()] = 0;

  char* command = strtok(commandBuffer, ";");
  while (command != 0) {
    char* separator = strchr(command, ':');
    if (separator != 0) {
      *separator = 0;
      --separator;
      if (*separator == 'c') {                 // channel count command
        int requested = atoi(separator + 2);
        if (requested >= 1 && requested <= MAX_CHANNELS) {
          numberOfChannels = requested;
        }
      }
      if (*separator == 'p') {                 // impulse marker command
        digitalWrite(messageImpulsPin, HIGH);
        messageImpulseTimer = (LENGTH_OF_MESSAGE_IMPULS * 10) / numberOfChannels;
      }
      // 's' (sampling rate) intentionally ignored, same as the original
    }
    command = strtok(0, ";");
  }

  timerAlarm(timer, BASE_PERIOD_US * numberOfChannels, true, 0);
  commandMode = 0;
}

void loop() {
  handleSerialCommands();

  portENTER_CRITICAL(&timerMux);
  bool sampleDue = (sampleTick > 0);
  if (sampleDue) sampleTick--;
  portEXIT_CRITICAL(&timerMux);

  if (sampleDue && commandMode != 1) {
    takeSample();
  }

  while (head != tail && commandMode != 1) {
    Serial.write(reading[tail]);
    tail = (tail + 1) % BUFFER_SIZE;
  }
}
