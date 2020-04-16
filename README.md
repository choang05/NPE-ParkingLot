**Objective:**

- Create a logistic system for car parking lots so residents can have designated parking spot based on personal verification.
- The system will utilize a network of RFID readers to detect vehicles and communicate with car tags. A progressive web application (PWA) will be used to coordinate, detect and assign currently parked cars to specific parking locations.

**Challenges:**

- Creating a network of RFID readers to communicate with a web application
  - Modeling and coordinating the data
  - Logistics for embedding the readers throughout the lot
  - Physical network can allow for continuous power supply and minimal latency/interference for communication
  - Wireless network may be cheaper but may result in interference from other devices or potential security issues.
- PWA will handle the communication between the server, the view for the application and receiving the data from the RFID readers.
  - Potential stack
    - Server: Nginx
    - Database: MariaDB
    - Back-end: Laravel
    - Front-end: Vue.js + [MaterializeCSS](http://materializecss.com/) / [Bulma](http://bulma.io/)

**Solutions:**

- **RFID tags and readers**
  - **Idea 1**
    - RFID tags can be used to send data back and forth when in proximity of a RFID reader that is embedded into the concrete. Client drives up into the parking spot where the RFID will trigger the reader and data will be evaluated. This is a very convenient for the client as all they need to do is keep the RFID tag in their car just like an EZ tag. However, this system is very expensive as each parking spot requires a UHF RFID reader.

![](RackMultipart20200416-4-n5pphd_html_832af0d90a8c5d27.gif)

  - **Idea 2**
    - To reduce the overall cost, we can station a lower frequency RFID reader station at each parking spot where the vehicle driver must scan their card to register their parking spot after they have parked. Although this significantly reduces RFID reader cost, it becomes a inconvenience for the client driver since they would need to manually scan their RFID tags.

![](RackMultipart20200416-4-n5pphd_html_22e6d6d20b76c179.gif)

  - **Idea 3 (unresearched)**
    - A possibility of reducing overall cost as well as reducing the need for readers at each parking spot would be to place multiple UHF RFID readers in strategic locations where a RFID tag inside the vehicle can be triangulated between multiple readers to determine a 3 dimensional position that can be evaluated by the system to check if which parking spots are being used and where that RFID tag is.

![](RackMultipart20200416-4-n5pphd_html_ec976ced7bf602ef.gif)

  - **Hardware** :
    - Microprocessor
      - Arduino
        - Only one language: Arduino C
        - Power efficient
        - Inexpensive ($3 - $20)
      - Raspberry pi
        - Multi Platform: Python, Linux, etc.
        - Model B: $35~
        - Model W: $5 - $10
    - RFID writable cards
      - Cards, tags, charms, keychain buttons, etc.
    - RFID Reader
      - At Least _Ultra High Frequency_ range (\&lt; 865 – 960 MHz)
  - **Pros**
    - Easy to set up
      - Readers prebuilt to be flat
      - Passive RFID cards given to clients with not extra steps
    - High Accuracy
      - RFIDs card can be programed to respond to specific codes from the reader.
  - **Cons**
    - Very expensive
      - RFID ranges above few cms require higher frequency readers.
      - RFID readers that read up to 6 meter can cost between $200-400 per reader depending on bulk-buys and distributors.
    - Readers won&#39;t detect when a vehicle without a RFID tag is in the parking slot.
      - Solution: use other sensors such as pressure/metal detectors/IR/Ultrasonic, etc. to test if a vehicle is within proximity.

- **Solution 2** : **Computer vision for license plates**
  - **Idea**
    - With the use of computer vision software and possibly neural networks such as _OpenCV_ and _TensorFlow_, we can use cameras tied to each parking slot take pictures of the license when a is within proximity. The image recognition algorithm will convert the license plate into characters that can be evaluated with the valid license plates in the database. Camera processing can be done by server or locally.

![](RackMultipart20200416-4-n5pphd_html_492326400bdae326.gif)

  - **Pros**
    - Setup difficulty may vary
      - Camera inside poles in front of every parking lot that is the same height as the license plate
      - Because cameras do all the work, clients do not need any peripherals such as RFID tags.
    - Cars have license plates on front and rear ends of the car which guarantees that parking orientation does not affect license plate visibility. Although this is by law in Texas, some clients do not follow this.
    - If using convolutional networks for training, license plate recognition can be more accurate and used to recognize multiple things such as vehicle shape, type, model, etc. or even client driver.
  - **Cons**
    - Can be expensive
      - HD cameras can be expensive (unresearched) per parking spot.
        - Solution: use less camera that cover more than 1 parking spot but accuracy reliability will suffer due from distance.
    - Camera obstructions
      - Low light levels at night (solution: use IR cameras)
      - Rain
      - Dirt
      - Protective glass wear and tear
      - Other obstructions? Squirrels, birds, etc.
    - Not power efficient
      - Requires cameras to always be on or pictures have to be taken at intervals which suffers reliability.
        - Solution: use sensors as stated for the RFID to test if a car is within proximity before turning on the camera.
    - License plate image recognition difficulty
      - There are many types of license plates with different fonts, styles, etc. and would need a well polished/trained algorithm to recognize them all.
