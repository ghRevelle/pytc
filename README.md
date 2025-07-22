# PyTC  

Python Tower Controller (PyTC) is a software package including a deep reinforcement learning algorithm and a flight radar simulator. It it not meant to be a replacement for Tower Control; it is a proof-of-concept for potentially utilizing DRL algorithms for the role.  

## Description  

PyTC comprises of several major parts:  
1. Sample training data and data manipulating algorithms
2. A DRL algorithm, training environment, and training function
3. A flight simulator to process commands issued by the algorithm

## Contents  

- **data**
    - data_filter.py: filter operations (takeoff and landing) data by airport
    - data_scraper.py: scrape data from ADS-B Exchange, filtered by ICAO hex codes
    - filtered_aircraft.csv: contains filtered data of all planes that took off or landed at Rapid City Regional Airport on July 1, 2024
    - filtered_operations.csv: contains the time and runway of planes that took off or landed at Rapid City Regional Airport on July 1, 2024
    - sources.txt: list of our sources
- **src**
    - \_\_init\_\_.py: registers our custom gymnasium environment
    - airport.py: airport and runway class
    - commands.py: command dataclass
    - DRL_algorithm.py: DRL algorithm, training function, etc.
    - DRL_env.py: custom gymnasium training environment
    - flightsim.py: flight simulator class
    - **main.py**: run tests here
    - plane_manager.py: plane manager class (stores plane data)
    - plane.py: plane class
    - pygame_display.py: Pygame display class
    - utils.py: useful conversion functions
- **pytc_env.yml**: virtual environment information

## Notes  

Rapid City Regional Airport is an arbitrary regional airport we chose for our testing.  

Why?  

- It is small enough in scale
- It does not have a lot of surrounding traffic
- It is in the US (more data available)
- One of our developers has visited before

