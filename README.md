<!-- badges: start -->

![Static Badge](https://img.shields.io/badge/lifecycle-wip-red)
![Static Badge](https://img.shields.io/badge/license-MIT-white)
![Static Badge](https://img.shields.io/badge/python-3.12.4-blue)

<!-- badges: end -->

# PyTC: Python Tower Controller 

Python Tower Controller (PyTC) is a software package that aims to train a deep reinforcement learning (DRL) algorithm to act as a Tower Controller for an airport. The PyTC package includes said DRL, as well as a Pygame-based simulation of the San Diego Intl. Airport and its associated aircraft (developed in-house).

We want to answer the question: how would DRL algorithms compare to real, human Tower Controllers? This issue has become pertinent recently, since the worsening strain on ATCs and Tower Controllers has led to disastrous air accidents in the US. While we don't claim to have a solution to such a multifaceted issue, we did want to examine how we could use machine learning to produce a viable AI Tower Controller. In the future, a tool similar to ours could take some burden off of Tower Controllers, or, if AI becomes truly robust and trusted, could serve as an Tower Controller "hub," communicating with and directing autonomous commercial aircraft without human intervention.

## Table of Contents
1. [Description](#description)
2. [File Tree](#file-tree)
3. [Software and Packages Used](#software-and-packages-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Notes and Simplifying Assumptions](#notes-and-simplifying-assumptions)

## Description  

PyTC is comprised of several major parts:  
1. Sample training data and data-scraping algorithms
2. A DRL algorithm, training environment, and training function
3. A flight simulator to process commands issued by the algorithm

## File Tree  

- **Data/**
    - AnalysisData/
        - aircraft_cleaned_20250301.csv: 5-second interval traces of aircraft that took off or landed at San Diego
        - operations_cleaned_20250301.csv: all flights that took off or landed at San Diego
    - sources.txt: list of our sources
- **Scripts/**
    - ProcessingScripts/
        - data_cleaner.ipynb: cleans data
        - data_filter.py: filters operations (takeoff and landing) data by airport
        - data_scraper.py: scrapes data from ADS-B Exchange, filtered by ICAO hex codes
        - initial_state_creator.ipynb: prepares and writes a python file containing the rolling initial state dictionary
    - \_\_init\_\_.py: registers our custom gymnasium environment
    - airport.py: airport and runway class
    - command_handlers.py: multiple command-handling classes to handle plane movements
    - commands.py: command dataclass
    - DRL_algorithm.py: DRL algorithm, training function, etc.
    - DRL_env.py: custom gymnasium training environment
    - flightsim.py: flight simulator class
    - **main.py**: run tests here
    - plane_manager.py: plane manager class (stores plane data)
    - plane.py: plane class
    - planestates.py: plane states Enum
    - pygame_display.py: Pygame display class
    - rolling_initial_state_20250301.py: a sample rolling initial state
    - utils.py: useful conversion functions
- **pytc_env.yml**: virtual environment information

## Software and Packages Used

- Python 3.12.4
- torch
- gymnasium
- pygame
- numpy
- pandas
- scikit-learn
- geopy
- shapely
- requests
- pip

## Installation

Before starting, ensure you have Python 3.12 and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.
1. Clone the repository:  
   ```bash
   git clone https://github.com/ghRevelle/pytc.git
    ``` 
2. Navigate to the project directory:  
   ```bash
   cd pytc
   ```
3. Create a virtual environment:  
   ```bash
   conda env create -f pytc_env.yml # Create a virtual environment named 'pytc'
   ```
4. Activate the virtual environment:  
   ```bash
   conda activate pytc
   ```
5. In the virtual environment, install torch (compatible with Python 3.12 and CUDA 12.8):
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

## Usage

### Data Collection  
1. Download operations.csv.gz from ADS-B Exchange for a specific date
2. Use **data_filter.py** to filter operations data to one specific airport
3. Use **data_scraper.py** to scrape plane data for every 5-second interval in the day, filtered by your filtered operations
4. Use **data_cleaner.ipynb** to clean your aircraft and operations data
5. Use **initial_state_creator.ipynb** to construct a rolling initial state dictionary to be used in the simulator

### Testing  
To run the tests, use main.py:
```bash
python .\Scripts\main.py
```
## Notes and Simplifying Assumptions

We chose San Diego International Airport (ICAO: KSAN) for our testing. Naturally, our model will be overfitted to work well for ONLY this airport--in real life, different airports could train a different version of the DRL suited to their specific airport conditions. 

### Why did we choose San Diego?  

- It has one runway: this makes our model as simple as possible
- It handles mostly commercial aircraft: this means approaching aircraft should be aligned with the runway multiple miles before entering tower airspace (as opposed to the more freeform general aviation alignment conditions)
- It is in the US: more data is available, and FAA regulations are easy to find

### Simplifying Assumptions
[under construction]

- Because of approach control’s wonderful work, planes will always spawn near-parallel to their landing runway.
- Because of approach control's wonderful work, planes will always spawn less than two turn radii away from the approach line.
- A plane going around will climb to approach altitude and despawn from the simulation -> in real life, the plane would continue ~5 nm out and put in a holding pattern, where approach control deals with them again; it's realistically not going to come back over the course of a simulation episode
- We will not consider emergencies, strange go-arounds, etc. that we find in our data -> our model serves as a proof-of-concept and shouldn't need to respond to uncommon scenarios
- Auto-throttle is always on for all of the planes -> this means that aircraft do not lose/gain speed in turns (though climbs and descents do influence speed slightly)
- Autopilot and ILS allow planes to always stay on course for landing. In our simulation, this is represented by planes automatically adjusting to be aligned and snapping their position and heading once close enough
- In real life, a flight might land, have a quick turnaround time, and take off again. Thus, the landing and takeoff are not independent. For our simulation, we will assume they are independent and treat them like two different aircraft.
