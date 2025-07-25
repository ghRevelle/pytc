<!-- badges: start -->

![Static Badge](https://img.shields.io/badge/lifecycle-wip-red)
![Static Badge](https://img.shields.io/badge/license-MIT-white)
![Static Badge](https://img.shields.io/badge/python-3.12.4-blue)

<!-- badges: end -->

# PyTC : Python Tower Controller 

Python Tower Controller (PyTC) is a software package that aims to train a deep reinforcement learning (DRL) algorithm to act as a Tower Controller for an airport. The PyTC package includes said DRL, as well as a Pygame-based simulation of the San Diego Intl. Airport and its associated aircraft (developed in-house).

We want to answer the question: how would DRL algorithms compare to real, human Tower Controllers? This issue has become pertinent recently, since the worsening strain on ATCs and Tower Controllers has led to disastrous air accidents in the US. While we don't claim to have a solution to such a multifaceted issue, we did want to examine how we could use machine learning to produce a viable AI Tower Controller. In the future, a tool similar to ours could take some burden off of Tower Controllers, or, if AI becomes truly robust and trusted, could serve as an Tower Controller "hub," communicating with and directing autonomous commercial aircraft without human intervention.

## Table of Contents
1. [Description](#description)
2. [File Tree](#file-tree)
3. [Software and Packages Used](#software-and-packages-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Notes](#notes)

## Description  

PyTC is comprised of several major parts:  
1. Sample training data and data-scraping algorithms
2. A DRL algorithm, training environment, and training function
3. A flight simulator to process commands issued by the algorithm

## File Tree  

- **Data/**
    - AnalysisData/
        - contains cleaned data
    - data_scraper.py: scrapes data from ADS-B Exchange, filtered by ICAO hex codes
    - sources.txt: list of our sources
- **Scripts/**
    - ProcessingScripts/
        - data_cleaner.ipynb: cleans data
        - data_filter.py: filters operations (takeoff and landing) data by airport
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
    - utils.py: useful conversion functions
- **pytc_env.yml**: virtual environment information

## Software and Packages Used

- Python 3.12.4
- pygame
- gymnasium
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

## Usage

To clean data, use data_cleaner.ipynb and do the following:
1. Load your filtered csv file
2. Run the code cells for your desired form of cleaning (clean aircraft vs operations)
3. You now have a clean csv file

To run the tests, use main.py:
```bash
python Scripts/main.py
```
## Notes  

San Diego International Airport is the airport we chose for our testing.  

Why?  

- It has 1 runway: this makes our model as simple as possible
- It handles mostly commercial aircraft: this means approaching aircraft should be aligned with the runway multiple miles before entering tower airspace
- It is in the US (more data available)

