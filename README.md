# PyTC  

Python Tower Controller (PyTC) is a software package including a deep reinforcement learning algorithm and a flight radar simulator. It it not meant to be a replacement for Tower Control; it is a proof-of-concept of a DRL algorithm that could potentially serve the role of a Tower Controller.

We want to answer the question: how would DRL algorithms compare to real, human Tower Controllers? This issue has become pertinent recently, since the worsening strain on ATCs and Tower Controllers has led to disastrous air accidents in the US. While we don't claim to have a solution to such a multifaceted issue, we did want to examine how we could use machine learning to produce a viable AI Tower Controller. In the future, a tool similar to ours could take some burden off of Tower Controllers, or, if AI becomes truly robust and trusted, could serve as an Tower Controller "hub", communicating with and directing autonomous commercial aircraft without human intervention.

## Description  

PyTC is comprised of several major parts:  
1. Sample training data and data manipulating algorithms
2. A DRL algorithm, training environment, and training function
3. A flight simulator to process commands issued by the algorithm

## Contents  

- **data/**
    - cleaned_data/
        - contains cleaned data
    - filtered_data/
        - contains filtered but not cleaned data
    - data_cleaner.ipynb: clean data
    - data_filter.py: filter operations (takeoff and landing) data by airport
    - data_scraper.py: scrape data from ADS-B Exchange, filtered by ICAO hex codes
    - sources.txt: list of our sources
- **src/**
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

## Instructions  

To clean data, use data_cleaner.ipynb and do the following:
1. Load your filtered csv file
2. Run the code cells for your desired form of cleaning (clean aircraft vs operations)
3. You now have a clean csv file

## Notes  

Rapid City Regional Airport is a regional airport we chose for our testing.  

Why?  

- It is not too large: it has just two runways and services either GA or small commercial aircraft
- It does not have a lot of traffic from surrounding airports
- It is in the US (more data available)
- One of our developers has visited before

