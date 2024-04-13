Doshi, Harsh Amit 22203459

Dulkin, Ilia 22112260

# Car Price Recommendation System

Repository: [Car Price Recommendation System](https://mygit.th-deg.de/id05260/car-price-recommendation-system
)

Original dataset taken from kaggle.com: [Dataset](https://www.kaggle.com/datasets/sukhmanibedi/cars4u)

Cleaned dataset suitable for GUI: [Dataset](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/used_cars_data_cleaned.csv?ref_type=heads)

Dataset suitable for model training: [Dataset](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/cardata_numerized.csv?ref_type=heads)
## Project Description
Our project is designed to help users from India predict the approximate price of a used car according to their preferences.

## Prerequisites

| Name | Version |
| ------ | ------ |
| Python | 3.11.4|
| PyQt6  | 6.6.1   |
| Matplotlib | 3.8.2|
| numpy  | 1.26.3   |
| pandas | 2.1.4|
| scikit-learn | 1.3.2|
| used_cars_data_cleaned | [link](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/used_cars_data_cleaned.csv?ref_type=heads)|
| cardata_numerized | [link](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/cardata_numerized.csv?ref_type=heads)|

## Installation
Please download [*main.py*](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/main.py?ref_type=heads), [*cardata_numerized.csv*](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/cardata_numerized.csv?ref_type=heads) and [*used_cars_data_cleaned.csv*](https://mygit.th-deg.de/id05260/car-price-recommendation-system/-/blob/main/used_cars_data_cleaned.csv?ref_type=heads) files into the same folder. Also the whole project can be downloaded as a zip file.

All libraries from prerequisites section should be also installed for success running of the program.

### Virtual Envirnments
If you use an IDE such as PyCharm, a virtual envirnment will be created automatically. \
Otherwise, you can create a virtual envirnment using the following commands: 

****For Windows**:**

    python -m venv /path/to/new/virtual/environment

****For Linux + macOS**:**

    (venv) $ python -m pip install <package-name>

Once you have created a virtual envirnment, you must activate it, before installing the packages.

****Windows**:**

    PS> venv\Scripts\activate
    (venv) PS>

****Linux + macOS**:**

    $ source venv/bin/activate
    (venv) $

You can then install the packages:

**Windows:**

    (venv) PS> python -m pip install <package-name>

**Linux + macOS:**

    (venv) $ python -m pip install <package-name>

You can find more information here: 
- [Official Documentation For Virtual Envirnments in Python](https://docs.python.org/3/library/venv.html) 
- [Virtual Envirnments Tutorial - RealPython](https://realpython.com/python-virtual-environments-a-primer/#how-can-you-customize-a-virtual-environment) (Personal login required)

## Basic Usage

1. Please run the main.py file.

2. After the application has started, the data file is read after clicking on the 'Start Analisys' button.

- Predicted price for default inputs is shown.
- 'Start Analisys' button is disabled, and its text is set to 'Interactive Analisys Started'.

3. Custom values can be submitted using various widgets.
- The plot reacts interactively to the input and gives a prediction.

4. The modified price is displayed on the plot, and on the text section above the 'Interactive Analisys Started' button.

> Note: Price is displayed in Euros and in lakhs on the plot.\
1 lakh Indian rupees = ~1000 Euros.


5. Additionally, the user can view some trends on the data using the buttons on the right of the plot canvas.
The text above them explains the plot.

6. Save button stores the plot in ".png" format at user's specified location.
- In case of unsaved data, the "quit button" asks for confirmation before closing.

## Implementation of the Requests

### Graphical User Interface (GUI)

#### Data Import
We import the data in our code using the "pandas.read_csv" method in the *generate_model* method.

#### Data Reading And Analysis
The data is analyzed after clicking on the "Start Analysis" button.

#### Input Widgets And Statistical Metrics
We use 3 different types of input widgets for 8 statistical metrics:
- QComboBox:
    - transmission_type: maps the transmission type of the car to our *prediction function*
    - fuel_type : maps the kind of fuel used to the *prediction function*
    - brand_class: maps the brand of the car to the *prediction function*
- QSpinbox:
    - power: maps the power of the car to our *prediction function*
    - kmd : maps the number of kilometers the car was used for to the *prediction function*
    - engine : maps the strength of the ca engine to the *prediction function*
- QSlider:
    - age_slider: maps the age of the car to the *prediction function*
    - seat_no: maps the number of seats in the car to the *prediction function*

The *prediction function* is called "show_prediction"

### Visualization And Data Overview For User
To implement this part we used Matplotlib 3.8.2 library and integrate it in PyQt6.
There are 6 plots available for user:
- Main plot with Year Vs Price of car
- Fuel Type Vs Average Mileage,
- Number Of Seats Vs Average Price
- Owner Type Vs Average Price
- Transmission Distribution
- Seats Number Distribution

### Data Analysis With Pandas and Numpy
To implement this part we used pandas 2.1.4, numpy 1.26.3 and scikit-learn 1.3.2 libraries.

Biggest part of this section was done during the dataset cleaning and exploration:
- data was observed
- otliers were removed
- during data exploration the left-skewed pattern of price and kilometer distribution were founded; it is the reason why log price and log km was used for model training and prediction

The part of data analysis which is available for user was done usimg 
pandas.describe(), pandars.info() libraries and scikit-learn functions m mean_squared_error() and r2_score()

### Scikit-Learn 
This section was implemented using LinearRegression class form scikit-learn library.

We splitted dataset from "used_cars_dataset_cleaned.csv" file into 4 parts: X, y for training the model and X, y for testing the model. 
- X_train - 70% of rows from all columns except "Price_log"
- X_test - remaining 30% of rows from all columns except "Price_log"
- y_train - corresponding 70% of rows from "Price_log" column. Using it as a labels for model
- y_test - corresponding 30% of rows from "Price_log" column. Using it to measere the accurcy of our model

Mean squared error is equal to 0.1, which could be called a good result 

## Work Done

Harsh's part:
- GUI
- Integration of Matplotlib in PyQt6
- Data analisys and plotting of:
    - Age Of Car Vs Price,
    - Fuel Type Vs Mileage,
    - Number Of Seats Vs Price

Ilia's part:
- Dataset cleaning, exploration and description.
- Scikit-Learn
- Data analisys and plotting of:
    - Owner Type Vs Price
    - Transmission Distribution
    - Seats Number Distribution
