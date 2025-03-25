# Truck Delivery Logistics
## Overview
The Data Analysis Tool is a Python application built using the Tkinter library for the graphical user interface (GUI) and various data analysis libraries such as Pandas, Matplotlib, Seaborn, and Scikit-learn. This tool allows users to perform various data analysis tasks, including supplier accuracy classification, trip classification, material booking analysis, KMeans clustering, and vehicle distance and fuel consumption calculations.

## Features
    Supplier Accuracy Classification: Analyze supplier performance based on on-time and delayed deliveries.
    Trip Classification: Classify trips based on customer bookings and visualize the results.
    Material Booking Analysis: Analyze material bookings by supplier and visualize the results.
    KMeans Clustering: Perform clustering analysis on supplier data based on specified features.
    Vehicle Distance and Fuel Consumption Calculator: Calculate total distance traveled and fuel consumption based on vehicle type and date range.
## Requirements
To run this application, you need to have the following Python packages installed:

  tkinter
  pandas
  matplotlib
  seaborn
  scikit-learn
  numpy
You can install the required packages using pip:

    pip install pandas matplotlib seaborn scikit-learn
## Usage
Run the Application: Execute the script to launch the Data Analysis Tool.
Select Analysis Type: Choose the type of analysis you want to perform from the dropdown menu.
Input Data: Depending on the selected analysis type, you will be prompted to input relevant data, such as CSV files, date ranges, or specific column names.
View Results: After performing the analysis, results will be displayed in the application, including visualizations such as bar charts, pie charts, and confusion matrices.
#Analysis Types
1. Supplier Accuracy Classification
Input the year for analysis and the column name containing date information.
The tool will classify suppliers based on on-time and delayed deliveries and display the results in pie charts.
2. Trip Classification
Browse for a CSV file containing trip data.
Input the month for analysis.
The tool will classify trips as "Market" or "Regular" and display the results.
3. Material Booking Analysis
Browse for a CSV file containing material booking data.
Input the supplier name code for analysis.
The tool will analyze bookings and display results, including a bar graph of materials shipped.
4. KMeans Clustering
Input column names for BookingID_Date, supplierNameCode, and Origin_Location.
The tool will perform KMeans clustering and display the results in scatter plots.
5. Vehicle Distance and Fuel Consumption Calculator
Load a CSV file containing vehicle trip data.
Input the start and end dates for analysis.
The tool will calculate total distance and fuel consumption for each vehicle type and display the results in a bar chart.
