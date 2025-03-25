import tkinter as tk
from tkinter import filedialog, simpledialog,messagebox
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from datetime import datetime
import matplotlib.colors as mcolors
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")
        self.root.geometry("800x600")
        self.scrollable_frame = ScrollableFrame(root)
        self.scrollable_frame.pack(fill="both", expand=True)
        self.label = tk.Label(self.scrollable_frame.scrollable_frame, text="Select Analysis Type:")
        self.label.pack()
        self.analysis_type = tk.StringVar(root)
        self.analysis_type.set("Select")
        self.analysis_dropdown = tk.OptionMenu(self.scrollable_frame.scrollable_frame, self.analysis_type, "Supplier Accuracy Classification", "Trip Classification", "Material Booking Analysis", "KMeans Clustering","Vehicle Distance and Fuel Consumption Calculator", command=self.show_input_fields)
        self.analysis_dropdown.pack()
        self.input_frame = ttk.Frame(self.scrollable_frame.scrollable_frame)
        self.input_frame.pack()
        self.result_label = tk.Label(self.scrollable_frame.scrollable_frame, text="")
        self.result_label.pack()
        self.entry_start_date=None
        self.entry_end_date=None
        self.btn_load_data=None
        self.btn_calculate=None
        self.result_text=None
        # Initialize variables
        self.supplier_data = None
        self.top_10_ontime = None
        self.top_10_delayed = None
    def show_input_fields(self, analysis_type):
        # Clear previous input fields and result label
        for widget in self.input_frame.winfo_children():
            widget.destroy()
            self.result_label.config(text="")
        if analysis_type == "Supplier Accuracy Classification":
            self.create_accuracy_input_fields()
        elif analysis_type == "Trip Classification":
            self.create_trip_input_fields()
        elif analysis_type == "Material Booking Analysis":
            self.create_material_input_fields()
        elif analysis_type == "KMeans Clustering":
            self.create_clustering_input_fields()
        elif analysis_type == "Vehicle Distance and Fuel Consumption Calculator":
            self.input_distance()
    # Function to load CSV data
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            global df1
            df1 = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Data loaded successfully!")
    def input_distance(self):
        # Create input fields and buttons in the scrollable frame
        self.btn_load_data = tk.Button(self.input_frame, text="Load CSV Data", command=self.load_data)
        self.btn_load_data.pack(pady=10)
        tk.Label(self.input_frame, text="Start Date (dd-mm-yyyy):").pack(pady=5)
        self.entry_start_date = tk.Entry(self.input_frame)
        self.entry_start_date.pack(pady=5)
        tk.Label(self.input_frame, text="End Date (dd-mm-yyyy):").pack(pady=5)
        self.entry_end_date = tk.Entry(self.input_frame)
        self.entry_end_date.pack(pady=5)
        self.btn_calculate = tk.Button(self.input_frame, text="Calculate Total Distance and Fuel Consumption", command=self.calculate_distance_and_fuel)
        self.btn_calculate.pack(pady=10)
        self.result_text = tk.Text(self.input_frame, height=10, width=70)
        self.result_text.pack(pady=10)
    def create_accuracy_input_fields(self):
        self.year_label = tk.Label(self.input_frame, text="Enter the year you want to analyze:")
        self.year_label.pack()
        self.year_entry = tk.Entry(self.input_frame)
        self.year_entry.pack()
        self.date_column_label = tk.Label(self.input_frame, text="Enter the column name containing the date information:")
        self.date_column_label.pack()
        self.date_column_entry = tk.Entry(self.input_frame)
        self.date_column_entry.pack()
        self.classify_button = tk.Button(self.input_frame, text="Classify Accuracy", command=self.classify_accuracy)
        self.classify_button.pack()
    def create_trip_input_fields(self):
        self.trip_label = tk.Label(self.input_frame, text="Browse CSV for Trip Analysis:")
        self.trip_label.pack()
        self.trip_button = tk.Button(self.input_frame, text="Browse CSV", command=self.classify_trips)
        self.trip_button.pack()
    def create_material_input_fields(self):
        self.material_label = tk.Label(self.input_frame, text="Browse CSV for Material Booking Analysis:")
        self.material_label.pack()
        self.material_button = tk.Button(self.input_frame, text="Browse CSV", command=self.classify_material_booking)
        self.material_button.pack()
    # Function to calculate total distance and fuel consumption
    def calculate_distance_and_fuel(self):
        try:
            start_date = pd.to_datetime(self.entry_start_date.get(), format='%d-%m-%Y')
            end_date = pd.to_datetime(self.entry_end_date.get(), format='%d-%m-%Y')
            if start_date > end_date:
                messagebox.showerror("Error", "Start date cannot be after end date")
                return
            # Extract the date part alone from 'BookingID_Date'
            df1['BookingID_Date'] = pd.to_datetime(df1['BookingID_Date'].str.split().str[0], format='%d-%m-%Y')
            # Filter the dataframe based on the date range
            filtered_df = df1[(df1['BookingID_Date'] >= start_date) & (df1['BookingID_Date'] <= end_date)]
            if filtered_df.empty:
                messagebox.showinfo("No Data", "No trips found in the given date range")
                return
            # Calculate total distance for each vehicle type
            total_distance = filtered_df.groupby('vehicleType')['TRANSPORTATION_DISTANCE_IN_KM'].sum()
            # Dictionary to hold fuel values for each vehicle type (Randomly assigned for demonstration)
            fuel_values = {
                '32 FT Single-Axle 7MT - HCV': 0.2,
                '32 FT Multi-Axle 14MT - HCV': 0.25,
                '1 MT Tata Ace (Open Body)': 0.1,
                '24 FT SXL Container': 0.18,
                '32 FT Multi-Axle MXL 18MT': 0.3,
                '19 FT OPEN BODY 8 MT': 0.15,
                '17 FT Container': 0.17,
                '20 FT SXL Container': 0.19,
                '1 MT Tata Ace (Closed Body)': 0.12,
                '19 FT Open 7MT - MCV': 0.16,
                '1.5 MT Pickup (Open Body)': 0.22,
                '22 FT Taurus Open 16MT - HCV': 0.27,
                '40 FT 3XL Trailer 35MT': 0.35,
                '40 FT Flat Bed Multi-Axle 27MT - Trailer': 0.28,
                '20 FT CLOSE 7MT-MCV': 0.16,
                '14 FT Open - 3 MT': 0.14,
                '1.5 MT Vehicle (Closed Body)': 0.13,
                '24 / 26 FT Taurus Open 21MT - HCV': 0.26,
                '32 FT Closed Container 15 MT': 0.22,
                '20 FT Open 9MT - MCV': 0.17,
                '40 FT Flat Bed Double-Axle 21MT - Trailer': 0.29,
                '19 FT SXL Container': 0.18,
                '17 FT Open 5MT - MCV': 0.16,
                '22 FT Closed Container': 0.21,
                'Mahindra LCV 1MT': 0.11,
                '22 FT Open Body 16MT': 0.23,
                '25 FT Open Body 21MT': 0.25,
                '30 FT Open SXL 30MT': 0.32,
                '26 FT Taurus Open 27MT - HCV': 0.27,
                '32 FT Open Tarus': 0.22,
                'Tata 407 Open 2MT - Mini-LCV': 0.14,
                '15 FT Single-Axle 7.2MT (8.5 H) Container - HCV': 0.19,
                '20 FT Closed Container 8MT': 0.18,
                'Tata 407 / 14 FT Open 3MT LCV': 0.15,
                '30 FT Open MXL 30MT': 0.33,
                '37 FT Trailer (Open)': 0.35,
                '24 / 26 FT Closed Container 20MT - HCV': 0.26,
                '28 FT Open Body 25MT': 0.28,
                '40 FT Flat Bed Single-Axle 20MT - Trailer': 0.25,
                '27 FT Open Body 21MT': 0.26,
                '25 FT Closed Body 20MT': 0.24,
                '24 | 26 FT Taurus Open 21MT - HCV': 0.26,
                '31 FT Open Body 18MT': 0.27,
                '50 FT Flat Bed 30MT - Trailer': 0.3
            }   
            # C alculate total fuel consumption for each vehicle type
            total_fuel_consumption = total_distance * pd.Series(fuel_values)
            # Display the result
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "Total Distance and Fuel Consumption for each VehicleType:\n\n")
            for vehicle_type, distance in total_distance.items():
                self.result_text.insert(tk.END, f"{vehicle_type}: {distance} km, Fuel Consumption: {total_fuel_consumption[vehicle_type]}liters\n")
            # Clear the previous plot, if exists
            if hasattr(root, 'plot_frame'):
                root.plot_frame.destroy()
            # Create a new frame to hold the plot
            root.plot_frame = tk.Frame(root)
            root.plot_frame.pack(pady=20)
            # Create a figure and plot within the frame
            fig, ax = plt.subplots(figsize=(12,6))
            total_fuel_consumption.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title('Total Fuel Consumption by Vehicle Type')
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Fuel Consumption (Liters)')
            ax.tick_params(axis='x', rotation=90)
            ax.grid(True)
            # Embed the plot into tkinter window
            canvas = FigureCanvasTkAgg(fig, master=self.input_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def classify_accuracy(self):
        year = self.year_entry.get()
        if not year.isdigit():
            tk.messagebox.showerror("Error", "Please enter a valid year.")
            return
        # Read CSV and filter data based on yea
        data = pd.read_csv("Delivery_truck_trip_data.csv", dtype=str) # Specify dtype=str to handle mixed types
        data['BookingID_Date'] = pd.to_datetime(data['BookingID_Date'], errors='coerce', dayfirst=True) # Set dayfirst=True for date parsing
        data = data[data['BookingID_Date'].dt.year == int(year)]
        # Treat 'G' and 'R' as 'Yes', convert them to 1, and treat all other values as 0
        data['ontime'] = (data['ontime'] == 'G').astype(int)
        data['delay'] = (data['delay'] == 'R').astype(int)
        # Group by supplier and calculate on-time and delayed deliveries
        supplier_data = data.groupby('supplierNameCode').agg({
            'ontime': 'sum',
            'delay': 'sum'
        }).reset_index()
        # Sort by on-time and delayed deliveries
        self.top_10_ontime = supplier_data.sort_values(by='ontime', ascending=False).head(10)
        self.top_10_delayed = supplier_data.sort_values(by='delay', ascending=False).head(10)
        # Create pie charts
        self.create_pie_chart(self.top_10_ontime, self.top_10_delayed, "Top 10 On-Time Deliveries", "Top 10 Delayed Deliveries")
    def create_pie_chart(self, data_ontime, data_delay, title_ontime, title_delay):
        if data_ontime is None or data_delay is None:
            return
        # Data for on-time deliveries
        labels_ontime = data_ontime['supplierNameCode']
        sizes_ontime = data_ontime['ontime']
        # Data for delayed deliveries
        labels_delay = data_delay['supplierNameCode']
        sizes_delay = data_delay['delay']
        # Create a figure and two subplots for the two pie charts
        fig, axes = plt.subplots(1, 2, figsize=(16,10))
        # Plot the pie chart for on-time deliveries
        axes[0].pie(sizes_ontime, labels=labels_ontime, autopct='%1.1f%%', startangle=140)
        axes[0].set_title(title_ontime)
        # Plot the pie chart for delayed deliveries
        axes[1].pie(sizes_delay, labels=labels_delay, autopct='%1.1f%%', startangle=140)
        axes[1].set_title(title_delay)
        # Adjust layout
        plt.tight_layout()
        # Convert the plot to a tkinter-compatible format
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        # Pack the canvas into the tkinter window
        canvas.get_tk_widget().pack()
    def classify_trips(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")
            return
        # Clean up column names
        df.columns = df.columns.str.strip()
        input_month = simpledialog.askstring("Input", "Enter the month (MM/YYYY):")
        if not input_month: 
            self.result_label.config(text="No input provided.")
            return
        try:
            df['BookingID_Date'] = pd.to_datetime(df['BookingID_Date'], dayfirst=True)
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")
            return
        df['Booking_Month'] = df['BookingID_Date'].dt.strftime('%m/%Y')
        trips_month = df[df['Booking_Month'] == input_month]
        if 'Market/Regular' not in trips_month.columns:
            self.result_label.config(text="Column 'Market/Regular' not found.")
            return
        # Naive Bayes Classification
        features = ['customerNameCode'] # You can add more relevant features
        if 'Market/Regular' not in df.columns:
            self.result_label.config(text="Column 'Market/Regular' not found in the dataset.")
            return
        trips_month = trips_month.dropna(subset=features + ['Market/Regular'])
        # Convert categorical features to numeric
        X = pd.get_dummies(trips_month[features])
        y = trips_month['Market/Regular'].apply(lambda x: 1 if x == 'Market' else 0)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train the Naive Bayes model
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        # Make predictions   
        y_pred = nb_model.   predict(X_test)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Regular', 'Market']) 
        self.result_label.config(text=f"Naive Bayes Classification Report:\nAccuracy: {accuracy:.2f}\n\n{report}")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Regular', 'Market'])
        # Generate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrix
        self.plot_confusion_matrix(cm, ['Regular', 'Market']) 
        self.result_label.config(text=f"Naive Bayes Classification Report:\nAccuracy: {accuracy:.2f}\n\n{report}")
        # Plot the graph for trip classification
        self.plot_trip_graphs(trips_month, 'Market Trips - Customer Bookings', 'Regular Trips - Customer Bookings')
    def plot_confusion_matrix(self, cm, labels):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        # Convert the plot to a tkinter-compatible format
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        # Pack the canvas into the tkinter window
        canvas.get_tk_widget().pack()
    def classify_material_booking(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")
            return
        supplier_name_code = simpledialog.askstring("Input", "Enter Supplier Name Code:")
        if not supplier_name_code:
            self.result_label.config(text="No supplier name code provided.")
            return
        material_analyzer = MaterialBookingAnalyzer(self.root)
        material_analyzer.analyze_csv(df, supplier_name_code)
    def plot_bar_graph(self, data, title):
        fig, ax = plt.subplots(figsize=(10,6))
        data.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(title)
        ax.set_xlabel('Supplier Name Code')
        ax.set_ylabel('Counts')
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    def plot_pie_chart(self, data, title):
        fig, ax = plt.subplots(figsize=(8, 8))
        data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    def plot_area_chart(self, data, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        data.plot(kind='area', ax=ax, stacked=False)
        ax.set_title(title)
        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Counts')
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    def plot_trip_graphs(self, trips_month, market_title, regular_title):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        if not trips_month.empty:
            trips_month[trips_month['Market/Regular'] == 'Market']['customerNameCode'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', title=market_title)
            trips_month[trips_month['Market/Regular'] == 'Regular']['customerNameCode'].value_counts().plot(kind='bar', ax=axes[1], color='salmon', title=regular_title)
        else:
            axes[0].set_title('Market Trips - No bookings')
            axes[1].set_title('Regular Trips - No bookings')
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    def create_clustering_input_fields(self):
        self.year_column_label = tk.Label(self.input_frame, text="Enter the column name for BookingID_Date:")
        self.year_column_label.pack()
        self.year_column_entry = tk.Entry(self.input_frame)
        self.year_column_entry.pack()
        self.supplier_column_label = tk.Label(self.input_frame, text="Enter the column name for supplierNameCode:")
        self.supplier_column_label.pack()
        self.supplier_column_entry = tk.Entry(self.input_frame)
        self.supplier_column_entry.pack()
        self.location_column_label = tk.Label(self.input_frame, text="Enter the column name for Origin_Location:")
        self.location_column_label.pack()
        self.location_column_entry = tk.Entry(self.input_frame)
        self.location_column_entry.pack()
        self.kmeans_button = tk.Button(self.input_frame, text="Perform KMeans Clustering", command=self.perform_kmeans_clustering)
        self.kmeans_button.pack()
    def perform_kmeans_clustering(self):
        year_column = self.year_column_entry.get()
        supplier_column = self.supplier_column_entry.get()
        location_column = self.location_column_entry.get()
        if not all([year_column, supplier_column, location_column]):
            tk.messagebox.showerror("Error", "Please fill in all input fields.")
            return 
        # Call the kmeans_clustering method with provided column names
        self.kmeans_clustering(year_column, supplier_column, location_column)
    def kmeans_clustering(self, year_column, supplier_column, location_column):
        # Load the CSV file
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            result_label.config(text=f"Error: {e}")
            return
        # Clean up column names
        df.columns = df.columns.str.strip()
        # Extract year from BookingID_Date
        df['Year'] = pd.to_datetime(df[year_column]).dt.year
        # Encode categorical columns
        le = LabelEncoder()
        df['Supplier_Code'] = le.fit_transform(df[supplier_column])
        # Handle missing values in location column
        df['State'] = df[location_column].apply(lambda x: str(x).lower().split(',')[-1].strip() if pd.notna(x) else '')
        df['State_Code'] = le.fit_transform(df['State']) 
        # Filter data for Market and Regular conditions
        market_df = df[df['Market/Regular'] == 'Market']
        regular_df = df[df['Market/Regular'] == 'Regular']
        # Create separate scatter plots for Market and Regular
        self.plot_cluster(df=market_df, title='Market')
        self.plot_cluster(df=regular_df, title='Regular')
    def plot_cluster(self, df, title):
        # Create a new Tkinter window
        window = tk.Toplevel(root)
        window.title(f"KMeans Clustering - {title}")
        # Create subplots for Market and Regularfig, ax = plt.subplots(figsize=(30,7))
        # Perform KMeans clustering
        n_clusters = 3 # Example number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df[['Supplier_Code', 'State_Code']])
        # Add clusters to DataFrame
        df['Cluster'] = clusters
        # Filter top 15 supplierNameCode when Market/Regular is "Regular"
        if title == "Regular":
            top_suppliers = df['supplierNameCode'].value_counts().nlargest(15).index
            df = df[df['supplierNameCode'].isin(top_suppliers)]
        # Visualize clustering
        fig, ax = plt.subplots(figsize=(20,15))
        scatter = ax.scatter(df['supplierNameCode'], df['State'], c=df['Cluster'], cmap='plasma', s=50)
        ax.set_title(f'KMeans Clustering - {title}')
        ax.set_xlabel('Supplier Name Code')
        ax.set_ylabel('State')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        ax.tick_params(axis='x', rotation=45)
        # Embed plot in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()
class MaterialBookingAnalyzer:
    def __init__(self, root):
        self.root = root
    def analyze_csv(self, df, supplier_name_code):
        supplier_data = df[df['supplierNameCode'] == supplier_name_code]
        if supplier_data.empty:
            print("No data found for the entered supplier name code.")
            return
        booking_counts = defaultdict(int)
        material_shipped = defaultdict(int)
        month_counts = defaultdict(int)
        for index, row in supplier_data.iterrows():
            booking_date = datetime.strptime(row['BookingID_Date'], "%d-%m-%Y")
            month = booking_date.strftime("%B")
            customer_name = row['customerNameCode']
            material = row['Material Shipped']
            booking_counts[customer_name] += 1
            material_shipped[material] += 1
            month_counts[month] += 1
        max_customer = max(booking_counts, key=booking_counts.get)
        max_material = max(material_shipped, key=material_shipped.get)
        max_month = max(month_counts, key=month_counts.get)
        result = f"The customer {max_customer} booked the most materials ({max_material}) and the supplier had the most bookings in {max_month}."
        self.display_result(result)
        self.plot_bar_graph(material_shipped, "Material Shipped", "Count of Bookings", "Bookings by Material")
    def display_result(self, result):
        result_window = tk.Toplevel(self.root)
        result_window.title("Material Booking Analysis Result")
        result_window.geometry("600x400")
        result_label = tk.Label(result_window, text=result)
        result_label.pack()
    def plot_bar_graph(self, data, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(20,15)) # Increase width and height
        labels = list(data.keys())
        counts = list(data.values())
        # Generate distinct colors for each bar
        if len(labels) > 10: # If more than 10 labels, use a large pool of colors
            random.seed(0)
            colors = random.sample(list(mcolors.CSS4_COLORS.values()), len(labels))
        else:
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'lime', 'pink']
        # Plotting a horizontal bar chart
        ax.barh(labels, counts, color=colors[:len(labels)])
        # Setting labels and title
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        ax.set_title(title)
        # Adjust the layout to make room for labels
        plt.tight_layout()
        # Create a new tkinter window to display the graph
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Material Booking Analysis Graph")
        # Convert the plot to a tkinter-compatible format
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        # Pack the canvas into the tkinter window
        canvas.get_tk_widget().pack()
root = tk.Tk()
app = AnalysisApp(root)
root.mainloop()
