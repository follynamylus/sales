#***************************************IMPORT PACKAGES*************************************************

import streamlit as st #<-----------------------Import Streamlit for building the web application
import statsmodels.api as sm #<-----------------Import Stats model for making the statistical forecast
import pandas as pd #<--------------------------Import Pandas for data preprocessing
import plotly.express as px  #<-----------------Import  Plotly Express for data visualization
import joblib

#***************************************BUILDING THE INPUT WIDGETS, TABS AND SIDEBAR***************************************
tab_2, tab_1,tab_3 = st.tabs(['DATAFRAME AND DOWNLOAD','VIEW PREDICTION','ABOUT APPLICATION'])#<--- Creation of three tabs

tab_2.title("SALES VISUALIZATION") #<------ Title for First Tab
tab_1.title("SALES PREDICTION") #<--------- Title for Second Tab
tab_3.title("ABOUT APPLICATION") #<--------------------Title for Third Tab

pred_type = st.sidebar.selectbox("Select for prediction type, Either Single or Multiple",('single','multiple'))# Widget for prediction type
if pred_type == 'multiple' : # <------------------ Condition for multiple predictions
    start = st.sidebar.date_input("Input the date from 2023 as start date") #<---------- Input Widget for the start date
    start_date = pd.to_datetime(start) # <-------------------------------------- Convert Start date to a date time format
    end = st.sidebar.date_input("Input the end date later than the start date") #<---------- Input widget for the end date
    if end == start_date : # <---------------------- Conditional Statement to check if the end date is equal to start date
        option = st.sidebar.selectbox("Tick how you want to forecast", ('forward','backward')) # Code to select prediction option.
        if option == 'forward' : # <---------------- Condition for forward prediction.
            steps = int(st.sidebar.number_input("Input The days to extend to", 0)) # <----- Number of days to predict forward
            end_date = start_date + pd.DateOffset(days = steps) # <----- Set end date for forward prediction
        elif option == 'backward' : # <-------------- Condition for backward prediction
            steps = int(st.sidebar.number_input("Input The days to extend from, not later than 2000 days", 0)) # Days to predict backward
            start_date = start_date - pd.DateOffset(days = steps) # Start date for backward prediction
            end_date = pd.to_datetime(end) # <------------------- End date for backward prediction
    else : # <------------------------- If end and start date are not the same
        end_date = pd.to_datetime(end) # <------------------ Set end date
else : # <------------------- Condition if not a multiple prediction
    date = st.sidebar.date_input("Input the date to predict") # <----------- Date input for a single prediction
    start_date = date # <------------------ Start date for a single prediction
    end_date = date # <----------------- End date for a single prediction

# ************************************************ BACKEND CODES ******************************************

df = pd.read_csv("Sales_data.csv")
df['Sales'] = df['Sales'].shift(7)
df.dropna(inplace=True)
df = df.sort_values('Date')
data = df.groupby('Date')['Sales'].sum().reset_index()
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index, infer_datetime_format=True)
data = data['Sales'].resample('MS').mean()
@st.cache_data
def load_model(file_name) : # <---------- Define the function.
    '''
    The Load model function loads pickled statistical models in the script. 
    It takes in file name as parameter.
    It returns the loaded model.
    '''
    return sm.iolib.smpickle.load_pickle(file_name) # <------------- Function return

model = sm.tsa.statespace.SARIMAX(data,order=(1,1,1),seasonal_order=(1,1,0,12),enforce_invertibility=False)
results = model.fit() # <-------------------- Load the temperature model
@st.cache_resource
def Forecast(results, start, end) : # <--------- Forecast function definition
    '''
    The Forecast function performs the tasks of making forecast/prediction.
    It create a dataframe from the predictions with flexible columns depending on the number of multiple choice with date as the index column.
    It takes three input parameters which are a model for prediction, a start date and an end date.
    It returns a dataframe
    '''
    df = pd.DataFrame() # <------------ Create an empty dataframe
    pred = results.get_prediction(start = pd.to_datetime(start), end = pd.to_datetime(end)) # Make prediction
    mean_disp = pred.predicted_mean # <---------- Create series from predicted values
    mean_df = mean_disp.to_frame() # <----------- Convert series to data frame
    mean_df.reset_index(inplace = True) # <------- Reset the data frame's index
    mean_df.columns = ['Date','sales'] # <-------- Rename the dataframe's column
    mean_df['Month'] = pd.to_datetime(mean_df['Date']).dt.month_name() # Create additional month column from the date column
    if df.empty == True : # <----------- Condition to check for empty dataframe
        mean_df['Month'] = pd.to_datetime(mean_df['Date']).dt.month_name() # <----- Create month column for empty dataframe
        df['Date'] = mean_df['Date'] # <-------- Add date column to the dataframe
        df['Sales (Naira)'] = mean_df['sales'] # <--------- Add temperature column to the data frame
    else : # <----- Condition if dataframe is not empty
        df['Sales (Naira)'] = mean_df['sales'] # <---------- Add temperature column to the data frame
    
    
    return df # <------------------ Return the dataframe.

def Plots(dataframe) :
    """
    It creates four plots which include line plot, bar plot, area plot and density contour plot from a data frame
    It takes in a data frame as a parameter
    It has no return value.
    """
    with st.expander("Click to view the line plot") : # <-------------- Create an expander for line plot
        st.write("The Line plot for the sales") # <--------- Write in the created expander
        fig = px.line(df, x= 'Date', y= df['Sales (Naira)']) # <--- Create a line plot of all weather features in the dataframe against the date
        st.plotly_chart(fig, use_container_width=True) # <------------- Fit the plot to the web page
    with st.expander("Click to view the grouped bar chart") : # <----------- Create an expander for grouped bar chart
        st.write("The Group bar chart for the sales") # <--------- Write in the created expander
        fig1 = px.bar(df, x= 'Date', y= df['Sales (Naira)'], barmode= 'group') #Create a  bar plot of all weather features in the dataframe against the date
        st.plotly_chart(fig1, use_container_width=True) 
    with st.expander("Click to view the Area plot") : # <-------------- Create an expander for Area plot
        st.write("The Area plot for the sales") # <--------- Write in the created expander
        fig = px.area(df, x= 'Date', y= df['Sales (Naira)']) # <--- Create a Area plot of all weather features in the dataframe against the date
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Click to view the Density_Contour plot") : # <-------------- Create an expander for Density contour plot
        st.write("The Density Contour plot for the sales") # <--------- Write in the created expander
        fig = px.density_contour(df, x= 'Date', y= df['Sales (Naira)']) # Create a Density Contour plot of all weather features in the dataframe against the date
        st.plotly_chart(fig, use_container_width=True)

def Single_pred(dataframe) : # <---------------- Function definition
    """
    The Single_pred function is used to view single prediction results for all the weather factors.
    The values are displayed as the user choose.
    It takes in dataframe as parameter
    It gives no return value
    """
    st.subheader("Date") # <----------------------- Code for subheader
    st.write(f"The date is {df['Date'][0]}") # <--------------------- Code to output the date
    st.write(f'The sales value is {round(df["Sales (Naira)"][0],4)} naira') # Code to output the temperature value


#******************************************************* OUTPUT FRONTEND CODES *********************************************************

with tab_1 : # <------------------------- Declare tab 1 container
    if pred_type == 'single' : # <-------------- code to check for single prediction
        df = Forecast(results, start_date, end_date) # <---------------- Call Forecast function in tab 1. 
        Single_pred(df) # <---------------------- Call to display single prediction values
    else : # <-------------------- Code if not single prediction
         if start_date > end_date : # <-------------- Condition if start date is greater than the end date
            df = Forecast(results, end_date, start_date) # <---------------- Call Forecast function
            Plots(df) # <------------- Call for plots
         else :
            df = Forecast(results, start_date, end_date) # <---------------- Call Forecast function
            Plots(df) # <------------- Call for plots

tab_2.dataframe(df) # <--------------- Display the dataframe on tab2
@st.cache_data # <------------- IMPORTANT: Cache the conversion to prevent computation on every rerun

def convert_df(df): # <--------------- Function declaration
    '''
    Convert_df function converts the resulting dataframe to a CSV file.
    It takes in a data frame as a aprameter.
    It returns a CSV file
    '''
    
    return df.to_csv().encode('utf-8') # <--------------- Return dataframe as a CSV file
csv = convert_df(df) # <------------ Convert_df function calling and assigning to a variable.
tab_2.success("Print Result as CSV file") # <--------------- A widget as heading for the download option in tab 2
tab_2.download_button("Download",csv,"Prediction.csv",'text/csv') # <------------------ Download button widget in tab 2.

tab_3.write(
    """
    The Web Application is used for making sales forecasts / predictions of a fast food enterprise.

    The web application makes use of statistical models trained using a data modeled on Item-7 sales from 1st of January 2015 till 31st if December 2022.
    The data which are aggregated to the average monthly value for the weather features.

    The application consists of an adjustable Sidebar for Input widgets,
    Visualizations are provided on the first tab while data frame of the predictions and download option are provided on the second tabs of the application
    """
)
tab_3.subheader("ABOUT THE FEATURES")
tab_3.write(
    '''
    The columns is, Sales with Naira as its denomination,
    
    '''
)
tab_3.subheader("ABOUT INPUT WIDGETS")
tab_3.write(
    """
    The Input widgets consist of date input type which is in a calender format with a drop down comprising the year, month and days in the months , 
    they are considered as the start date and end date , the date widgets have a default value of the day's date. 

    There is a widget to select the prediction type either single or multiple.

     There is the number input widget that serves as an alternative when considering number of days to forecast / predict ,it is only provided
     when the start date and end date is the same ,it has a default value of zero.

     There is a widget to input date for single prediction.

    The option widget if a select box dropdown, with option to choose for either a forward or backward prediction. Its default is forward .
    

    All the input widgets are contained in the sidebar
    """
)
tab_3.subheader("ABOUT OUTPUT WIDGETS")
tab_3.write(
    """
    The Output widgets are in tabs .

    Tab 1  the PREDICTION AND DOWNLOAD tab, It displays the prediction dataframe and a download button option for downloading the file in a 
    CSV format.

    Tab 2 the VISUALIZATION tab contains graphs and plots of the predictions. The graphs are in expanders that include the Line plot , Bar chart ,
    stacked Area plot and stacked Density Contour plot . These variables are flexible and multiple can be selected at a time,
     it also can be activated or deactivated by clicking on their names by the top
    right corner of Visualization tab, all these are for multiple predictions.

    Tab 2 for single prediction displays values for the date and the date the sales were made.


    Tab 3 ABOUT APPLICATION tab, contains information on the application widgets .
    """
)