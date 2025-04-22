# The way to run the application will be:
1. Cleanse the data - To make sure our model gets the correct data to train and test it. 
    - Make sure you have your data file in the /data folder present inside the sklearn-env
    - What do we do in data cleansing?
        - 1. filling columns NaN values
        - 2. cleaning and converting RSSI and Timestamp specifically
        - 3. convert mac address to int
        - 4. cleaning and converting column types
        - 5. save the cleaned data to the script dir
    - How to cleansed the collection csv data file
        - cd sklearn-env
        - .\ble-env\Scripts\activate
        - pipenv shell
        - python  .\src\cleaningdata.py --csv .\data\<your-file.csv>
2. Read the cleansed data
    - Now we will run the cleansed csv file to feature engineering
    - How to perform feature engineering
       - cd sklearn-env
       - 
3. Store the output of the model
4. 