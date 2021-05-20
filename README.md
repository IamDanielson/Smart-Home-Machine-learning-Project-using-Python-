# ---------------------------------------- Combined Machine Learning Model ---------------------------------------------

# This Script combines the function of the HAR and UBA within a single layout. The system first runs the HAR then stores
# Output from the HAR in a list variable that always shifts the 1st previous and 2nd previous predicted activity
# accordingly to fit the input for the UBA. The Time date feature is also appended to this list variable to have a
# Complete input variable for the UBA. Input variable is then plugged into UBA model to predict anomaly.

# THE DATA-SET FOR TRAINING THE MODELS

har_trainData = {'ms00': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'ms01': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'ms06': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'ms07': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'ms04': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 'ms05': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 'ms02': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                 'ms03': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 'ms08': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 'Ts09': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                 'Ws10': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 'ms11': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 'Activity': ['hex', 'hen', 'kex', 'ken', 'ren', 'rex', 'sng', 'eng', 'bng', 'cng', 'dng', 'lng',
                              'cay']}

uba_trainData = {'day': ['Sunday', 'Sunday', 'Sunday', 'Sunday', 'Sunday', 'Monday', 'Monday', 'Monday', 'Monday',
                         'Monday', 'Tuesday', 'Tuesday', 'Tuesday', 'Tuesday', 'Tuesday', 'Wednesday', 'Wednesday',
                         'Wednesday', 'Wednesday', 'Wednesday', 'Thursday', 'Thursday', 'Thursday', 'Thursday',
                         'Thursday', 'Friday', 'Friday', 'Friday', 'Friday', 'Friday', 'Saturday', 'Saturday',
                         'Saturday', 'Saturday', 'Saturday'],
                 'hour': [0, 3, 7, 9, 16, 1, 5, 8, 11, 18, 1, 4, 8, 16, 21, 0, 4, 9, 18, 22, 1, 6, 8, 19, 23, 1, 5,
                          9, 19, 23, 0, 4, 10, 15, 20],
                 'c_act': ['DNG', 'SNG', 'HEX', 'HEN', 'KEX', 'DNG', 'LNG', 'ENG', 'HEX', 'HEN', 'SNG', 'REX', 'CAY',
                           'HEX', 'KEN', 'REN', 'SNG', 'HEX', 'CAY', 'KEX', 'REX', 'KEN', 'ENG', 'KEX', 'DNG', 'SNG',
                           'BNG', 'HEX', 'CAY', 'REX', 'DNG', 'SNG', 'REN', 'CNG', 'KEX'],
                 'p1_act': ['KEX', 'LNG', 'CAY', 'HEX', 'KEN', 'HEN', 'REN', 'KEX', 'REX', 'HEX', 'DNG', 'BNG', 'HEN',
                            'DNG', 'ENG', 'SNG', 'KEX', 'CAY', 'REX', 'CNG', 'LNG', 'DNG', 'KEX', 'CNG', 'HEN', 'DNG',
                            'LNG', 'KEX', 'REX', 'BNG', 'CAY', 'DNG', 'HEN', 'REN', 'KEN'],
                 'p2_act': ['BNG', 'REN', 'KEX', 'KEX', 'ENG', 'ENG', 'KEX', 'CNG', 'CAY', 'DNG', 'KEX', 'REN', 'LNG',
                            'BNG', 'HEX', 'ENG', 'KEN', 'DNG', 'LNG', 'KEN', 'REN', 'KEN', 'CNG', 'KEN', 'HEX', 'KEX',
                            'REN', 'REX', 'REN', 'LNG', 'ENG', 'KEX', 'SNG', 'DNG', 'HEN'],
                 'outcome': ['no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no',
                             'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no',
                             'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes']
                 }
# IMPORTING LIBRARIES

import datetime as dt                                                   # Library for checking date and time of the day.
import calendar                                         # Library for utilizing calendar features, such as weekday name.
from sklearn import preprocessing                                                           # importing label encoder.
from sklearn.naive_bayes import GaussianNB                                      # import the Gaussian Naive Bayes Model.

# ENCODING THE NON-NUMERICAL FEATURE OF HAR & UBA TRAINING DATA
newForm = preprocessing.LabelEncoder()                                                          # creating label encoder

Activity_ecd = newForm.fit_transform(har_trainData['Activity'])         # converting activity string labels into numbers

en_day = newForm.fit_transform(uba_trainData['day'])                                  # converting weekdays into numbers
en_c_act = newForm.fit_transform(uba_trainData['c_act'])                   # converting c_act string labels into numbers
en_p1_act = newForm.fit_transform(uba_trainData['p1_act'])                # converting p1_act string labels into numbers
en_p2_act = newForm.fit_transform(uba_trainData['p2_act'])                # converting p2_act string labels into numbers
en_outcome = newForm.fit_transform(uba_trainData['outcome'])             # converting outcome string labels into numbers

# COMBINING FEATURES OF HAR TRAIN DATA INTO WORK TABLE

har_features = zip(har_trainData['ms00'], har_trainData['ms01'], har_trainData['ms06'], har_trainData['ms07'],
                   har_trainData['ms04'], har_trainData['ms05'], har_trainData['ms02'], har_trainData['ms03'],
                   har_trainData['ms08'], har_trainData['Ts09'], har_trainData['Ws10'], har_trainData['ms11'])
# Converts har_features into list variable
har_features = list(har_features)

# COMBINING FEATURES OF UBA TRAIN DATA INTO WORK TABLE

uba_features = zip(en_day, uba_trainData['hour'], en_c_act, en_p1_act, en_p2_act)
# Converts uba_features into list variable
uba_features = list(uba_features)

# CREATING & RUNNING THE HAR MODEL

harModel = GaussianNB()                                               # create a Gaussian Naive Bayes classifier for HAR
harModel.fit(har_features, Activity_ecd)                                  # feeding data set into the model for training

# CREATING & RUNNING THE UBA MODEL

ubaModel = GaussianNB()                                               # create a Gaussian Naive Bayes classifier for UBA
ubaModel.fit(uba_features, en_outcome)                                    # feeding data set into the model for training

# INITIALISING SENSOR VARIABLES FOR HAR INPUT
MS00, MS01, MS02, MS03, MSO4, MSO5, MS06, MS07, MSO8, TS09, WS10, MS11 = 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0

# INITIALISING VARIABLES FOR UBA INPUT
currentDayVal, currentHour, currentAct, p1Act, p2Act = 3, 8, 5, 6, 5
interrupt = 0

# PROGRAM LOOP
while 1 == 1:
    heat, water = 0, 0,
    anyPin = int(input("which state do you state you need"))                           # SIMULATING AN INTERRUPT TRIGGER
    if heat == 1 or water == 1 or anyPin == 1:
        interrupt = 1
    else:
        interrupt = 0

    # Combining sensor inputs in a list
    har_input = [MS00, MS01, MS02, MS03, MSO4, MSO5, MS06, MS07, MSO8, TS09, WS10, MS11]

    while interrupt == 1:
        harPrediction = harModel.predict([har_input])  # Run the model to predict an outcome
        harPrediction = int(list(harPrediction)[0])

        # Generating current weekday name and hour of the day
        currentDay = calendar.day_name[dt.datetime.now().weekday()]
        weekdayChart = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        currentDayVal = weekdayChart.index(currentDay)
        currentHour = dt.datetime.now().hour

        # getting values for input to the UBA

        p2Act = p1Act
        p1Act = currentAct
        currentAct = harPrediction

        uba_input = [currentDayVal, currentHour, currentAct, p1Act, p2Act]      # putting values in a list for UBA input
        uba_Prediction = ubaModel.predict([uba_input])                             # Run the model to predict an outcome
        uba_Prediction = int(list(uba_Prediction)[0])

        if uba_Prediction == 0:
            print("Normal, no cause for alarm")                                                  # SIMULATING A RESPONSE
        else:
            print("Abnormal, security threat !!!")                                               # SIMULATING A RESPONSE
        interrupt = 0
