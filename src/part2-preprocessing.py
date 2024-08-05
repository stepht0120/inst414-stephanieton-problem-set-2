'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

# loading in data
pred_universe_raw = pd.read_csv('data/pred_universe_raw.csv')
arrest_events_raw = pd.read_csv('data/arrest_events_raw.csv')

# perform a full outer join/merge on 'person_id'
df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer')

# create a column 'y' which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date
df_arrests['y'] = df_arrests.apply(
    lambda row: any((row['arrest_date_event'] > row['arrest_date_univ']) & 
                    (row['arrest_date_event'] <= row['arrest_date_univ'] + pd.Timedelta(days=365)) & 
                    (row['charge_category'] == 'felony')),
    axis=1
)

# what share of arrestees were rearrested for a felony crime in the next year?
share_rearrested_felony = df_arrests['y'].mean()
print("What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year?")
print(share_rearrested_felony)

# create a predictive feature 'current_charge_felony' which equals 1 if the current arrest was for a felony charge, and 0 otherwise
df_arrests['current_charge_felony'] = df_arrests['charge_category'].apply(lambda x: 1 if x == 'felony' else 0)

# What share of current charges are felonies?
share_current_felony = df_arrests['current_charge_felony'].mean()
print("What share of current charges are felonies?")
print(share_current_felony)

# Create a predictive feature 'num_fel_arrests_last_year' which is the total number arrests in the one year prior to the current charge
def count_fel_arrests_last_year(row):
    arrest_date = row['arrest_date_univ']
    person_id = row['person_id']
    past_arrests = df_arrests[(df_arrests['person_id'] == person_id) & 
                              (df_arrests['arrest_date_event'] < arrest_date) & 
                              (df_arrests['arrest_date_event'] >= arrest_date - pd.Timedelta(days=365)) & 
                              (df_arrests['charge_category'] == 'felony')]
    return len(past_arrests)

df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(count_fel_arrests_last_year, axis=1)

# What is the average number of felony arrests in the last year?
avg_fel_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
print("What is the average number of felony arrests in the last year?")
print(avg_fel_arrests_last_year)

# Print the mean of 'num_fel_arrests_last_year'
print("Mean of 'num_fel_arrests_last_year':", df_arrests['num_fel_arrests_last_year'].mean())

# Print pred_universe head
print(pred_universe_raw.head())

# Save df_arrests as a CSV for use in PART 3
df_arrests.to_csv('data/df_arrests.csv', index=False)

# Return df_arrests for use in main.py
df_arrests



