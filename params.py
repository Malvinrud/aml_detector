import os
import numpy as np


#### Mostly dummy, left some for orientation ####

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")


##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "aml_detector", "raw_data")

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
