import pandas as pd
import numpy as np

def create_features(temp,humidity,air_quality,dust):

    input_df=pd.DataFrame([{

        "Temperature":temp,

        "Humidity":humidity,

        "AirQuality":air_quality,

        "Dust":dust,

        "Temp_Hum_Index":
        temp*humidity/100,

        "Air_Dust_Product":
        air_quality*dust,

        "AirQuality_Log":
        np.log1p(air_quality),

        "Dust_Log":
        np.log1p(dust),

        "Air_Dust_Ratio":
        air_quality/(dust+1)

    }])

    return input_df