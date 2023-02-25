#%%
import functools as ft
import pandas as pd
import os
import itertools
from sklearn.cluster import KMeans

# dirname=os.path.dirname(__file__)
# filepath=os.path.join(dirname,filepath)
def fill_monthly_demand_data(demand_df, dt):
    """this function fills daily data from demand data with monthly values only"""
    hour = dt.hour + 1
    month = dt.month
    return demand_df.loc[hour, month]


def get_demand_monthly(filepath):
    """this converts the monthly demand data into daily"""
    dirname = os.path.abspath("")
    filepath = os.path.join(dirname, filepath)
    demand_df = pd.read_excel(filepath)
    demand_df.set_index("HOURS", inplace=True)
    year = filepath[-9:-5]
    earliest, latest = f"{year}-01-01T00:00", f"{year}-12-31T23:00"
    dti = pd.date_range(earliest, latest, freq="H")
    demand_df_formatted = pd.DataFrame(columns=["datetime", "value"], index=dti)
    demand_df_formatted["datetime"] = demand_df_formatted["datetime"].index
    demand_df_formatted["demand"] = demand_df_formatted["datetime"].apply(
        lambda x: fill_monthly_demand_data(demand_df, x)
    )
    return demand_df_formatted


def get_supply(filepath, format="%m/%d/%Y %H:%M"):
    """this gets the supply values and formats it"""
    supply_df = pd.read_csv(filepath)
    supply_df.drop(columns=["local_time"], inplace=True)
    supply_df["datetime"] = pd.to_datetime(
        supply_df["datetime"],
    )
    return supply_df


def fill_daily_demand_data(demand_df, dt):
    hour = dt.hour
    day = dt.day
    month = dt.month
    for i in range(10):
        if pd.notnull(demand_df[month].loc[hour, day - i]) and pd.notna(
            demand_df[month].loc[hour, day - i]
        ):
            return demand_df[month].loc[hour, day - i]


def get_demand_daily():
    """this converts the daily demand data into into one dataframe"""
    year = 2015
    earliest, latest = f"{year}-01-01T00:00", f"{year}-12-31T23:00"
    dti = pd.date_range(earliest, latest, freq="H")
    demand_df_formatted = pd.DataFrame(columns=["datetime", "demand"], index=dti)
    demand_df_formatted["datetime"] = demand_df_formatted["datetime"].index
    month = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    demand_df = {}
    i = 1
    for m in month:
        # filepath = f"demand_data_formatted\\2015_demand_data\{m}2015.xls"
        dirname = os.path.abspath("")
        # filepath = os.path.join(dirname, filepath)
        filepath = os.path.join(
            dirname, "demand_data_formatted", "2015_demand_data", f"{m}2015.xls"
        )
        demand_df[i] = pd.read_excel(
            filepath, "Totals", skiprows=37, nrows=24, usecols="B:AF"
        )
        i = i + 1
    demand_df_formatted["demand"] = demand_df_formatted["datetime"].apply(
        lambda x: fill_daily_demand_data(demand_df, x)
    )

    return demand_df_formatted


class data_oneyear:
    def __init__(self, demand_df, wind_df, solar_df, max_demand):
        """all data frames will have two columns, datetime and value in addition the in"""
        self.demand_df = demand_df
        self.wind_df = wind_df
        self.solar_df = solar_df
        self.demand_df["demand"] = self.demand_df["demand"] / max_demand
        self.wind_df["electricity"] = (
            self.wind_df["electricity"] / self.wind_df["electricity"].max()
        )
        self.solar_df["electricity"] = (
            self.solar_df["electricity"] / self.solar_df["electricity"].max()
        )

    def write_main_df(self):
        dfs = [self.demand_df, self.wind_df, self.solar_df]
        self.df = ft.reduce(
            lambda left, right: pd.merge(left, right, on="datetime"), dfs
        )
        self.df["date"] = self.df["datetime"].apply(lambda x: x.date())
        self.demand_df_grouped = self.df.groupby("date")["demand"].apply(list)
        self.wind_df_grouped = self.df.groupby("date")["electricity_x"].apply(list)
        self.pv_df_grouped = self.df.groupby("date")["electricity_y"].apply(list)
        dfs = [self.demand_df_grouped, self.wind_df_grouped, self.pv_df_grouped]
        self.dfcomplete = ft.reduce(
            lambda left, right: pd.merge(left, right, on="date"), dfs
        )
        self.dfcomplete["combined"] = self.dfcomplete.values.tolist()
        self.dfcomplete["combined"] = self.dfcomplete["combined"].apply(
            lambda l: list(itertools.chain(*l))
        )
        self.cluster_input = pd.DataFrame(
            self.dfcomplete["combined"].tolist(), index=self.dfcomplete
        )


def mergeall(*dfs):
    return pd.concat(dfs)


#%%
# 2015
demand_df = get_demand_daily()
filepath_pv = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "PV", "PV_baalbeck_2015.csv"
)
filepath_wind = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "Wind", "Wind_Akkar_2015.csv"
)
pv_df = get_supply(filepath_pv)
pv_df.rename(columns={"electricity": "pv"})

wind_df = get_supply(filepath_wind)
wind_df.rename(columns={"electricity": "wind"})

data_2015 = data_oneyear(demand_df, wind_df, pv_df, demand_df["demand"].max())
data_2015.write_main_df()

#%%
# 2016
filepath_demand = os.path.join(
    os.path.abspath(""), "demand_data_formatted", "Average-Demand-2016.xlsx"
)
demand_df = get_demand_monthly(filepath_demand)

filepath_pv = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "PV", "PV_baalbeck_2016.csv"
)
filepath_wind = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "Wind", "Wind_Akkar_2016.csv"
)
pv_df = get_supply(filepath_pv)
pv_df.rename(columns={"electricity": "pv"})

wind_df = get_supply(filepath_wind)
wind_df.rename(columns={"electricity": "wind"})

data_2016 = data_oneyear(demand_df, wind_df, pv_df, demand_df["demand"].max())
data_2016.write_main_df()

#%%
# 2017

filepath_demand = os.path.join(
    os.path.abspath(""), "demand_data_formatted", "Average-Demand-2017.xlsx"
)
demand_df = get_demand_monthly(filepath_demand)

filepath_pv = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "PV", "PV_baalbeck_2017.csv"
)
filepath_wind = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "Wind", "Wind_Akkar_2017.csv"
)
pv_df = get_supply(filepath_pv)
pv_df.rename(columns={"electricity": "pv"})

wind_df = get_supply(filepath_wind)
wind_df.rename(columns={"electricity": "wind"})

data_2017 = data_oneyear(demand_df, wind_df, pv_df, demand_df["demand"].max())
data_2017.write_main_df()

#%%
# 2018
filepath_demand = os.path.join(
    os.path.abspath(""), "demand_data_formatted", "Average-Demand-2018.xlsx"
)
demand_df = get_demand_monthly(filepath_demand)

filepath_pv = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "PV", "PV_baalbeck_2018.csv"
)
filepath_wind = os.path.join(
    os.path.abspath(""), "supply_data_formatted", "Wind", "Wind_Akkar_2018.csv"
)
pv_df = get_supply(filepath_pv)
pv_df.rename(columns={"electricity": "pv"})

wind_df = get_supply(filepath_wind)
wind_df.rename(columns={"electricity": "wind"})

data_2018 = data_oneyear(demand_df, wind_df, pv_df, demand_df["demand"].max())
data_2018.write_main_df()

#%%
# all dataframes
all_data_df = mergeall(
    data_2015.cluster_input,
    data_2016.cluster_input,
    data_2017.cluster_input,
    data_2018.cluster_input,
)
#%%
# clustering
class cluster_data:
    def __init__(self, all_data_df, n_clusters):
        self.df = all_data_df
        self.km = KMeans(
            n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        self.y_km = self.km.fit_predict(self.df)
        self.centroids = self.km.cluster_centers_
        self.inertia = self.km.inertia_


#%%
# finding best number of reprenstative days

inertia_list = []
for n in range(1, 30):
    inertia_list.append(cluster_data(all_data_df, n).inertia)


# %%
import matplotlib.pyplot as plt

plt.plot(range(1, 30), inertia_list)
plt.savefig("inertia.png")
plt.show()

#   def __init__(self, demand_df,wind_df,solar_df):
# all data frames should have 2 columns, date and the corresponding value

#      self.demand=demand_df
#     self.wind=wind_df

#    self.solar=solar_df


# def process_data(self):
#   self.demand.interpolate(method='linear', limit_direction='forward', axis=0,inplace=True)
#  dfs = [self.demand, self.wind,self.solar]
# self.df=ft.reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)

# %%
