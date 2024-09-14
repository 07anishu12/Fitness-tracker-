import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor


#read the file 

df= pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_columns = list(df.columns[:6])
#plotting outliers

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]= 100

df[["acc_x","label"]].boxplot(by="label",figsize=(20,10))
df[["acc_y","label"]].boxplot(by="label",figsize=(20,10))
df[["gyr_x","label"]].boxplot(by="label",figsize=(20,10))
df[["gyr_y","label"]].boxplot(by="label",figsize=(20,10))


df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))


# Function to plot the outliers and non-outliers
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    # Drop rows with missing values in the specified columns
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    
    # Ensure the outlier column is boolean
    dataset[outlier_col] = dataset[outlier_col].astype("bool")
    
    # Reset index if requested
    if reset_index:
        dataset = dataset.reset_index(drop=True)
    
    fig, ax = plt.subplots()
    
    plt.xlabel("Samples")
    plt.ylabel("Value")
    
    # Plotting non-outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    
    # Plotting outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )
    
    # Adding legend
    plt.legend(
        ["No Outlier - " + col, "Outlier - " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True
    )
    
    # Display the plot
    plt.show()


#insert iqr function

def mark_outliers_iqr(dataset, col):
    dataset = dataset.copy()
    
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    dataset[col + "_outlier"] = (
        (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
    )
    
    return dataset


#plot a single column

col= "acc_x"
dataset = mark_outliers_iqr(df,col)

plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)


for col in outlier_columns:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)

#checking for the normal distribution

df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3,3))


#chauvenet function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

#loop over all the columns


for col in outlier_columns:
    dataset= mark_outliers_chauvenet(df,col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
    
#local outlier factor (distance based) 

#lof function

def mark_outliers_lof(dataset,columns,n=20):
    dataset=dataset.copy()
    
    lof= LocalOutlierFactor(n_neighbors=n)
    data= dataset[columns]
    outliers= lof.fit_predict(data)
    X_scores= lof.negative_outlier_factor_
    
    dataset["outlier_lof"]=outliers==-1
    return dataset, outliers, X_scores


dataset, outliers, X_scores= mark_outliers_lof(df,outlier_columns)
for col in outlier_columns:    
    plot_binary_outliers(dataset=dataset,col=col,outlier_col="outlier_lof",reset_index=True)
    

#check outliers grouped by labels

label= "bench"
for col in outlier_columns:
    dataset= mark_outliers_iqr(df[df["label"]== label], col)
    plot_binary_outliers(dataset, col, col+"_outlier", reset_index=True)
    
for col in outlier_columns:
    dataset= mark_outliers_chauvenet(df[df["label"]== label], col)
    plot_binary_outliers(dataset, col, col+"_outlier", reset_index=True)
    
dataset, outliers, X_scores= mark_outliers_lof(df[df["label"]== label],outlier_columns)
for col in outlier_columns:    
    plot_binary_outliers(dataset=dataset,col=col,outlier_col="outlier_lof",reset_index=True)
    
df


#testing on single column

col = "gyr_z"
dataset= mark_outliers_chauvenet(df,col=col)
dataset[dataset["gyr_z_outlier"]]
dataset.loc[dataset["gyr_z_outlier"],"gyr_z"] = np.nan

#create a loop 

outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset= mark_outliers_chauvenet(df[df["label"] == label], col)
        #replacing values marked as outliers with nan
        dataset.loc[dataset[col + "_outlier"],col] = np.nan
        
        #update the column in the original dataframe
        
        outliers_removed_df.loc[(outliers_removed_df["label"]== label), col]= dataset[col]
        
        n_outliers = len(dataset)- len (dataset[col].dropna())
        print(f"Removed {n_outliers} from { col} for {label}")


outliers_removed_df.info()


# export the dataframe

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")