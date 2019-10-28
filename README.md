# Market Value Prediction of House

The objective of the project is to develop a model to predict a home’s market value. The work is implemented in Python. 

### Dependencies

The following publicly available packages are used in the implementation:

•	Numpy and Pandas for data manipulation

•	Matplotlib for visualization

•	Sklearn as the machine learning library

•	Geopy for calculating distances between GPS coordinates 

### Data Preparation

The training dataset contained 11588 data records of home sale transactions in 2015 for a county in Washington. 
The data contains 24 attributes of the transactions, including the dollar amount of the sale.  The dollar amount of the sale, named "SaleDollarCnt", is the output variable and the goal of the ML regression model was to predict the SaleDollarCnt given the other 23 attributes of the transaction. Data preparation entailed the following tasks: 1) handling missing values, 2) handling corrupted data records and outliers, and 3) engineering features. 

#### Missing values

The training dataset contained missing values for the fields given in Table 1 . For GarageSquareFeet and ViewType, the absence of a numerical value (NA) was replaced with 0s. In GarageSquareFeet, 0 indicates the absence of a garage. In ViewType, 0 indicates the absence of a view. 

BGMedHomeValue, BGMedRent, BGMedYearBuilt contain missing values. The BG (blockgroup) information is missing. So these features are mean/mode imputed. To give the model an indication about imputed rows, a binary column is added. 

#### Outliers

Houses with a price outside the "three-sigma" range is removed. Other attributes can be used too to remove outliers. 

#### Feature Engineering

Price per square feet, distance to the city center, HomeAge and whether the house has a view and a garage were added. 

Categorical Variables handled with One-hot encoding (low cardinality) and target encoding (high cardinality).

### Model

In this code a Random Forest model is used. A tree based method is used given the categorical variables. 




