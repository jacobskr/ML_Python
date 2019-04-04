# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data\\Data.csv')

# Taking care of missing data
  # Going to impute mean for missing values in this example
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical variables
dataset$Country = factor(x = dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(x = dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))
