# Natural Language Processing

# Importing the dataset
dataset_orig = read.delim('Data\\Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the text
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_orig$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, sparse = 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$liked = dataset_orig$Liked

# Encoding target feature as a factor
dataset$liked = factor(dataset$liked, levels = c(0,1))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$liked, SplitRatio = 0.8)
training_set = subset(x = dataset, split == TRUE)
test_set = subset(x = dataset, split == FALSE)

# Build Random Forest Classification model (RF is used often with NLP)
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$liked,
                          ntree = 10)

# Predicting the Test Set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)


