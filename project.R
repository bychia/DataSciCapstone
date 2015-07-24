#########################
# STEP 1: Project Setup
#########################
options(java.parameters = "-Xmx24000m")

#import Library
library(NLP) 
library(openNLP) 
library(tm) 
library(RWeka) 
library(ggplot2) 
library(wordcloud)
library(reshape)
library(stringr)

#set project directory
projectDir <- "~/git/DataSciCapstone"
swiftKeyDir <- "Coursera-SwiftKey/final/en_US/" #we are interested in the english version.
setwd(projectDir)

# Load the data
blogs <- readLines(paste(swiftKeyDir, "en_US.blogs.txt", sep=""))
news <- readLines(paste(swiftKeyDir, "en_US.news.txt", sep=""))
twitter <- readLines(paste(swiftKeyDir, "en_US.twitter.txt", sep=""))

# Save the data
save(blogs, file="blogs.RData")
save(news, file="news.RData")
save(twitter, file="twitter.RData")
load(file = "blogs.RData")
load(file = "news.RData")
load(file = "twitter.RData")

##################################
# STEP 2: Basic Data Exploratory
##################################
exploreDataSet <- function(eachFile, dir) 
{
  fileName <- unlist(strsplit(eachFile,"[.]"))[2]
  fileDir <- paste(dir,eachFile, sep="")
  fileInfo <- file.info(fileDir)
  fileSize <- fileInfo$size / 1024^2
  data <- get(fileName)
  nLength <- length(data)
  
  nChars <- nchar(data)
  totalChars <- sum(nChars)
  maxNChar <- nChars[which.max(nChars)]
  minNChar <- nChars[which.min(nChars)]
  
  nWords <- sapply(strsplit(data, "[ ]"), length)
  totalWords <- sum(nWords)
  maxNWords <- nWords[which.max(nWords)]
  minNWords <- nWords[which.min(nWords)]
  
  return(c(fileName, eachFile, round(fileSize, digits=2), nLength, totalChars, maxNChar, minNChar, totalWords, maxNWords, minNWords));
}

lsDataExplored <- lapply(list.files(path = swiftKeyDir), exploreDataSet, dir=swiftKeyDir)
dfDataExplored <- data.frame(matrix(unlist(lsDataExplored), nrow=NROW(lsDataExplored), byrow=TRUE))
colnames(dfDataExplored) <- c("data","file","filesize(MB)","num.of.Lines", "num.of.Chars", "max.Chars.in.a.line", "min.Chars.in.a.line", "total.Words", "max.Words.in.a.line", "min.Words.in.a.line")

dfSubDataExplored <- subset.data.frame(x=dfDataExplored, select=c(data,num.of.Lines,num.of.Chars,total.Words))

# melt the data frame for plotting
dfSubDataExplored.m <- melt(dfSubDataExplored, id.vars='data')

# Plot the information about the dataset files. 
ggplot(dfSubDataExplored.m, aes(data, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") 
+ ggtitle("Overall Exploratory View of Dataset")

# remove data
rm(blogs,news,twitter)


###########################
# STEP 3: Data Processing
###########################
# Break the big data into smaller size
tinyDataGenerator <- function(dVec, dataName, start, size){
  dataSize <- length(dVec)
  limit <- start*size
  start <- (limit -size)+1
  if(limit > dataSize && start < dataSize){
    limit <- dataSize
  }else if(limit > dataSize && start < dataSize){
    print("No more dVec")
    return
  }
  dVec[start:limit]
}

# Process vector characters by converting to corpus. Output to Data Frame.
corpus2DataFrame <- function(dVec){
  options(mc.cores=6)
  
  corpus <- VCorpus(VectorSource(dVec)) # Building the main corpus
  corpus <- tm_map(corpus, removeNumbers) # removing numbers
  corpus <- tm_map(corpus, stripWhitespace) # removing whitespaces
  corpus <- tm_map(corpus, content_transformer(tolower)) #lowercasing all contents
  
  remove.slang <- content_transformer(function(x, pattern, replaceWith) gsub(pattern, replaceWith, x))
  corpus <- tm_map(corpus, remove.slang, "'ve", " have")
  corpus <- tm_map(corpus, remove.slang, "'s", " is")
  corpus <- tm_map(corpus, remove.slang, "'m", " am")
  corpus <- tm_map(corpus, remove.slang, "'ll", " will")
  corpus <- tm_map(corpus, remove.slang, "'re", " are")
  corpus <- tm_map(corpus, remove.slang, "can't", "can not")
  corpus <- tm_map(corpus, remove.slang, "n't", " not")
  corpus <- tm_map(corpus, remove.slang, "'d", " would")
  corpus <- tm_map(corpus, remove.slang, "'n ", "ing ")
  corpus <- tm_map(corpus, remove.slang, "&", " and")
  corpus <- tm_map(corpus, remove.slang, "[^[:alnum:][:blank:]]", "")
  corpus <- tm_map(corpus, removePunctuation) # removing special characters
  
  # badwords.txt from https://raw.githubusercontent.com/shutterstock/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en
  profanity <- readLines("badwords.txt", warn=FALSE)
  corpus <- tm_map(corpus, removeWords, profanity) 

  # convert from corpus to dataframe & save to RDS
  corpusDF <-data.frame(text=unlist(sapply(corpus, `[`, "content")), stringsAsFactors=F)
  corpusDF
}

# data frame from NGram Tokenizer; 2,3,4-Grams
dFNGramTokenizer <- function(data, dfName, start){
  options(mc.cores=6)
  
  nGram_Delimiter <-  " \\r\\n\\t.,;:\"()?!"
  
  # 2 Gram Tokenizer
  bitGramToken <- NGramTokenizer(data, Weka_control(min=2,max=2, delimiters = nGram_Delimiter))
  saveRDS(bitGramToken, file=paste(dfName, "/", "bitGramToken", "." , start, ".RDS", sep=""))
  
  # 3 Gram Tokenizer
  triGramToken <- NGramTokenizer(data, Weka_control(min=3,max=3, delimiters = nGram_Delimiter))
  saveRDS(triGramToken, file=paste(dfName, "/", "triGramToken", "." , start, ".RDS", sep=""))
  
  # 4 Gram Tokenizer
  quadGramToken <- NGramTokenizer(data, Weka_control(min=4,max=4, delimiters = nGram_Delimiter))
  saveRDS(quadGramToken, file=paste(dfName, "/", "quadGramToken", "." , start, ".RDS", sep=""))
}


#####################################
# STEP 4: Data Processing Execution
#####################################
# Execute the tiny data generator and output corpus to dataframe for NGram Tokening
# for(counter in x:102){
#   exploreSize <- 10000
#   tinyData <- tinyDataGenerator(news, "news", counter, exploreSize)
#   corpusDF <- corpus2DataFrame(tinyData)
#   dFNGramTokenizer(corpusDF, "news", counter)
# }

# Merge files and save as RDS.
mergeSaveRDS <- function(directory, type){
  listDir <- list.files(path = directory, pattern = type)
  finalVec <- ""
  for(eachFile in listDir){
    print(eachFile)
    file <- paste(directory, "/", eachFile, sep="")
    vec <- readRDS(file)
    if(length(finalVec)==1){
      finalVec <- vec
    }else{
      finalVec <- c(finalVec, vec)
    }
  }
  fileName <- paste(directory, "_", type, ".RDS", sep="")
  saveRDS(finalVec, file=fileName)
  print(paste("Merge RDS files completed:", fileName))
}

# Execute mergeSaveRDS on news, blogs, twitter.
mergeSaveRDS("news", "bitGramToken")
mergeSaveRDS("news", "triGramToken")
mergeSaveRDS("news", "quadGramToken")

mergeSaveRDS("blogs", "bitGramToken")
mergeSaveRDS("blogs", "triGramToken")
mergeSaveRDS("blogs", "quadGramToken")

mergeSaveRDS("twitter", "bitGramToken")
mergeSaveRDS("twitter", "triGramToken")
mergeSaveRDS("twitter", "quadGramToken")

# Process the 2,3,4 gramToken
mergeSaveRDSByType <- function(type){
  listDir <- list.files(pattern = type)
  finalVec <- ""
  for(eachFile in listDir){
    vec <- readRDS(eachFile)
    if(length(finalVec)==1){
      finalVec <- vec
    }else{
      finalVec <- c(finalVec, vec)
    }
  }
  fileName <- paste(type, ".RDS", sep="")
  saveRDS(finalVec, file=fileName)
  print(paste("Merge RDS files completed:", fileName))
}

mergeSaveRDSByType("bitGramToken")
mergeSaveRDSByType("triGramToken")
mergeSaveRDSByType("quadGramToken")

# save the final gramToken data frame out
saveDF <- function(directory, fileName){
  filePath <- paste(fileName, ".RDS", sep="")
  eachFile <- readRDS(file=filePath)
  dataFr <- data.frame(table(eachFile))
  dataFr <- dataFr[order(dataFr$Freq,decreasing = TRUE),]
  names(dataFr) <- c("words", "frequency")
  fileName <- paste(directory, "/", fileName, "DF", ".RDS", sep="")
  saveRDS(dataFr, file=fileName)
  print(paste("Merge RDS files completed:", fileName))
}

saveDF("dataFrame", "bitGramToken")
saveDF("dataFrame", "triGramToken")
saveDF("dataFrame", "quadGramToken")

##################################################################################
# STEP 5: Clean and generate final data frame to lookup for next word prediction
##################################################################################
# Remove the low frequencies records.
cleanDF <- function(directory, newDirectory, fileName){
  filePath <- paste(directory, "/", fileName, ".RDS", sep="")
  eachFile <- readRDS(file=filePath)
  eachFile <- eachFile[eachFile$frequency>=10,]
  rownames(eachFile) <- 1:nrow(eachFile)
  filePath <- paste(newDirectory, "/", fileName, "_clean.RDS", sep="")
  saveRDS(eachFile, file=filePath)
}

cleanDF("dataFrame", "dataFrameClean", "bitGramTokenDF")
cleanDF("dataFrame", "dataFrameClean", "triGramTokenDF")
cleanDF("dataFrame", "dataFrameClean", "quadGramTokenDF")

# Generate Key for the Data Frame.
generateKeyDF <- function(directory, fileName, nGram){
  filePath <- paste(directory, "/", fileName, ".RDS", sep="")
  eachFile <- readRDS(file=filePath)
  # last min fix
  eachFile$words <- gsub("ca not", "can not", eachFile$words)
  keys <- str_split_fixed(eachFile$words, " ", nGram)
  key <- ""
  for(i in 1:nGram-1){
    if(i==1){
      key <- keys[,i]
    }else{
      key <- paste(key, keys[,i], sep=" ")  
    }
  }
  eachFile <- cbind(eachFile, key)
  filePath <- paste(directory, "/", fileName, "_final.RDS", sep="")
  saveRDS(eachFile, file=filePath)
}

generateKeyDF("dataFrameClean", "bitGramTokenDF_clean", 2)
generateKeyDF("dataFrameClean", "triGramTokenDF_clean", 3)
generateKeyDF("dataFrameClean", "quadGramTokenDF_clean", 4)

######################################
# STEP 6: Read and Predict next word
######################################
bitData <- readRDS(file="dataFrameClean/bitGramTokenDF_clean_final.RDS")
triData <- readRDS(file="dataFrameClean/triGramTokenDF_clean_final.RDS")
quadData1 <- readRDS(file="dataFrameClean/quadGramTokenDF_clean_final1.RDS")

# Retrieve next top word list
topNextWord <- function(data, words, size){
  recommendList <- head(data[data$key==words,]$words, size)
  gsub(paste(words," ", sep=""), "", recommendList)
}

# suggest the next word (length defined by the `size` param)
suggestNextWord <- function(words, data, size)
{
  splitWords <- unlist(strsplit(words, " "))
  splitWords <- splitWords[splitWords!=""]
  # find out what type of data
  dataSize <- length(unlist(strsplit(data$words[1], " ")))
  if(dataSize <= length(splitWords)){
    searchWords <- tail(splitWords, dataSize-1)
    searchText <- ""
    for(i in 1: length(searchWords)){
      if(i > 1){
        searchText <- paste(searchText, searchWords[i], sep=" ")
      }else{
        searchText <- searchWords[i]
      }
    }
    lowerCaseWords <- tolower(searchText)
    topNextWord(data, lowerCaseWords, size)
  }
}

##########################################
# STEP 7: Example of predicting next word
##########################################
searchText <- "bruises from playing outside"
suggestNextWord(corpus2DataFrame(searchText)$text, quadData, 20)
suggestNextWord(corpus2DataFrame(searchText)$text, triData, 20)
suggestNextWord(corpus2DataFrame(searchText)$text, bitData, 20)

