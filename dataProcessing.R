install.packages("keras")
install.packages("tfdatasets")
install.packages("magick")
library(keras)
library(tfdatasets)
library(magick)
library(dplyr)

raw_data <- read.csv(file='D:\\Project_Data\\train_csv\\train.csv')
train_directory <- "../input/trainimages/trainResized/"
raw_data$landmark_id <- as.character(raw_data$landmark_id)
for(i in 289827:length(raw_data$landmark_id)){
  raw_data[i,1] <- paste0(train_directory,raw_data[i,2],"/",raw_data[i,1],".jpg")
  print(i)
}
write.csv(raw_data,"D:\\Project_Data\\train_csv\\trainProcessed.csv", row.names = FALSE)


length(raw_data$landmark_id)
raw_data[289827,1]
tail(raw_data)
train_df <- train_df %>% group_by(landmark_id) %>% summarise(n = n()) #n ranges from 2-2231 w/ outlier of 6271
train_df <- train_df %>% summarise(mean = mean(n)) #mean is 19.4
train_df <- arrange(train_df,n)
train_df <- train_df %>% filter(n>140)
head(train_df)
tail(train_df)
length(train_df$n)

barplot(train_df$n, ylab="images", xlab="imageIds")

#restructure data storage
d <- "D:\\Project_Data\\test_img"
d2 <- "D:\\Project_Data\\data\\test"
for (i in 1:length(train_df$id)){
  im_id <- train_df[i,1]
  
  data_dir_old <- Get_Dir(im_id,d)#get path to old directory
  data_dir_new <- Get_New_Dir(im_id,d2,train_df[i,2])#get path to new directory
  
  file.rename(from=data_dir_old, to=data_dir_new)
}

Get_New_Dir <- function(image_id,data_dir, landmark){
  imgfname <- paste(c(data_dir,landmark), collapse = '\\')
  if (!isTRUE(file.info(imgfname)$isdir)) dir.create(imgfname, recursive=TRUE)
    imgfname <- paste(imgfname,"\\",image_id,".jpg", sep='')
  return(imgfname)
}

Get_Dir <- function(image_id, data_dir){
  imgfname <- paste(c(data_dir,substr(image_id,1,1),substr(image_id,2,2),substr(image_id,3,3),image_id), collapse = '\\')
  imgfname <- paste(imgfname,".jpg", sep='')
  return(imgfname)
}



#create a sample data set
sample_n <- sample(1:nrow(train_df), 200, replace = FALSE, prob = NULL)
head(train_df[sample_n[1],1])

Get_Dir(train_df[sample_n[1],1],d)
