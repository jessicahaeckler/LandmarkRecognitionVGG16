library(magick)
library(purrr)
library(dplyr)

#write("TMP = D:/temp", file=file.path('C:/Users/badwo/Documents/.Renviron'))

train_df <- read.csv(file='D:\\Project_Data\\train_csv\\train.csv')
dp <- "D:/Project_Data/data/trainResized/"
ndp <- "D:/Project_Data/data/trainResized2/"
length(train_df$landmark_id)
ids <- train_df %>% group_by(landmark_id) %>% summarise(n = n())
ids <- ids %>% filter(n>100, n<4000)#1970 image classes
train_df2 <- train_df %>% filter(landmark_id %in% ids$landmark_id)
train_df2$landmark_id <- as.character(train_df2$landmark_id)
head(train_df2)
#train_df2 <- train_df %>% filter(landmark_id>198970)

for (i in unique(train_df2$landmark_id)){
  capturas <- list.files(paste0(dp,i), pattern = "*.JPG", recursive=T, ignore.case=T)
  
  setwd(paste0(dp,i))
  
  images <- map(capturas, image_read)
  images <- image_join(images)
  
  
  #img <- image_read(paste0(dp, i, "*.jpg"))
  resize = function(x=images){image_scale(image_scale(x,"224"),"224")}
  
  img_resized=lapply(images, resize)
  
  #create new folders
  if (!isTRUE(file.info(paste0(ndp,i))$isdir)){
    dir.create(paste0(ndp,i))
  }
  
  #resize_write()
  for(j in 1:length(img_resized)){
    
    image_write(img_resized[[j]], path=paste0(ndp, i,"/",capturas[[j]]), format="jpg" )
  }
  
  rm(images,img_resized, resize,capturas,j)
  invisible(gc())
  
}
setwd(paste0(dp))
head(train_df2)
