{"metadata":{"kernelspec":{"name":"ir","display_name":"R","language":"R"},"language_info":{"name":"R","codemirror_mode":"r","pygments_lexer":"r","mimetype":"text/x-r-source","file_extension":".r","version":"4.0.5"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"#imports\nlibrary(tidyverse)\nlibrary(keras)\nlibrary(caret)\nlibrary(reticulate)\nlibrary(tensorflow)\ninstall.packages(\"tfhub\")\nlibrary(tfhub)","metadata":{"_uuid":"051d70d956493feee0c6d64651c6a088724dca2a","_execution_state":"idle","execution":{"iopub.status.busy":"2022-05-13T01:43:10.922832Z","iopub.execute_input":"2022-05-13T01:43:10.925029Z","iopub.status.idle":"2022-05-13T01:43:32.449090Z"},"trusted":true},"execution_count":1,"outputs":[{"name":"stderr","text":"── \u001b[1mAttaching packages\u001b[22m ─────────────────────────────────────── tidyverse 1.3.1 ──\n\n\u001b[32m✔\u001b[39m \u001b[34mggplot2\u001b[39m 3.3.5     \u001b[32m✔\u001b[39m \u001b[34mpurrr  \u001b[39m 0.3.4\n\u001b[32m✔\u001b[39m \u001b[34mtibble \u001b[39m 3.1.6     \u001b[32m✔\u001b[39m \u001b[34mdplyr  \u001b[39m 1.0.8\n\u001b[32m✔\u001b[39m \u001b[34mtidyr  \u001b[39m 1.2.0     \u001b[32m✔\u001b[39m \u001b[34mstringr\u001b[39m 1.4.0\n\u001b[32m✔\u001b[39m \u001b[34mreadr  \u001b[39m 2.1.2     \u001b[32m✔\u001b[39m \u001b[34mforcats\u001b[39m 0.5.1\n\n── \u001b[1mConflicts\u001b[22m ────────────────────────────────────────── tidyverse_conflicts() ──\n\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mfilter()\u001b[39m masks \u001b[34mstats\u001b[39m::filter()\n\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mlag()\u001b[39m    masks \u001b[34mstats\u001b[39m::lag()\n\nLoading required package: lattice\n\n\nAttaching package: ‘caret’\n\n\nThe following object is masked from ‘package:purrr’:\n\n    lift\n\n\nThe following object is masked from ‘package:httr’:\n\n    progress\n\n\n\nAttaching package: ‘tensorflow’\n\n\nThe following object is masked from ‘package:caret’:\n\n    train\n\n\nInstalling package into ‘/usr/local/lib/R/site-library’\n(as ‘lib’ is unspecified)\n\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# **Pre-processing**","metadata":{}},{"cell_type":"code","source":"#define variables\nresults <- read.csv(\"../input/csv224/train_csv224.csv\")\ntrain_directory <- \"../input/trainimages224/trainResized2/\" #only has images from n>100 ... n<4000\nsize = c(224L,224L)\nepochs = 30\nbatch_size = 200\ninput_shapeT = c(size,3L)\n\n#get df\nset.seed(123)\nids <- results %>% group_by(landmark_id) %>% summarise(n = n())\nids <- ids %>% filter(n>295, n<4000) #204 classes\nraw_data <- results %>% filter(landmark_id %in% ids$landmark_id)\nraw_data$landmark_id <- as.character(raw_data$landmark_id)\nclasses = length(unique(raw_data$landmark_id))\n\n#get train/test split\nind <- createDataPartition(\n    y = raw_data$landmark_id,\n    times = 1,\n    p = .70,\n    list = FALSE\n)\ntrain_data <- raw_data[ind,]\ntest_data <- raw_data[-ind,]\n\n#get validation/test split\nind <- createDataPartition(\n    y = test_data$landmark_id,\n    times = 1,\n    p = .50,\n    list = FALSE\n)\nvalid_data <- test_data[ind,]\ntest_data <- test_data[-ind,]\n\n#sanity check\nhead(train_data)\nclasses\nlength(unique(train_data$landmark_id))\nlength(unique(valid_data$landmark_id))\nlength(unique(test_data$landmark_id))","metadata":{"execution":{"iopub.status.busy":"2022-05-13T01:45:29.019408Z","iopub.execute_input":"2022-05-13T01:45:29.020977Z","iopub.status.idle":"2022-05-13T01:45:35.713762Z"},"trusted":true},"execution_count":6,"outputs":[{"output_type":"display_data","data":{"text/html":"<table class=\"dataframe\">\n<caption>A data.frame: 6 × 3</caption>\n<thead>\n\t<tr><th></th><th scope=col>X</th><th scope=col>id</th><th scope=col>landmark_id</th></tr>\n\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>\n</thead>\n<tbody>\n\t<tr><th scope=row>1</th><td>120</td><td>../input/trainimages224/trainResized2/27/00cba0067c078490.jpg</td><td>27</td></tr>\n\t<tr><th scope=row>3</th><td>122</td><td>../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg</td><td>27</td></tr>\n\t<tr><th scope=row>4</th><td>123</td><td>../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg</td><td>27</td></tr>\n\t<tr><th scope=row>5</th><td>124</td><td>../input/trainimages224/trainResized2/27/0319627771784e54.jpg</td><td>27</td></tr>\n\t<tr><th scope=row>6</th><td>125</td><td>../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg</td><td>27</td></tr>\n\t<tr><th scope=row>7</th><td>126</td><td>../input/trainimages224/trainResized2/27/038450020a8c3338.jpg</td><td>27</td></tr>\n</tbody>\n</table>\n","text/markdown":"\nA data.frame: 6 × 3\n\n| <!--/--> | X &lt;int&gt; | id &lt;chr&gt; | landmark_id &lt;chr&gt; |\n|---|---|---|---|\n| 1 | 120 | ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg | 27 |\n| 3 | 122 | ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg | 27 |\n| 4 | 123 | ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg | 27 |\n| 5 | 124 | ../input/trainimages224/trainResized2/27/0319627771784e54.jpg | 27 |\n| 6 | 125 | ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg | 27 |\n| 7 | 126 | ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg | 27 |\n\n","text/latex":"A data.frame: 6 × 3\n\\begin{tabular}{r|lll}\n  & X & id & landmark\\_id\\\\\n  & <int> & <chr> & <chr>\\\\\n\\hline\n\t1 & 120 & ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg & 27\\\\\n\t3 & 122 & ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg & 27\\\\\n\t4 & 123 & ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg & 27\\\\\n\t5 & 124 & ../input/trainimages224/trainResized2/27/0319627771784e54.jpg & 27\\\\\n\t6 & 125 & ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg & 27\\\\\n\t7 & 126 & ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg & 27\\\\\n\\end{tabular}\n","text/plain":"  X   id                                                            landmark_id\n1 120 ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg 27         \n3 122 ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg 27         \n4 123 ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg 27         \n5 124 ../input/trainimages224/trainResized2/27/0319627771784e54.jpg 27         \n6 125 ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg 27         \n7 126 ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg 27         "},"metadata":{}},{"output_type":"display_data","data":{"text/html":"204","text/markdown":"204","text/latex":"204","text/plain":"[1] 204"},"metadata":{}},{"output_type":"display_data","data":{"text/html":"204","text/markdown":"204","text/latex":"204","text/plain":"[1] 204"},"metadata":{}},{"output_type":"display_data","data":{"text/html":"204","text/markdown":"204","text/latex":"204","text/plain":"[1] 204"},"metadata":{}},{"output_type":"display_data","data":{"text/html":"204","text/markdown":"204","text/latex":"204","text/plain":"[1] 204"},"metadata":{}}]},{"cell_type":"markdown","source":"# **Data Augmentation**","metadata":{}},{"cell_type":"code","source":"train_aug <- image_data_generator(rotation_range = 10,\n                                  width_shift_range=0.2,\n                                  height_shift_range=0.2,\n                                  horizontal_flip = TRUE,\n                                  rescale=1./255,\n                                  zoom_range=0.2,\n                                  fill_mode='nearest')\n\nvalid_aug <- image_data_generator(rescale=1./255)","metadata":{"execution":{"iopub.status.busy":"2022-05-12T04:08:42.854363Z","iopub.execute_input":"2022-05-12T04:08:42.855595Z","iopub.status.idle":"2022-05-12T04:08:51.982551Z"},"trusted":true},"execution_count":3,"outputs":[{"name":"stderr","text":"Loaded Tensorflow version 2.6.3\n\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# **Build Model**","metadata":{}},{"cell_type":"code","source":"#Base model (VGG)\nbase_model <- application_vgg16(weights = 'imagenet', include_top = FALSE, input_shape = input_shapeT)\n\nlayers <- base_model$layers\nfor (layer in base_model$layers)\n  layer$trainable <- FALSE\n\npredictions <- base_model$output %>% \n    layer_flatten() %>%\n    layer_dense(units = 250, activation = 'relu', regularizer_l2(l = 0.01)) %>%\n    layer_dense(units = 250, activation = 'relu', regularizer_l2(l = 0.01)) %>%\n    layer_batch_normalization() %>%\n    layer_dense(units = classes, activation = 'softmax')\n\n#make last 2 trainable\nfor (i in 14:length(layers))\n  layers[[i]]$trainable <- TRUE\n\nmodel <- keras_model(inputs = base_model$input, outputs = predictions)\nmodel %>% compile(optimizer = optimizer_adam(learning_rate=1e-4, amsgrad = TRUE), loss = 'categorical_crossentropy', metrics = 'accuracy')\nmodel\n#visualize layers\n# for (i in 1:length(layers)){\n#     cat(i, layers[[i]]$name,model$layers[[i]]$trainable, \"\\n\")\n    \n# }","metadata":{"execution":{"iopub.status.busy":"2022-05-12T04:08:51.986025Z","iopub.execute_input":"2022-05-12T04:08:51.987499Z","iopub.status.idle":"2022-05-12T04:08:55.351036Z"},"trusted":true},"execution_count":4,"outputs":[{"output_type":"display_data","data":{"text/plain":"Model\nModel: \"model\"\n________________________________________________________________________________\nLayer (type)                        Output Shape                    Param #     \n================================================================================\ninput_1 (InputLayer)                [(None, 224, 224, 3)]           0           \n________________________________________________________________________________\nblock1_conv1 (Conv2D)               (None, 224, 224, 64)            1792        \n________________________________________________________________________________\nblock1_conv2 (Conv2D)               (None, 224, 224, 64)            36928       \n________________________________________________________________________________\nblock1_pool (MaxPooling2D)          (None, 112, 112, 64)            0           \n________________________________________________________________________________\nblock2_conv1 (Conv2D)               (None, 112, 112, 128)           73856       \n________________________________________________________________________________\nblock2_conv2 (Conv2D)               (None, 112, 112, 128)           147584      \n________________________________________________________________________________\nblock2_pool (MaxPooling2D)          (None, 56, 56, 128)             0           \n________________________________________________________________________________\nblock3_conv1 (Conv2D)               (None, 56, 56, 256)             295168      \n________________________________________________________________________________\nblock3_conv2 (Conv2D)               (None, 56, 56, 256)             590080      \n________________________________________________________________________________\nblock3_conv3 (Conv2D)               (None, 56, 56, 256)             590080      \n________________________________________________________________________________\nblock3_pool (MaxPooling2D)          (None, 28, 28, 256)             0           \n________________________________________________________________________________\nblock4_conv1 (Conv2D)               (None, 28, 28, 512)             1180160     \n________________________________________________________________________________\nblock4_conv2 (Conv2D)               (None, 28, 28, 512)             2359808     \n________________________________________________________________________________\nblock4_conv3 (Conv2D)               (None, 28, 28, 512)             2359808     \n________________________________________________________________________________\nblock4_pool (MaxPooling2D)          (None, 14, 14, 512)             0           \n________________________________________________________________________________\nblock5_conv1 (Conv2D)               (None, 14, 14, 512)             2359808     \n________________________________________________________________________________\nblock5_conv2 (Conv2D)               (None, 14, 14, 512)             2359808     \n________________________________________________________________________________\nblock5_conv3 (Conv2D)               (None, 14, 14, 512)             2359808     \n________________________________________________________________________________\nblock5_pool (MaxPooling2D)          (None, 7, 7, 512)               0           \n________________________________________________________________________________\nflatten (Flatten)                   (None, 25088)                   0           \n________________________________________________________________________________\ndense_2 (Dense)                     (None, 250)                     6272250     \n________________________________________________________________________________\ndense_1 (Dense)                     (None, 250)                     62750       \n________________________________________________________________________________\nbatch_normalization (BatchNormaliza (None, 250)                     1000        \n________________________________________________________________________________\ndense (Dense)                       (None, 1970)                    494470      \n================================================================================\nTotal params: 21,545,158\nTrainable params: 16,269,202\nNon-trainable params: 5,275,956\n________________________________________________________________________________\n\n"},"metadata":{}}]},{"cell_type":"markdown","source":"# **Fit Model**","metadata":{}},{"cell_type":"code","source":"#callback\ncsv_logger = callback_csv_logger('training.log')\n\ntrain_gen = flow_images_from_dataframe(\ndataframe = train_data,\ny_col = \"landmark_id\",\nx_col = \"id\",\nsubset=\"training\",\ngenerator = train_aug,\ntarget_size = size,\ncolor_mode = \"rgb\",\nclass_mode = \"categorical\",\nbatch_size = batch_size,\nshuffle = TRUE)\n\nvalid_gen = flow_images_from_dataframe(\n    dataframe = valid_data,\n    y_col = \"landmark_id\",\n    x_col = \"id\",\n    subset=\"training\",\n    generator = valid_aug, #add validation augmentation to preprocess images the same as train\n    target_size = size,\n    color_mode = \"rgb\",\n    class_mode = \"categorical\",\n    batch_size = batch_size,\n    shuffle = TRUE)\n\nhistory_cnn <- model %>%\n  fit(train_gen,\n        validation_data = valid_gen,\n        steps_per_epoch = train_gen$n/batch_size,\n        validation_steps = valid_gen$n/batch_size,\n        epochs = epochs,\n        callbacks=csv_logger,\n        verbose = 1)\n","metadata":{"execution":{"iopub.status.busy":"2022-05-12T04:08:55.354205Z","iopub.execute_input":"2022-05-12T04:08:55.355578Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"# **Results**","metadata":{}},{"cell_type":"code","source":"history_cnn\nplot(history_cnn)\n#save model\nsave_model_tf(model, \"mymodel/\", include_optimizer = TRUE)\nmodel_ <- load_model_tf(\"mymodel/\")\n#model\n\nlist.files(path = \"../input/trainimages224/trainResized2/230/\")\naccuracy <- results <- read.csv(\"../working/training.log\")\naccuracy\n#repeat compile and on...\n\ntest_gen = flow_images_from_dataframe(\n    dataframe = test_data,\n    y_col = \"landmark_id\",\n    x_col = \"id\",\n    subset=\"training\",\n    generator = valid_aug, #add validation augmentation to preprocess images the same as train\n    target_size = size,\n    color_mode = \"rgb\",\n    class_mode = \"categorical\",\n    batch_size = batch_size,\n    shuffle = TRUE)\n\nscore <- model %>% evaluate(test_gen, verbose = 0)\nscore","metadata":{"execution":{"iopub.status.busy":"2022-05-12T16:55:26.271086Z","iopub.execute_input":"2022-05-12T16:55:26.275157Z","iopub.status.idle":"2022-05-12T16:55:26.428679Z"},"trusted":true},"execution_count":1,"outputs":[{"ename":"ERROR","evalue":"Error in eval(expr, envir, enclos): object 'history_cnn' not found\n","traceback":["Error in eval(expr, envir, enclos): object 'history_cnn' not found\nTraceback:\n"],"output_type":"error"}]}]}