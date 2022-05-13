{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6fef3ecc",
   "metadata": {
    "_execution_state": "idle",
    "_uuid": "051d70d956493feee0c6d64651c6a088724dca2a",
    "execution": {
     "iopub.execute_input": "2022-05-13T02:03:20.920011Z",
     "iopub.status.busy": "2022-05-13T02:03:20.917429Z",
     "iopub.status.idle": "2022-05-13T02:03:38.549089Z",
     "shell.execute_reply": "2022-05-13T02:03:38.547508Z"
    },
    "papermill": {
     "duration": 17.64007,
     "end_time": "2022-05-13T02:03:38.551965",
     "exception": false,
     "start_time": "2022-05-13T02:03:20.911895",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "── \u001b[1mAttaching packages\u001b[22m ─────────────────────────────────────── tidyverse 1.3.1 ──\n",
      "\n",
      "\u001b[32m✔\u001b[39m \u001b[34mggplot2\u001b[39m 3.3.5     \u001b[32m✔\u001b[39m \u001b[34mpurrr  \u001b[39m 0.3.4\n",
      "\u001b[32m✔\u001b[39m \u001b[34mtibble \u001b[39m 3.1.6     \u001b[32m✔\u001b[39m \u001b[34mdplyr  \u001b[39m 1.0.8\n",
      "\u001b[32m✔\u001b[39m \u001b[34mtidyr  \u001b[39m 1.2.0     \u001b[32m✔\u001b[39m \u001b[34mstringr\u001b[39m 1.4.0\n",
      "\u001b[32m✔\u001b[39m \u001b[34mreadr  \u001b[39m 2.1.2     \u001b[32m✔\u001b[39m \u001b[34mforcats\u001b[39m 0.5.1\n",
      "\n",
      "── \u001b[1mConflicts\u001b[22m ────────────────────────────────────────── tidyverse_conflicts() ──\n",
      "\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mfilter()\u001b[39m masks \u001b[34mstats\u001b[39m::filter()\n",
      "\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mlag()\u001b[39m    masks \u001b[34mstats\u001b[39m::lag()\n",
      "\n",
      "Loading required package: lattice\n",
      "\n",
      "\n",
      "Attaching package: ‘caret’\n",
      "\n",
      "\n",
      "The following object is masked from ‘package:purrr’:\n",
      "\n",
      "    lift\n",
      "\n",
      "\n",
      "The following object is masked from ‘package:httr’:\n",
      "\n",
      "    progress\n",
      "\n",
      "\n",
      "\n",
      "Attaching package: ‘tensorflow’\n",
      "\n",
      "\n",
      "The following object is masked from ‘package:caret’:\n",
      "\n",
      "    train\n",
      "\n",
      "\n",
      "Installing package into ‘/usr/local/lib/R/site-library’\n",
      "(as ‘lib’ is unspecified)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#imports\n",
    "library(tidyverse)\n",
    "library(keras)\n",
    "library(caret)\n",
    "library(reticulate)\n",
    "library(tensorflow)\n",
    "install.packages(\"tfhub\")\n",
    "library(tfhub)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd298d9a",
   "metadata": {
    "papermill": {
     "duration": 0.003136,
     "end_time": "2022-05-13T02:03:38.558943",
     "exception": false,
     "start_time": "2022-05-13T02:03:38.555807",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Pre-processing**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b7bb74e4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-05-13T02:03:38.593517Z",
     "iopub.status.busy": "2022-05-13T02:03:38.566506Z",
     "iopub.status.idle": "2022-05-13T02:03:51.154113Z",
     "shell.execute_reply": "2022-05-13T02:03:51.152616Z"
    },
    "papermill": {
     "duration": 12.594686,
     "end_time": "2022-05-13T02:03:51.156797",
     "exception": false,
     "start_time": "2022-05-13T02:03:38.562111",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 3</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>X</th><th scope=col>id</th><th scope=col>landmark_id</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>120</td><td>../input/trainimages224/trainResized2/27/00cba0067c078490.jpg</td><td>27</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>122</td><td>../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg</td><td>27</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>123</td><td>../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg</td><td>27</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>124</td><td>../input/trainimages224/trainResized2/27/0319627771784e54.jpg</td><td>27</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>125</td><td>../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg</td><td>27</td></tr>\n",
       "\t<tr><th scope=row>7</th><td>126</td><td>../input/trainimages224/trainResized2/27/038450020a8c3338.jpg</td><td>27</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 3\n",
       "\\begin{tabular}{r|lll}\n",
       "  & X & id & landmark\\_id\\\\\n",
       "  & <int> & <chr> & <chr>\\\\\n",
       "\\hline\n",
       "\t1 & 120 & ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg & 27\\\\\n",
       "\t3 & 122 & ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg & 27\\\\\n",
       "\t4 & 123 & ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg & 27\\\\\n",
       "\t5 & 124 & ../input/trainimages224/trainResized2/27/0319627771784e54.jpg & 27\\\\\n",
       "\t6 & 125 & ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg & 27\\\\\n",
       "\t7 & 126 & ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg & 27\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 3\n",
       "\n",
       "| <!--/--> | X &lt;int&gt; | id &lt;chr&gt; | landmark_id &lt;chr&gt; |\n",
       "|---|---|---|---|\n",
       "| 1 | 120 | ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg | 27 |\n",
       "| 3 | 122 | ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg | 27 |\n",
       "| 4 | 123 | ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg | 27 |\n",
       "| 5 | 124 | ../input/trainimages224/trainResized2/27/0319627771784e54.jpg | 27 |\n",
       "| 6 | 125 | ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg | 27 |\n",
       "| 7 | 126 | ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg | 27 |\n",
       "\n"
      ],
      "text/plain": [
       "  X   id                                                            landmark_id\n",
       "1 120 ../input/trainimages224/trainResized2/27/00cba0067c078490.jpg 27         \n",
       "3 122 ../input/trainimages224/trainResized2/27/0110a04e618bc368.jpg 27         \n",
       "4 123 ../input/trainimages224/trainResized2/27/026afdc670937e3b.jpg 27         \n",
       "5 124 ../input/trainimages224/trainResized2/27/0319627771784e54.jpg 27         \n",
       "6 125 ../input/trainimages224/trainResized2/27/0362bd7cb1d405e6.jpg 27         \n",
       "7 126 ../input/trainimages224/trainResized2/27/038450020a8c3338.jpg 27         "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "204"
      ],
      "text/latex": [
       "204"
      ],
      "text/markdown": [
       "204"
      ],
      "text/plain": [
       "[1] 204"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "204"
      ],
      "text/latex": [
       "204"
      ],
      "text/markdown": [
       "204"
      ],
      "text/plain": [
       "[1] 204"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "204"
      ],
      "text/latex": [
       "204"
      ],
      "text/markdown": [
       "204"
      ],
      "text/plain": [
       "[1] 204"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "204"
      ],
      "text/latex": [
       "204"
      ],
      "text/markdown": [
       "204"
      ],
      "text/plain": [
       "[1] 204"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#define variables\n",
    "results <- read.csv(\"../input/csv224/train_csv224.csv\")\n",
    "train_directory <- \"../input/trainimages224/trainResized2/\" #only has images from n>100 ... n<4000\n",
    "size = c(224L,224L)\n",
    "epochs = 30\n",
    "batch_size = 200\n",
    "input_shapeT = c(size,3L)\n",
    "\n",
    "#get df\n",
    "set.seed(123)\n",
    "ids <- results %>% group_by(landmark_id) %>% summarise(n = n())\n",
    "ids <- ids %>% filter(n>295, n<4000) #204 classes\n",
    "raw_data <- results %>% filter(landmark_id %in% ids$landmark_id)\n",
    "raw_data$landmark_id <- as.character(raw_data$landmark_id)\n",
    "classes = length(unique(raw_data$landmark_id))\n",
    "\n",
    "#get train/test split\n",
    "ind <- createDataPartition(\n",
    "    y = raw_data$landmark_id,\n",
    "    times = 1,\n",
    "    p = .70,\n",
    "    list = FALSE\n",
    ")\n",
    "train_data <- raw_data[ind,]\n",
    "test_data <- raw_data[-ind,]\n",
    "\n",
    "#get validation/test split\n",
    "ind <- createDataPartition(\n",
    "    y = test_data$landmark_id,\n",
    "    times = 1,\n",
    "    p = .50,\n",
    "    list = FALSE\n",
    ")\n",
    "valid_data <- test_data[ind,]\n",
    "test_data <- test_data[-ind,]\n",
    "\n",
    "#sanity check\n",
    "head(train_data)\n",
    "classes\n",
    "length(unique(train_data$landmark_id))\n",
    "length(unique(valid_data$landmark_id))\n",
    "length(unique(test_data$landmark_id))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85585813",
   "metadata": {
    "papermill": {
     "duration": 0.004125,
     "end_time": "2022-05-13T02:03:51.165318",
     "exception": false,
     "start_time": "2022-05-13T02:03:51.161193",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Data Augmentation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7e59dfc2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-05-13T02:03:51.175337Z",
     "iopub.status.busy": "2022-05-13T02:03:51.174092Z",
     "iopub.status.idle": "2022-05-13T02:04:01.601332Z",
     "shell.execute_reply": "2022-05-13T02:04:01.599262Z"
    },
    "papermill": {
     "duration": 10.435569,
     "end_time": "2022-05-13T02:04:01.604700",
     "exception": false,
     "start_time": "2022-05-13T02:03:51.169131",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Loaded Tensorflow version 2.6.3\n",
      "\n"
     ]
    }
   ],
   "source": [
    "train_aug <- image_data_generator(rotation_range = 10,\n",
    "                                  width_shift_range=0.2,\n",
    "                                  height_shift_range=0.2,\n",
    "                                  horizontal_flip = TRUE,\n",
    "                                  rescale=1./255,\n",
    "                                  zoom_range=0.2,\n",
    "                                  fill_mode='nearest')\n",
    "\n",
    "valid_aug <- image_data_generator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aeaf655b",
   "metadata": {
    "papermill": {
     "duration": 0.004176,
     "end_time": "2022-05-13T02:04:01.613702",
     "exception": false,
     "start_time": "2022-05-13T02:04:01.609526",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Build Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "07bfc8d7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-05-13T02:04:01.625173Z",
     "iopub.status.busy": "2022-05-13T02:04:01.623607Z",
     "iopub.status.idle": "2022-05-13T02:04:06.055690Z",
     "shell.execute_reply": "2022-05-13T02:04:06.053738Z"
    },
    "papermill": {
     "duration": 4.440581,
     "end_time": "2022-05-13T02:04:06.058592",
     "exception": false,
     "start_time": "2022-05-13T02:04:01.618011",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Model\n",
       "Model: \"model\"\n",
       "________________________________________________________________________________\n",
       "Layer (type)                        Output Shape                    Param #     \n",
       "================================================================================\n",
       "input_1 (InputLayer)                [(None, 224, 224, 3)]           0           \n",
       "________________________________________________________________________________\n",
       "block1_conv1 (Conv2D)               (None, 224, 224, 64)            1792        \n",
       "________________________________________________________________________________\n",
       "block1_conv2 (Conv2D)               (None, 224, 224, 64)            36928       \n",
       "________________________________________________________________________________\n",
       "block1_pool (MaxPooling2D)          (None, 112, 112, 64)            0           \n",
       "________________________________________________________________________________\n",
       "block2_conv1 (Conv2D)               (None, 112, 112, 128)           73856       \n",
       "________________________________________________________________________________\n",
       "block2_conv2 (Conv2D)               (None, 112, 112, 128)           147584      \n",
       "________________________________________________________________________________\n",
       "block2_pool (MaxPooling2D)          (None, 56, 56, 128)             0           \n",
       "________________________________________________________________________________\n",
       "block3_conv1 (Conv2D)               (None, 56, 56, 256)             295168      \n",
       "________________________________________________________________________________\n",
       "block3_conv2 (Conv2D)               (None, 56, 56, 256)             590080      \n",
       "________________________________________________________________________________\n",
       "block3_conv3 (Conv2D)               (None, 56, 56, 256)             590080      \n",
       "________________________________________________________________________________\n",
       "block3_pool (MaxPooling2D)          (None, 28, 28, 256)             0           \n",
       "________________________________________________________________________________\n",
       "block4_conv1 (Conv2D)               (None, 28, 28, 512)             1180160     \n",
       "________________________________________________________________________________\n",
       "block4_conv2 (Conv2D)               (None, 28, 28, 512)             2359808     \n",
       "________________________________________________________________________________\n",
       "block4_conv3 (Conv2D)               (None, 28, 28, 512)             2359808     \n",
       "________________________________________________________________________________\n",
       "block4_pool (MaxPooling2D)          (None, 14, 14, 512)             0           \n",
       "________________________________________________________________________________\n",
       "block5_conv1 (Conv2D)               (None, 14, 14, 512)             2359808     \n",
       "________________________________________________________________________________\n",
       "block5_conv2 (Conv2D)               (None, 14, 14, 512)             2359808     \n",
       "________________________________________________________________________________\n",
       "block5_conv3 (Conv2D)               (None, 14, 14, 512)             2359808     \n",
       "________________________________________________________________________________\n",
       "block5_pool (MaxPooling2D)          (None, 7, 7, 512)               0           \n",
       "________________________________________________________________________________\n",
       "flatten (Flatten)                   (None, 25088)                   0           \n",
       "________________________________________________________________________________\n",
       "dense_2 (Dense)                     (None, 250)                     6272250     \n",
       "________________________________________________________________________________\n",
       "dense_1 (Dense)                     (None, 250)                     62750       \n",
       "________________________________________________________________________________\n",
       "batch_normalization (BatchNormaliza (None, 250)                     1000        \n",
       "________________________________________________________________________________\n",
       "dense (Dense)                       (None, 204)                     51204       \n",
       "================================================================================\n",
       "Total params: 21,101,892\n",
       "Trainable params: 15,825,936\n",
       "Non-trainable params: 5,275,956\n",
       "________________________________________________________________________________\n",
       "\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Base model (VGG)\n",
    "base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE, input_shape = input_shapeT)\n",
    "\n",
    "layers <- base_model$layers\n",
    "for (layer in base_model$layers)\n",
    "  layer$trainable <- FALSE\n",
    "\n",
    "predictions <- base_model$output %>% \n",
    "    layer_flatten() %>%\n",
    "    layer_dense(units = 250, activation = 'relu', regularizer_l2(l = 0.01)) %>%\n",
    "    layer_dense(units = 250, activation = 'relu', regularizer_l2(l = 0.01)) %>%\n",
    "    layer_batch_normalization() %>%\n",
    "    layer_dense(units = classes, activation = 'softmax')\n",
    "\n",
    "#make last 2 trainable\n",
    "for (i in 14:length(layers))\n",
    "  layers[[i]]$trainable <- TRUE\n",
    "\n",
    "model <- keras_model(inputs = base_model$input, outputs = predictions)\n",
    "model %>% compile(optimizer = optimizer_adam(learning_rate=1e-4, amsgrad = TRUE), loss = 'categorical_crossentropy', metrics = 'accuracy')\n",
    "model\n",
    "#visualize layers\n",
    "# for (i in 1:length(layers)){\n",
    "#     cat(i, layers[[i]]$name,model$layers[[i]]$trainable, \"\\n\")\n",
    "    \n",
    "# }"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "863bac79",
   "metadata": {
    "papermill": {
     "duration": 0.004138,
     "end_time": "2022-05-13T02:04:06.067107",
     "exception": false,
     "start_time": "2022-05-13T02:04:06.062969",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Fit Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5c0cec13",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-05-13T02:04:06.079475Z",
     "iopub.status.busy": "2022-05-13T02:04:06.077607Z",
     "iopub.status.idle": "2022-05-13T09:28:55.978669Z",
     "shell.execute_reply": "2022-05-13T09:28:55.976943Z"
    },
    "papermill": {
     "duration": 26689.909262,
     "end_time": "2022-05-13T09:28:55.981539",
     "exception": false,
     "start_time": "2022-05-13T02:04:06.072277",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#callback\n",
    "csv_logger = callback_csv_logger('training.log')\n",
    "\n",
    "train_gen = flow_images_from_dataframe(\n",
    "dataframe = train_data,\n",
    "y_col = \"landmark_id\",\n",
    "x_col = \"id\",\n",
    "subset=\"training\",\n",
    "generator = train_aug,\n",
    "target_size = size,\n",
    "color_mode = \"rgb\",\n",
    "class_mode = \"categorical\",\n",
    "batch_size = batch_size,\n",
    "shuffle = TRUE)\n",
    "\n",
    "valid_gen = flow_images_from_dataframe(\n",
    "    dataframe = valid_data,\n",
    "    y_col = \"landmark_id\",\n",
    "    x_col = \"id\",\n",
    "    subset=\"training\",\n",
    "    generator = valid_aug, #add validation augmentation to preprocess images the same as train\n",
    "    target_size = size,\n",
    "    color_mode = \"rgb\",\n",
    "    class_mode = \"categorical\",\n",
    "    batch_size = batch_size,\n",
    "    shuffle = TRUE)\n",
    "\n",
    "history_cnn <- model %>%\n",
    "  fit(train_gen,\n",
    "        validation_data = valid_gen,\n",
    "        steps_per_epoch = train_gen$n/batch_size,\n",
    "        validation_steps = valid_gen$n/batch_size,\n",
    "        epochs = epochs,\n",
    "        callbacks=csv_logger,\n",
    "        verbose = 1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9eab4452",
   "metadata": {
    "papermill": {
     "duration": 0.019426,
     "end_time": "2022-05-13T09:28:56.014177",
     "exception": false,
     "start_time": "2022-05-13T09:28:55.994751",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Results**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3acf26b0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-05-13T09:28:58.694801Z",
     "iopub.status.busy": "2022-05-13T09:28:58.692759Z",
     "iopub.status.idle": "2022-05-13T09:31:18.959940Z",
     "shell.execute_reply": "2022-05-13T09:31:18.957640Z"
    },
    "papermill": {
     "duration": 142.933756,
     "end_time": "2022-05-13T09:31:18.963091",
     "exception": false,
     "start_time": "2022-05-13T09:28:56.029335",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "Final epoch (plot to see history):\n",
       "        loss: 0.05023\n",
       "    accuracy: 0.9909\n",
       "    val_loss: 0.609\n",
       "val_accuracy: 0.8569 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "`geom_smooth()` using formula 'y ~ x'\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [],
      "text/latex": [],
      "text/markdown": [],
      "text/plain": [
       "character(0)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 30 × 5</caption>\n",
       "<thead>\n",
       "\t<tr><th scope=col>epoch</th><th scope=col>accuracy</th><th scope=col>loss</th><th scope=col>val_accuracy</th><th scope=col>val_loss</th></tr>\n",
       "\t<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><td> 0</td><td>0.2333797</td><td>3.99047232</td><td>0.3910714</td><td>2.9906664</td></tr>\n",
       "\t<tr><td> 1</td><td>0.4771953</td><td>2.59334040</td><td>0.5421429</td><td>2.0925627</td></tr>\n",
       "\t<tr><td> 2</td><td>0.6009443</td><td>1.91097498</td><td>0.6332143</td><td>1.6535573</td></tr>\n",
       "\t<tr><td> 3</td><td>0.6823995</td><td>1.47734320</td><td>0.6853572</td><td>1.3482165</td></tr>\n",
       "\t<tr><td> 4</td><td>0.7382768</td><td>1.18916893</td><td>0.7276428</td><td>1.1667498</td></tr>\n",
       "\t<tr><td> 5</td><td>0.7829847</td><td>0.97390169</td><td>0.7605714</td><td>1.0119931</td></tr>\n",
       "\t<tr><td> 6</td><td>0.8168136</td><td>0.80787081</td><td>0.7678571</td><td>0.9499411</td></tr>\n",
       "\t<tr><td> 7</td><td>0.8462878</td><td>0.67830932</td><td>0.7820000</td><td>0.8789225</td></tr>\n",
       "\t<tr><td> 8</td><td>0.8712699</td><td>0.56868708</td><td>0.7960000</td><td>0.8056636</td></tr>\n",
       "\t<tr><td> 9</td><td>0.8891622</td><td>0.48593161</td><td>0.8045000</td><td>0.7756767</td></tr>\n",
       "\t<tr><td>10</td><td>0.9064281</td><td>0.41537219</td><td>0.8031428</td><td>0.7723230</td></tr>\n",
       "\t<tr><td>11</td><td>0.9223952</td><td>0.34840104</td><td>0.8210000</td><td>0.7039852</td></tr>\n",
       "\t<tr><td>12</td><td>0.9319755</td><td>0.30177882</td><td>0.8173571</td><td>0.7247105</td></tr>\n",
       "\t<tr><td>13</td><td>0.9408070</td><td>0.26057839</td><td>0.8299286</td><td>0.6812239</td></tr>\n",
       "\t<tr><td>14</td><td>0.9500359</td><td>0.22668238</td><td>0.8290000</td><td>0.6895351</td></tr>\n",
       "\t<tr><td>15</td><td>0.9581952</td><td>0.19683102</td><td>0.8385714</td><td>0.6587067</td></tr>\n",
       "\t<tr><td>16</td><td>0.9642459</td><td>0.17027883</td><td>0.8363571</td><td>0.6576850</td></tr>\n",
       "\t<tr><td>17</td><td>0.9681422</td><td>0.15089858</td><td>0.8353571</td><td>0.6702758</td></tr>\n",
       "\t<tr><td>18</td><td>0.9744374</td><td>0.12953679</td><td>0.8394285</td><td>0.6562402</td></tr>\n",
       "\t<tr><td>19</td><td>0.9746207</td><td>0.12232443</td><td>0.8399286</td><td>0.6453697</td></tr>\n",
       "\t<tr><td>20</td><td>0.9786698</td><td>0.10656504</td><td>0.8496429</td><td>0.6129417</td></tr>\n",
       "\t<tr><td>21</td><td>0.9807631</td><td>0.09689708</td><td>0.8470714</td><td>0.6176516</td></tr>\n",
       "\t<tr><td>22</td><td>0.9821535</td><td>0.08956139</td><td>0.8428571</td><td>0.6384725</td></tr>\n",
       "\t<tr><td>23</td><td>0.9842926</td><td>0.08162012</td><td>0.8525000</td><td>0.6096703</td></tr>\n",
       "\t<tr><td>24</td><td>0.9854233</td><td>0.07750850</td><td>0.8488572</td><td>0.6173916</td></tr>\n",
       "\t<tr><td>25</td><td>0.9869055</td><td>0.06968780</td><td>0.8494286</td><td>0.6221929</td></tr>\n",
       "\t<tr><td>26</td><td>0.9882806</td><td>0.06344935</td><td>0.8507143</td><td>0.6245096</td></tr>\n",
       "\t<tr><td>27</td><td>0.9893349</td><td>0.05840765</td><td>0.8577857</td><td>0.6006460</td></tr>\n",
       "\t<tr><td>28</td><td>0.9890904</td><td>0.05663479</td><td>0.8569286</td><td>0.6001117</td></tr>\n",
       "\t<tr><td>29</td><td>0.9909239</td><td>0.05023117</td><td>0.8569286</td><td>0.6089571</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 30 × 5\n",
       "\\begin{tabular}{lllll}\n",
       " epoch & accuracy & loss & val\\_accuracy & val\\_loss\\\\\n",
       " <int> & <dbl> & <dbl> & <dbl> & <dbl>\\\\\n",
       "\\hline\n",
       "\t  0 & 0.2333797 & 3.99047232 & 0.3910714 & 2.9906664\\\\\n",
       "\t  1 & 0.4771953 & 2.59334040 & 0.5421429 & 2.0925627\\\\\n",
       "\t  2 & 0.6009443 & 1.91097498 & 0.6332143 & 1.6535573\\\\\n",
       "\t  3 & 0.6823995 & 1.47734320 & 0.6853572 & 1.3482165\\\\\n",
       "\t  4 & 0.7382768 & 1.18916893 & 0.7276428 & 1.1667498\\\\\n",
       "\t  5 & 0.7829847 & 0.97390169 & 0.7605714 & 1.0119931\\\\\n",
       "\t  6 & 0.8168136 & 0.80787081 & 0.7678571 & 0.9499411\\\\\n",
       "\t  7 & 0.8462878 & 0.67830932 & 0.7820000 & 0.8789225\\\\\n",
       "\t  8 & 0.8712699 & 0.56868708 & 0.7960000 & 0.8056636\\\\\n",
       "\t  9 & 0.8891622 & 0.48593161 & 0.8045000 & 0.7756767\\\\\n",
       "\t 10 & 0.9064281 & 0.41537219 & 0.8031428 & 0.7723230\\\\\n",
       "\t 11 & 0.9223952 & 0.34840104 & 0.8210000 & 0.7039852\\\\\n",
       "\t 12 & 0.9319755 & 0.30177882 & 0.8173571 & 0.7247105\\\\\n",
       "\t 13 & 0.9408070 & 0.26057839 & 0.8299286 & 0.6812239\\\\\n",
       "\t 14 & 0.9500359 & 0.22668238 & 0.8290000 & 0.6895351\\\\\n",
       "\t 15 & 0.9581952 & 0.19683102 & 0.8385714 & 0.6587067\\\\\n",
       "\t 16 & 0.9642459 & 0.17027883 & 0.8363571 & 0.6576850\\\\\n",
       "\t 17 & 0.9681422 & 0.15089858 & 0.8353571 & 0.6702758\\\\\n",
       "\t 18 & 0.9744374 & 0.12953679 & 0.8394285 & 0.6562402\\\\\n",
       "\t 19 & 0.9746207 & 0.12232443 & 0.8399286 & 0.6453697\\\\\n",
       "\t 20 & 0.9786698 & 0.10656504 & 0.8496429 & 0.6129417\\\\\n",
       "\t 21 & 0.9807631 & 0.09689708 & 0.8470714 & 0.6176516\\\\\n",
       "\t 22 & 0.9821535 & 0.08956139 & 0.8428571 & 0.6384725\\\\\n",
       "\t 23 & 0.9842926 & 0.08162012 & 0.8525000 & 0.6096703\\\\\n",
       "\t 24 & 0.9854233 & 0.07750850 & 0.8488572 & 0.6173916\\\\\n",
       "\t 25 & 0.9869055 & 0.06968780 & 0.8494286 & 0.6221929\\\\\n",
       "\t 26 & 0.9882806 & 0.06344935 & 0.8507143 & 0.6245096\\\\\n",
       "\t 27 & 0.9893349 & 0.05840765 & 0.8577857 & 0.6006460\\\\\n",
       "\t 28 & 0.9890904 & 0.05663479 & 0.8569286 & 0.6001117\\\\\n",
       "\t 29 & 0.9909239 & 0.05023117 & 0.8569286 & 0.6089571\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 30 × 5\n",
       "\n",
       "| epoch &lt;int&gt; | accuracy &lt;dbl&gt; | loss &lt;dbl&gt; | val_accuracy &lt;dbl&gt; | val_loss &lt;dbl&gt; |\n",
       "|---|---|---|---|---|\n",
       "|  0 | 0.2333797 | 3.99047232 | 0.3910714 | 2.9906664 |\n",
       "|  1 | 0.4771953 | 2.59334040 | 0.5421429 | 2.0925627 |\n",
       "|  2 | 0.6009443 | 1.91097498 | 0.6332143 | 1.6535573 |\n",
       "|  3 | 0.6823995 | 1.47734320 | 0.6853572 | 1.3482165 |\n",
       "|  4 | 0.7382768 | 1.18916893 | 0.7276428 | 1.1667498 |\n",
       "|  5 | 0.7829847 | 0.97390169 | 0.7605714 | 1.0119931 |\n",
       "|  6 | 0.8168136 | 0.80787081 | 0.7678571 | 0.9499411 |\n",
       "|  7 | 0.8462878 | 0.67830932 | 0.7820000 | 0.8789225 |\n",
       "|  8 | 0.8712699 | 0.56868708 | 0.7960000 | 0.8056636 |\n",
       "|  9 | 0.8891622 | 0.48593161 | 0.8045000 | 0.7756767 |\n",
       "| 10 | 0.9064281 | 0.41537219 | 0.8031428 | 0.7723230 |\n",
       "| 11 | 0.9223952 | 0.34840104 | 0.8210000 | 0.7039852 |\n",
       "| 12 | 0.9319755 | 0.30177882 | 0.8173571 | 0.7247105 |\n",
       "| 13 | 0.9408070 | 0.26057839 | 0.8299286 | 0.6812239 |\n",
       "| 14 | 0.9500359 | 0.22668238 | 0.8290000 | 0.6895351 |\n",
       "| 15 | 0.9581952 | 0.19683102 | 0.8385714 | 0.6587067 |\n",
       "| 16 | 0.9642459 | 0.17027883 | 0.8363571 | 0.6576850 |\n",
       "| 17 | 0.9681422 | 0.15089858 | 0.8353571 | 0.6702758 |\n",
       "| 18 | 0.9744374 | 0.12953679 | 0.8394285 | 0.6562402 |\n",
       "| 19 | 0.9746207 | 0.12232443 | 0.8399286 | 0.6453697 |\n",
       "| 20 | 0.9786698 | 0.10656504 | 0.8496429 | 0.6129417 |\n",
       "| 21 | 0.9807631 | 0.09689708 | 0.8470714 | 0.6176516 |\n",
       "| 22 | 0.9821535 | 0.08956139 | 0.8428571 | 0.6384725 |\n",
       "| 23 | 0.9842926 | 0.08162012 | 0.8525000 | 0.6096703 |\n",
       "| 24 | 0.9854233 | 0.07750850 | 0.8488572 | 0.6173916 |\n",
       "| 25 | 0.9869055 | 0.06968780 | 0.8494286 | 0.6221929 |\n",
       "| 26 | 0.9882806 | 0.06344935 | 0.8507143 | 0.6245096 |\n",
       "| 27 | 0.9893349 | 0.05840765 | 0.8577857 | 0.6006460 |\n",
       "| 28 | 0.9890904 | 0.05663479 | 0.8569286 | 0.6001117 |\n",
       "| 29 | 0.9909239 | 0.05023117 | 0.8569286 | 0.6089571 |\n",
       "\n"
      ],
      "text/plain": [
       "   epoch accuracy  loss       val_accuracy val_loss \n",
       "1   0    0.2333797 3.99047232 0.3910714    2.9906664\n",
       "2   1    0.4771953 2.59334040 0.5421429    2.0925627\n",
       "3   2    0.6009443 1.91097498 0.6332143    1.6535573\n",
       "4   3    0.6823995 1.47734320 0.6853572    1.3482165\n",
       "5   4    0.7382768 1.18916893 0.7276428    1.1667498\n",
       "6   5    0.7829847 0.97390169 0.7605714    1.0119931\n",
       "7   6    0.8168136 0.80787081 0.7678571    0.9499411\n",
       "8   7    0.8462878 0.67830932 0.7820000    0.8789225\n",
       "9   8    0.8712699 0.56868708 0.7960000    0.8056636\n",
       "10  9    0.8891622 0.48593161 0.8045000    0.7756767\n",
       "11 10    0.9064281 0.41537219 0.8031428    0.7723230\n",
       "12 11    0.9223952 0.34840104 0.8210000    0.7039852\n",
       "13 12    0.9319755 0.30177882 0.8173571    0.7247105\n",
       "14 13    0.9408070 0.26057839 0.8299286    0.6812239\n",
       "15 14    0.9500359 0.22668238 0.8290000    0.6895351\n",
       "16 15    0.9581952 0.19683102 0.8385714    0.6587067\n",
       "17 16    0.9642459 0.17027883 0.8363571    0.6576850\n",
       "18 17    0.9681422 0.15089858 0.8353571    0.6702758\n",
       "19 18    0.9744374 0.12953679 0.8394285    0.6562402\n",
       "20 19    0.9746207 0.12232443 0.8399286    0.6453697\n",
       "21 20    0.9786698 0.10656504 0.8496429    0.6129417\n",
       "22 21    0.9807631 0.09689708 0.8470714    0.6176516\n",
       "23 22    0.9821535 0.08956139 0.8428571    0.6384725\n",
       "24 23    0.9842926 0.08162012 0.8525000    0.6096703\n",
       "25 24    0.9854233 0.07750850 0.8488572    0.6173916\n",
       "26 25    0.9869055 0.06968780 0.8494286    0.6221929\n",
       "27 26    0.9882806 0.06344935 0.8507143    0.6245096\n",
       "28 27    0.9893349 0.05840765 0.8577857    0.6006460\n",
       "29 28    0.9890904 0.05663479 0.8569286    0.6001117\n",
       "30 29    0.9909239 0.05023117 0.8569286    0.6089571"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       ".dl-inline {width: auto; margin:0; padding: 0}\n",
       ".dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}\n",
       ".dl-inline>dt::after {content: \":\\0020\"; padding-right: .5ex}\n",
       ".dl-inline>dt:not(:first-of-type) {padding-left: .5ex}\n",
       "</style><dl class=dl-inline><dt>loss</dt><dd>0.596700251102448</dd><dt>accuracy</dt><dd>0.859835207462311</dd></dl>\n"
      ],
      "text/latex": [
       "\\begin{description*}\n",
       "\\item[loss] 0.596700251102448\n",
       "\\item[accuracy] 0.859835207462311\n",
       "\\end{description*}\n"
      ],
      "text/markdown": [
       "loss\n",
       ":   0.596700251102448accuracy\n",
       ":   0.859835207462311\n",
       "\n"
      ],
      "text/plain": [
       "     loss  accuracy \n",
       "0.5967003 0.8598352 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\naP75Sz9/OfurhQtBJOYlEl4khpL6Be7cufOjjz5SS8Qdg/xllOjY/YShQ4eOHTt23rx5pVZ+\n8uTJaZ9/nnD/vu2mn4/Pgm++6dixY1nO2mQybdmy5fbt2wRB1KtXr2/fvmJxWZcNYBjm7t27\naWlpAQEBrq6ur2E4C1ahXOssdBfv/HfoxNh726orKACgtTfGfn7V9ihjjGF53rleg8Ltjenn\nlqXqQPTUQE/bsM8ybvxfg0fFCmgUKh4Vax8eFVsq4YyKtbHVA8/2saswLMsePnz41q1bIpGo\nXr16rVu3fnYbFxeX/Pz8ovUcOXLkow+Hh7q6jKpT00+tStfrozKy3BzUHw7opyRIZDIio4Ew\nm8BkRK/lLMoHQfBiCYjFPEXxIjGIJVaAf0+eRBzbPsBfKqIAgOH5329Fn05OnTp9emBoGIjF\nQJK8VKZ0crJynAmeRKioqKj3unXzkMumN20Q5uKcoNEuuhB1X6Pdtm1biS9yUVeuXBk54sPU\ntHS5SMTxvJlhggIDN2zcWHpwBzh8+PDMGTMSk5JsNxs3brx48eKaNWuW5QXQ6/W7d++OiYmR\ny+VNmzZt27ZtWfYqymw2P+8CNx4V+ypuru5b95MdMreGgwd1dIdH+zf8rm0y8MHBXwbG5GwK\nkbd3dz6ucxg7c2oDX/n9OxfW/bA32JM9n0yu2PDTiAG9NSc6+bY91GH2qqFhjQf2rWt/YwXx\nX/wagIOdgIIUDnb24WBXKhzsyuLZYAcAJ06cmDF9enxCwXoYkZGRCxcurFKlSrF9EcOAyUiY\nTaDXEybjzQvnj+3f1zk4qI57wYcfz8Pb1KLEkxSIRLxEkvLoUV6+LszFSfy4IdDCsr/cvEuq\nHYaOHMmLxCAWAUFyEilQFIhEvEgEJMWLJQazuVHz5oTZ9OO7ke2D/HmAfXH3x/97XO3mfvbc\nOfvtdocPHx40aFCAg3pig7r+DuobmVkrL1/nRaKjx44FBgbar/zEiRMfT5iQ8egRRRAMxwFA\n84iI9b/84uLiUupZx8bGfjV37rlz5wxGo7+f34cjRowaNUr09MJ6zwt2WVlZf/31V3x8vKur\na8uWLZs2bVrq0xW6d+/e1atXTSZTjRo1Gjdu/EJtk5Ur2AHAxc0LZixdf/nuQ6Ty6tBv2oZF\nkSplDdt0J4bkox+N/uLI5ds6kUd4g4gpi1c1Na1r1P7LNKvifk66u/X2++/0OHIjxanm/9Ju\n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\n2zW+JkvJFwAAAABJRU5ErkJggg=="
     },
     "metadata": {
      "image/png": {
       "height": 420,
       "width": 420
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "history_cnn\n",
    "plot(history_cnn)\n",
    "#save model\n",
    "save_model_tf(model, \"mymodel/\", include_optimizer = TRUE)\n",
    "model_ <- load_model_tf(\"mymodel/\")\n",
    "#model\n",
    "\n",
    "list.files(path = \"../input/trainimages224/trainResized2/230/\")\n",
    "accuracy <- results <- read.csv(\"../working/training.log\")\n",
    "accuracy\n",
    "#repeat compile and on...\n",
    "\n",
    "test_gen = flow_images_from_dataframe(\n",
    "    dataframe = test_data,\n",
    "    y_col = \"landmark_id\",\n",
    "    x_col = \"id\",\n",
    "    subset=\"training\",\n",
    "    generator = valid_aug, #add validation augmentation to preprocess images the same as train\n",
    "    target_size = size,\n",
    "    color_mode = \"rgb\",\n",
    "    class_mode = \"categorical\",\n",
    "    batch_size = batch_size,\n",
    "    shuffle = TRUE)\n",
    "\n",
    "score <- model %>% evaluate(test_gen, verbose = 0)\n",
    "score"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.0.5"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 26882.333467,
   "end_time": "2022-05-13T09:31:20.145973",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-05-13T02:03:17.812506",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
