Plastic Waste Detection with Deep Learning
==========================================

![](https://secure.gravatar.com/avatar/0549e3c28f5182ade3fa87adca74a433?s=32&d=mm&r=g) [ Floris Alexandrou](https://learnopencv.com/author/floris/)

January 25, 2022 [Leave a Comment](https://learnopencv.com/plastic-waste-detection-with-deep-learning/#disqus_thread)

[Deep Learning](https://learnopencv.com/category/deep-learning/) [Object Detection](https://learnopencv.com/category/object-detection/)

January 25, 2022 By [Leave a Comment](https://learnopencv.com/plastic-waste-detection-with-deep-learning/#disqus_thread)

[

![Blue Globe Photo by Louis Reed on Unsplash](https://learnopencv.com/wp-content/uploads/2022/01/Blue-Globe-Photo-by-Louis-Reed-on-Unsplash-1024x1024.jpg "Blue-Globe-Photo-by-Louis-Reed-on-Unsplash – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Blue-Globe-Photo-by-Louis-Reed-on-Unsplash.jpg)

Photo by [Louis Reed](https://unsplash.com/@_louisreed?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

With each passing day, the effect of climate change is becoming all too real. From hurricanes and wildfires to melting ice and rising sea levels, its impact keeps getting worse. We need to act urgently to prevent and reverse the damage. 

Plastic pollution contributes enormously to the problem. We use **481.6 billion** plastic bottles every year and only about 9% of them are recycled.

In this article, we’ll look at how computer vision can be used to identify plastic bottles. As a beginner myself, I won’t get into the details of each technology, but instead, I will share a variety of tools that helped me make the process much faster. The purpose of this article is to serve as a practical introduction to the high-level fundamental concepts of machine learning and computer vision. You only need a Google account and an appetite for knowledge and contribution to follow along.

The complete code can be found by the end of the article as Google Colab links.

The Plastic Waste Problem
-------------------------

According to recent studies, 8 million tons of plastic are dumped into the ocean every year and it is estimated that by [2050 there will be more plastic in the ocean than fish](https://www.science.org/doi/10.1126/science.1260352 " 2050 there will be more plastic in the ocean than fish") \[1\]. 

The problem doesn’t end there.

Due to prolonged exposure to the sun, water, and air, the plastic is eventually broken down into microplastic which is eaten by the fish, other sea mammals, and birds. Each year, [12,000 to 24,000 tons of plastic are consumed by fish](https://www.biologicaldiversity.org/campaigns/ocean_plastics "12,000 to 24,000 tons of plastic are consumed by fish") in the North Pacific, causing intestinal injury and death, as well as passing plastic up the food chain to larger fish, marine mammals, and human seafood diners \[2\]. 

[

![](https://learnopencv.com/wp-content/uploads/2021/05/2__opencv_02-2.jpg "2__opencv_02 – LearnOpenCV ")

](http://opencv.org/courses/)

**Official OpenCV Courses**  
Start your exciting journey from an absolute Beginner to Mastery in AI, Computer Vision & Deep Learning!

[Learn More](https://opencv.org/courses/)

When microplastics eventually dissolve over a period of 400 years, [toxins are released in most cases](https://www.conservation.org/stories/ocean-pollution-11-facts-you-need-to-know "toxins are released in most cases"), further polluting the sea \[3\].

[

![Photo by Naja Bertolt Jensen on Unsplash](https://learnopencv.com/wp-content/uploads/2022/01/Plastic-Waste-in-oceans-Photo-by-Naja-Bertolt-Jensen-on-Unsplash.jpg "Plastic-Waste-in-oceans-Photo-by-Naja-Bertolt-Jensen-on-Unsplash – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Plastic-Waste-in-oceans-Photo-by-Naja-Bertolt-Jensen-on-Unsplash.jpg)

Photo by [Naja Bertolt Jensen](https://unsplash.com/@naja_bertolt_jensen?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

Marine litter is accumulated from the land but also from the sea. Land-based marine litter consists of landfills, industrial emissions, municipal sewerage, and others, whereas sea-based litter is mainly caused by fishing and ship waste dumping \[4\]\[5\]. 

Unless we take deliberate steps to mitigate the problem, the challenges listed above will undoubtedly intensify. In the next section, we will see how computer vision can help reduce them.

**Computer Vision and Deep Learning to the Rescue**
---------------------------------------------------

Computer vision is a very large field of Artificial Intelligence both in terms of breadth and depth. The branch of Artificial Intelligence that solves problems using deep neural networks is called **Deep Learning**.

In this part, we will briefly go over 3 fundamental tasks that are used for many applications ranging from augmented reality to self-driving cars. 

These 3 tasks are image classification, object detection, and image segmentation.

[

![Spatial Localization and Detection ](https://learnopencv.com/wp-content/uploads/2022/01/Photo-by-Fei-Fei-Li-Andrej-Karpathy-Justin-Johnson-2016-cs231n-Lecture-8 — Slide-8-Spatial-Localization-and-Detection.jpg "Photo-by-Fei-Fei-Li-Andrej-Karpathy-Justin-Johnson-2016-cs231n-Lecture-8 — Slide-8-Spatial-Localization-and-Detection – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Photo-by-Fei-Fei-Li-Andrej-Karpathy-Justin-Johnson-2016-cs231n-Lecture-8 — Slide-8-Spatial-Localization-and-Detection.jpg)

Photo by [Fei-Fei Li, Andrej Karpathy & Justin Johnson (2016) cs231n, Lecture 8 — Slide 8, _Spatial Localization and Detection_](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf)

As illustrated in the image above, [image classification](https://learnopencv.com/neural-networks-a-30000-feet-view-for-beginners/ "image classification") answers the question “**what** is this image?” 

In other words, the output of an image classifier is a **class label**. 

Object detection and image segmentation on the other hand also provide information on **where** is the person or the item that we are interested in. 

[Object detection](https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/ "Object detection") does that by returning a class label along with a bounding box  for each detection. We can visualize the bounding box by drawing rectangles over the original image. 

Finally, [image segmentation](https://learnopencv.com/image-segmentation/ "image segmentation") provides class labels for each pixel in an image \[6\]. 

As you can see, image classification provides no information about the location of the object. Object detection provides a rough localization of the object using a bounding box. Image Segmentation provides the localization information at pixel level which is an even finer level of granularity. 

**Data Collection** for Plastic Waste Detection
-----------------------------------------------

Having sufficient amounts of high-quality data is probably the most important part of any machine learning project as every machine learning algorithm requires some form of data to train. 

Supervised Machine learning-based computer vision algorithms are usually trained with the same kind of information that it expects as output. For example, in object detection, the input is an image and the output is a set of class labels and bounding boxes. To train an object detector, we need a dataset of images, and for every image in the dataset, we need bounding boxes around objects and their corresponding class labels. 

Thankfully, there is an abundance of high-quality datasets for litter detection and classification. A complete list of those datasets can be found under this [GitHub repository](https://github.com/AgaMiko/waste-datasets-review) \[7\]. 

In case you did not find a dataset to solve your specific problem then you can build your own by collecting and annotating images. There are many ways to collect images for machine learning such as taking pictures with a camera, web image scraping, or generating synthetic images. In the next section, we will see how to scrape and annotate google images to build our dataset.

### **Data Collection by Google Image Scraping**

Web scraping in general is a technique used to extract all kinds of data from websites. In this project, we use web scraping as a means to collect images for our dataset. Before you use the scraper, I recommend doing a real Google search to see if the images it returns are relevant to your problem. The following code demonstrates how to extract photos from a Google search for “plastic bottle”.

[

![Image Scrapper Code](https://learnopencv.com/wp-content/uploads/2022/01/Part-of-the-Image-Scraper-code.png "Part-of-the-Image-Scraper-code – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Part-of-the-Image-Scraper-code.png)

Part of the Image Scraper code (Image by Author)

Once the scraper is done with the images, we then run a script to delete any duplicate images that may have appeared in our google search before proceeding with the annotation.

[

![Code for Deletion of duplicate images ](https://learnopencv.com/wp-content/uploads/2022/01/Deletion-of-duplicate-images-code-Image-by-Author.png "Deletion-of-duplicate-images-code-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Deletion-of-duplicate-images-code-Image-by-Author.png)

Deletion of duplicate images code (Image by Author)

### Image **Annotating for Object Detection**

Before starting with the annotations, I highly recommend watching this short [YouTube video](https://youtu.be/pJaM06FG-wQ) from Roboflow for some tips on how to annotate images correctly for object detection. For this project, I have annotated my images with the main 4 classes of litter, plastic, metal, paper, and glass. The following image presents the annotation tool which is used within Google Colab.

[

![Demonstration of Image Annotation tool](https://learnopencv.com/wp-content/uploads/2022/01/Demonstration-of-the-annotation-tool-Image-by-Author.png "Demonstration-of-the-annotation-tool-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Demonstration-of-the-annotation-tool-Image-by-Author.png)

Demonstration of the [annotation tool](https://github.com/gereleth/jupyter-bbox-widget)

**Data Preparation and Model Training**
---------------------------------------

After we’re done building our dataset, we then split our data into train, validation, and test subsets. Usually, the splitting is done with a ratio of 70% training data and 15% validation and test data (70–15–15 for short). We could also split them with 60–20–20 or even 50–25–25, it depends on a variety of factors but the best ratio for a specific problem is generally found through trial and error.

[

![Code For Dataset splitting using the split-folders library](https://learnopencv.com/wp-content/uploads/2022/01/Dataset-splitting-code-using-the-split-folders-library-Image-by-Author.png "Dataset-splitting-code-using-the-split-folders-library-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Dataset-splitting-code-using-the-split-folders-library-Image-by-Author.png)

Dataset splitting code using the [split-folders library](https://github.com/jfilter/split-folders)

For an easy-to-understand and implement solution, we will use the imageAI library to train and test our model. This library contains the YoloV3 model, which is one of the most popular and well-rounded models for object detection. We will download and use a pre-trained model since it will require fewer data but also fewer iterations to train. 

Training a pre-trained model is a technique commonly known as **transfer learning** and is considered a good practice among machine learning practitioners \[8\]. After preparing the data and downloading our model, the training takes just 5 lines of code.

[

![Code for training model using ImageAI](https://learnopencv.com/wp-content/uploads/2022/01/Model-training-code-using-ImageAI-Image-by-Author.png "Model-training-code-using-ImageAI-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Model-training-code-using-ImageAI-Image-by-Author.png)

Model training code using [ImageAI](https://github.com/OlafenwaMoses/ImageAI)

Deep Learning **Model Testing and Evaluation**
----------------------------------------------

Finally, the most fun part of our process is testing. Once we are done training our model, we can test it using images from our test set or even run it on videos. Testing the model with imageAI takes 6 lines of code with 2 additional lines to display the image and its corresponding bounding boxes. 

[

![Code for testing Model](https://learnopencv.com/wp-content/uploads/2022/01/Model-testing-code-using-ImageAI-Image-by-Author.png "Model-testing-code-using-ImageAI-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Model-testing-code-using-ImageAI-Image-by-Author.png)

Model testing code using [ImageAI](https://github.com/OlafenwaMoses/ImageAI)

Running the code on a testing image produces the following image output. 

[

![Image with Model predictions](https://learnopencv.com/wp-content/uploads/2022/01/Image-with-model-predictions-Image-by-Author.jpg "Image-with-model-predictions-Image-by-Author – LearnOpenCV ")

](https://learnopencv.com/wp-content/uploads/2022/01/Image-with-model-predictions-Image-by-Author.jpg)

Image with model predictions

We can observe that the plastic bounding box is on the bottle but it does not contain the whole item. Moreover, there are some false positives of paper that could be filtered by increasing the minimum threshold. The numbers indicate the confidence percentage of the model for each detection which is relatively low. It can be increased by training the model further but most importantly by collecting more data.

### **Next Steps**

This article provides one of the simplest ways to collect data and train an object detector but it sure isn’t the best nor the most efficient way.

*   For a faster and more smooth annotation process, I encourage you to look for a more sophisticated annotation tool and annotate your images locally. 
*   When we are done collecting and annotating our data, it is considerably faster to pull them from GitHub instead of copying them from google drive. This can help reduce training times dramatically.
*   Even though ImageAI is an incredible tool that is extremely easy to use compared to other tools and frameworks, using a lower-level framework such as PyTorch and Tensorflow could provide more training speed and flexibility.
*   Image segmentation would be more useful for automatic litter collection using robotics. It provides higher accuracy as it applies per pixel detection and the good part is that there is an abundance of [datasets](https://github.com/AgaMiko/waste-datasets-review) available.

### **Google Colab Links**

[https://colab.research.google.com/drive/1VG6VwqxyIJH9YMuixvUEqznfsUSjCdsD?usp=sharing](https://colab.research.google.com/drive/1VG6VwqxyIJH9YMuixvUEqznfsUSjCdsD?usp=sharing)

[https://colab.research.google.com/drive/1o-CcNk1A1ENaf6XwJ4YAd9lPQkfHo2tJ?usp=sharing](https://colab.research.google.com/drive/1o-CcNk1A1ENaf6XwJ4YAd9lPQkfHo2tJ?usp=sharing)

#### **References**

\[1\] Jambeck, Jenna R., et al. “Plastic waste inputs from land into the ocean.” _Science_ 347.6223 (2015): 768–771.

\[2\] “Ocean Plastics Pollution.” _Ocean Plastics Pollution_, [https://www.biologicaldiversity.org/campaigns/ocean\_plastics](https://www.biologicaldiversity.org/campaigns/ocean_plastics/.)

\[3\] “Ocean Pollution: 11 Facts You Need to Know.” _Ocean Pollution — 11 Facts You Need to Know_, [https://www.conservation.org/stories/ocean-pollution-11-facts-you-need-to-know](https://www.conservation.org/stories/ocean-pollution-11-facts-you-need-to-know)

\[4\] [GES, MSFD. “Identifying Sources of Marine Litter.” (2016)](https://mcc.jrc.ec.europa.eu/documents/201703030936.pdf)

\[5\] “Our Oceans, Seas and Coasts.” _Marine Litter — GES — Environment — European Commission_, [](https://ec.europa.eu/environment/marine/good-environmental-status/descriptor-10/index_en.htm.) [https://ec.europa.eu/environment/marine/good-environmental-status/descriptor-10/index\_en.htm](https://ec.europa.eu/environment/marine/good-environmental-status/descriptor-10/index_en.htm)

\[6\] Fletcher, Natalie. “A Comparison of Classification vs Detection vs Segmentation Models.” _Clarifai_, [https://www.clarifai.com/blog/class](https://www.clarifai.com/blog/classification-vs-detection-vs-segmentation-models-the-differences-between-them-and-how-each-impact-your-results.)[i](https://www.clarifai.com/blog/classification-vs-detection-vs-segmentation-models-the-differences-between-them-and-how-each-impact-your-results.)[fication-vs-detection-vs-segmentation-models-the-differences-between-them-and-how-each-impact-your-results](https://www.clarifai.com/blog/classification-vs-detection-vs-segmentation-models-the-differences-between-them-and-how-each-impact-your-results.)

\[7\] [https://github.com/AgaMiko/waste-datasets-review](https://github.com/AgaMiko/waste-datasets-review)

\[8\] Brownlee, Jason. “A Gentle Introduction to Transfer Learning for Deep Learning.” _Machine Learning Mastery_, 16 Sept. 2019, [https://machinelearningmastery.com/transfer-learning-for-deep-learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/.)

#### **Tools**

[https://github.com/gereleth/jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget)

[https://github.com/jfilter/split-folders](https://github.com/jfilter/split-folders)

[https://github.com/OlafenwaMoses/ImageAI](https://github.com/OlafenwaMoses/ImageAI)

[https://github.com/AndrewCarterUK/pascal-voc-writer](https://github.com/AndrewCarterUK/pascal-voc-writer)

[https://github.com/elisemercury/Duplicate-Image-Finder](https://github.com/elisemercury/Duplicate-Image-Finder)
