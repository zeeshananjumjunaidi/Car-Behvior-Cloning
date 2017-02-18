# Car-Behvior-Cloning
<h1>Car</h1>
<h3>Data Collection</h3>
<p>I drove around 25k frames including 1st & 2nd map in the simulator. Most of the frames are from center lane driving and 20% are from side lane recovery. I used PS4 joystick as it is only the appropriate way for proper driving. My target was to train this network only for 1st map as required for this project. I read in a blog post that at least minimum 40k images are required to train a perfect model. I used 25k to save time for training the model, and to avoid memory overflow. And it works.
Following are the details of my dataset.
</p>
<ul><li> no. of features: 	25656</li>
<li> Shape of single feature:	(100, 220, 3)</li>
<li> Shape of all features:	(25656, 100, 220, 3)</li>
<li> Shape of labels(steering):	(25656,)</li></ul>
<img src='doc_image/data_size.png'/>

Some images from the dataset: 

<img src='doc_image/center_2017_02_10_13_48_47_600.jpg'/>
<img src='doc_image/center_2017_02_10_13_52_22_872.jpg'/>
<br/>
then I cropped the images and took lower area (road) and resize it to (100,200,3):

<img src='doc_image/cropped_img.png'/>
<img src='doc_image/cropped_img1.png'/>
<img src='doc_image/cropped_img_2.png'/>
