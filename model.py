import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import sys
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)
import _pickle as cPickle
from random import randint


x = tf.placeholder('float')
y = tf.placeholder('int32')
mode=tf.placeholder('bool')



BATCH_SIZE=256
LAMBDHA=0.005
LEARNING_RATE=0.001

variables_dict={
		"weightConvl1": tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32],mean=0.0,stddev=0.1,dtype=tf.float32),
	        name="weightConvl1"),
		"weightConvl2": tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32],mean=0.0,stddev=0.1),
	        name="weightConvl2"),
		"weightConvl3": tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64],mean=0.0,stddev=0.1 ),
	        name="weightConvl3"),
		"weightConvl4": tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64],mean=0.0,stddev=0.1 ),
	        name="weightConvl4"),
		"weightConvl5": tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128],mean=0.0,stddev=0.1 ),
	        name="weightConvl5"),
		"weightConvl6": tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128],mean=0.0,stddev=0.1 ),
	        name="weightConvl6"),
		"weightConvl7": tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256],mean=0.0,stddev=0.1 ),
	        name="weightConvl7"),
		"weightConvl8": tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256],mean=0.0,stddev=0.1 ),
	        name="weightConvl8"),
		"weightFC1": tf.Variable(tf.truncated_normal(shape=[1024,1024],mean=0.0,stddev=0.1 ),
	        name="weightFC1"),
		"weightFC2": tf.Variable(tf.truncated_normal(shape=[1024,512],mean=0.0,stddev=0.1 ),
	        name="weightFC2"),
		"weightFC3": tf.Variable(tf.truncated_normal(shape=[512,10],mean=0.0,stddev=0.1 ),
	        name="weightFC3"),
        "bias1": tf.Variable(tf.truncated_normal(shape=[1024],mean=1.0,stddev=0.1 ),
	        name="bias1"),
        "bias2": tf.Variable(tf.truncated_normal(shape=[1,512],mean=1.0,stddev=0.1 ),
	        name="bias2"),
	    "bias3": tf.Variable(tf.truncated_normal(shape=[1,10],mean=1.0,stddev=0.1 ),
	        name="bias3")      
}

def getData():
	pathname = os.path.dirname(sys.argv[0])
	fullpath = os.path.abspath(pathname) 
	
	with open(pathname+"/cifar-10-batches-py/data_batch_1", 'rb') as fo:
		
		dict1 = cPickle.load(fo,encoding='bytes')
	with open(pathname+"/cifar-10-batches-py/data_batch_2", 'rb') as fo:
		
		dict2 = cPickle.load(fo,encoding='bytes')
	with open(pathname+"/cifar-10-batches-py/data_batch_3", 'rb') as fo:
	
		dict3= cPickle.load(fo,encoding='bytes')
	with open(pathname+"/cifar-10-batches-py/data_batch_4", 'rb') as fo:
	
		dict4 = cPickle.load(fo,encoding='bytes')
	with open(pathname+"/cifar-10-batches-py/data_batch_5", 'rb') as fo:
	
		dict5 = cPickle.load(fo,encoding='bytes')
	with open(pathname+"/cifar-10-batches-py/test_batch", 'rb') as fo:
		
		dict6 = cPickle.load(fo,encoding='bytes')
	inputData_test=dict6[b'data']
	labels_test=dict6[b'labels']

	inputData=dict1[b'data']
	labels=dict1[b'labels']
	
	inputData=np.concatenate((inputData,dict2[b'data']))
	labels=np.concatenate((labels,dict2[b'labels']))	
	inputData=np.concatenate((inputData,dict3[b'data']))
	labels=np.concatenate((labels,dict3[b'labels']))	
	inputData=np.concatenate((inputData,dict4[b'data']))
	labels=np.concatenate((labels,dict4[b'labels']))	
	inputData=np.concatenate((inputData,dict5[b'data']))
	labels=np.concatenate((labels,dict5[b'labels']))	
	
	labels=np.matrix(labels)
	inputData=np.matrix(inputData)

	labels=np.matrix(labels)
	inputData_test=np.matrix(inputData_test)
	
	return inputData.astype(np.float32),labels.transpose(),inputData_test.astype(np.float32),labels_test


def neuralNetwork(input_layer,mode):
	conv1 = tf.nn.relu(tf.nn.conv2d(input=input_layer, filter=variables_dict["weightConvl1"],strides=[1,1,1,1],padding="SAME"))
	conv2 = tf.nn.conv2d(input=conv1, filter=variables_dict["weightConvl2"],strides=[1,1,1,1],padding="SAME")
	relu1=tf.nn.relu(features=conv2)
	pool1=tf.nn.max_pool(value=relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")	
	norm1=tf.layers.batch_normalization(pool1,axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer=tf.zeros_initializer(),gamma_initializer=tf.ones_initializer(),moving_mean_initializer=tf.zeros_initializer(),moving_variance_initializer=tf.ones_initializer(),beta_regularizer=None,gamma_regularizer=None,training=False,
    trainable=True )	
	
	layerDrop1=tf.layers.dropout(pool1,0.2,training=mode)
	
	conv3 = tf.nn.relu(tf.nn.conv2d(input=layerDrop1, filter=variables_dict["weightConvl3"],strides=[1,1,1,1],padding="SAME"))
	conv4 = tf.nn.conv2d(input=conv3, filter=variables_dict["weightConvl4"],strides=[1,1,1,1],padding="SAME")
	relu2=tf.nn.relu(features=conv4)
	pool2=tf.nn.max_pool(value=relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")	
	norm2=tf.layers.batch_normalization(pool2,axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer=tf.zeros_initializer(),gamma_initializer=tf.ones_initializer(),moving_mean_initializer=tf.zeros_initializer(),moving_variance_initializer=tf.ones_initializer(),beta_regularizer=None,gamma_regularizer=None,training=False,
    trainable=True )
	relu2=tf.nn.relu(features=norm2)
	layerDrop2=tf.layers.dropout(relu2,0.2,training=mode)	
	
	conv5 = tf.nn.relu(tf.nn.conv2d(input=layerDrop2, filter=variables_dict["weightConvl5"],strides=[1,1,1,1],padding="SAME"))
	conv6 = tf.nn.conv2d(input=conv5, filter=variables_dict["weightConvl6"],strides=[1,1,1,1],padding="SAME")
	relu3=tf.nn.relu(features=conv6)
	pool3=tf.nn.max_pool(value=relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")	
	norm3=tf.layers.batch_normalization(pool3,axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer=tf.zeros_initializer(),gamma_initializer=tf.ones_initializer(),moving_mean_initializer=tf.zeros_initializer(),moving_variance_initializer=tf.ones_initializer(),beta_regularizer=None,gamma_regularizer=None,training=False,
    trainable=True )
	relu3=tf.nn.relu(features=norm3)
	layerDrop3=tf.layers.dropout(relu3,0.2,training=mode)
	

	conv7 = tf.nn.relu(tf.nn.conv2d(input=layerDrop3, filter=variables_dict["weightConvl7"],strides=[1,1,1,1],padding="SAME"))
	conv8 = tf.nn.conv2d(input=conv7, filter=variables_dict["weightConvl8"],strides=[1,1,1,1],padding="SAME")
	relu4=tf.nn.relu(features=conv8)
	pool4=tf.nn.max_pool(value=relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")	
	norm4=tf.layers.batch_normalization(pool4,axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer=tf.zeros_initializer(),gamma_initializer=tf.ones_initializer(),moving_mean_initializer=tf.zeros_initializer(),moving_variance_initializer=tf.ones_initializer(),beta_regularizer=None,gamma_regularizer=None,training=False,
    trainable=True )
	relu4=tf.nn.relu(features=norm4)
	layerDrop4=tf.layers.dropout(relu4,0.2,training=mode)
	


	
	inp=tf.transpose(layerDrop4,perm=[0,3,1,2])
	inputFC=tf.reshape(inp,[-1,1024])
	
	layerFC1=tf.nn.relu(tf.add(tf.matmul(inputFC,variables_dict["weightFC1"]),variables_dict["bias1"]))
	layerDrop5=tf.layers.dropout(layerFC1,0.5,training=mode)
	layerFC2=tf.nn.relu(tf.add(tf.matmul(layerDrop5,variables_dict["weightFC2"]),variables_dict["bias2"]))
	layerDrop6=tf.layers.dropout(layerFC2,0.5,training=mode)
	layerFC3=tf.add(tf.matmul(layerDrop6,variables_dict["weightFC3"]),variables_dict["bias3"])  
	
	return layerFC3

def main():
	
	[inputData,labels,inputData_test,labels_test]=getData()
	tempData=np.transpose(np.reshape(np.asarray(inputData),(-1,3, 32,32)), (0,2,3,1))
	tempData_test=np.transpose(np.reshape(np.asarray(inputData_test),(-1,3, 32,32)), (0,2,3,1))

	layerFC2=neuralNetwork(x,mode)	

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layerFC2, labels=y))
	tloss=loss+(LAMBDHA/BATCH_SIZE)*(tf.reduce_sum(tf.multiply(variables_dict["weightFC1"],variables_dict["weightFC1"]))+tf.reduce_sum(tf.multiply(variables_dict["weightFC2"],variables_dict["weightFC2"]))+tf.reduce_sum(tf.multiply(variables_dict["weightFC3"],variables_dict["weightFC3"])))
	optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(tloss)
	
	imageFlip=np.zeros((3472,32,32,3))
	imageContrast=np.zeros((3472,32,32,3))
	imageBrightness=np.zeros((3472,32,32,3))
	labelFlip=np.zeros((3472,1))
	labelBright=np.zeros((3472,1))
	labelContrast=np.zeros((3472,1))
	
	
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())
		with tf.device('/gpu:0'):
			d=tf.placeholder('float')
			e=tf.placeholder('float')
			f=tf.placeholder('float')
			g=tf.placeholder('float')
			flip=tf.image.random_flip_left_right(d)	
			bright=tf.image.random_brightness(e,63)
			contrast=tf.image.random_contrast(f,0.8,1.2)
			standard=tf.image.per_image_standardization(g)
				

		for i in range(3472):
			index1=randint(0,49999)
			index2=randint(0,49999)
			index3=randint(0,49999)
			imageFlip[i]=sess.run(flip,feed_dict={d:tempData[index1]})
			imageBrightness[i]=sess.run(bright,feed_dict={e:tempData[index2]})
			imageContrast[i]=sess.run(contrast,feed_dict={f:tempData[index3]})
			labelFlip[i]=labels[index1]
			labelBright[i]=labels[index2]
			labelContrast[i]=labels[index3]
			
		labelFlip=np.concatenate((labelFlip,labelBright))
		labelFlip=np.concatenate((labelFlip,labelContrast))
		labels=np.concatenate((labels,labelFlip))
		imageFlip=np.concatenate((imageFlip,imageBrightness))
		imageFlip=np.concatenate((imageFlip,imageContrast))
		tempData=np.vstack([tempData,imageFlip])
		indexData=np.zeros(tempData.shape[0]).astype('int32')
		indexData[:]=np.arange(0,tempData.shape[0],1)
		shuffleData=np.zeros(indexData.shape[0])
		shuffleData=indexData
		shuffleData=sess.run(tf.random_shuffle(shuffleData))		
		tempData=tempData[shuffleData]
		labels=labels[shuffleData]
        
		a=sess.run(tf.one_hot(indices=tf.cast(labels.transpose(), tf.int32), depth=10))
		
		tempData=np.transpose(tempData,[0,2,1,3])
		tempData_test=np.transpose(tempData_test,[0,2,1,3])
        
		
		for i in range(60416):
			tempData[i]=sess.run(standard,feed_dict={g:tempData[i]})
		
		for i in range(10000):
			tempData_test[i]=sess.run(standard,feed_dict={g:tempData_test[i]})
        
	
		
		
		for i in range(300):
			tloss1=0
			for j in range(0,60416, BATCH_SIZE):
				temp=tempData[j:j+ BATCH_SIZE,:]
				b=a[0,j:j+ BATCH_SIZE,:]							
				op,loss1=sess.run([optimizer,tloss],feed_dict={x:temp,y:b,mode:True})
				tloss1+=loss1
				
			print("i=",i,tloss1)	
			
		print("----------------------------------------------------------")
		
		
		score=sess.run([layerFC2],feed_dict={x:tempData_test,mode:False})		
		predicted_label=sess.run(tf.argmax(score[0],1))		
		count=0

		for i in range(10000):
			if predicted_label[i]==labels_test[i]:
				count=count+1	
			print(predicted_label[i],labels_test[i])
		print("accuracy=",(count/10000))		
		
main()



