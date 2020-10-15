import tensorflow.compat.v1 as tf
import random
import time

tf.disable_v2_behavior()

# 两个随机输入参数x,y
inputs = tf.placeholder(tf.float32,[2])

# 加法公式为 v = F(x,y) = a[0]*x+a[1]*y
# 训练的参数为:a[0],a[1]
a = tf.Variable(tf.random_uniform([2],0,1.0))

v = tf.reduce_sum(tf.multiply(inputs,a),0)

# x+y的真正结果，这个结果是程序求解出来的，当作监督学习的标签
y_input = tf.placeholder(tf.float32)
# 误差：ai输出的v值与真实标签值之间的误差
cost =  tf.losses.mean_squared_error(y_input,v)
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 加法，用来求解两个随机值的加和
def model_add(a,b):
	return a+b

with  tf.Session() as session:

	session.run(tf.global_variables_initializer())

	timeStamp = str(int(time.time()))

	writer = tf.summary.FileWriter('E:/Logs/'+timeStamp+'/',session.graph)
	
	print('a:',session.run(a))

	cost_list = []

	tf.summary.scalar("cost",cost)
	merged_summaries = tf.summary.merge_all()

	# 训练
	for i in range(10000):
		# 随机两个值
		test_a = random.uniform(0,10)
		test_b = random.uniform(0,10)
		array = []
		array.append(test_a)
		array.append(test_b)
		sum_value = model_add(test_a,test_b)
	
		# 优化
		optimizer.run(feed_dict={y_input:sum_value,inputs:array},session=session)

		# 计算误差，打印误差
		c = cost.eval(feed_dict={y_input:sum_value,inputs:array},session=session)
		# print(c)

		summary = session.run(merged_summaries, feed_dict={y_input:sum_value,inputs:array})
		writer.add_summary(summary=summary, global_step=i)

		#保存30个误差结果
		cost_list.append(c)
		if len(cost_list) > 30:
			cost_list.pop(0)

		# 平均误差足够小，不再训练。防止单个误差的偶然性很小导致训练结果不稳定
		if sum(cost_list)/len(cost_list) <0.00001:
			break


	#训练的参数
	print('a:',session.run(a))

	# 测试
	for i in range(10):
		test_a = random.uniform(0,5500)+5000
		test_b = random.uniform(0,5500)+5000
		array = []
		array.append(test_a)
		array.append(test_b)
		result = session.run(v,feed_dict={inputs:array})
		print(' ',test_a, ' + ',test_b,' = ',result)