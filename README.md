# Water-Quality

Hello World

hello khomsan

หา index ใน maaxscores
หา model ที่ดีที่สุด
m=max(scoresl['test_accuracy'])
a=scoresl['test_accuracy']
max_indices = np.where(a == m)[0]
print('test_accuracy max',max_indices)

เลือก model ที่ดีที่สุด
allmodels = scoresl['estimator']
models = allmodels[int(max_indices)]
print(models)