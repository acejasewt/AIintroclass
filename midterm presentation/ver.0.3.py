#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 載入資料集
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

X_train, y_train = load_mnist('./data', kind='train')
X_test, y_test = load_mnist('./data', kind='t10k')


# In[3]:


# 正規化處理
ave_input = np.average(X_train, axis=0)
std_input = np.std(X_train, axis=0)
input_data = (X_train - ave_input) / std_input


# In[4]:


# 標籤種類
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9


# In[5]:


# One-hot encoding 和切割資料
correct_data = np.zeros((len(y_train), 10))

for i in range(len(y_train)):
    correct_data[i,y_train[i]]=1.0
    
n_data = len(y_train)
(X_train, X_valid) = X_train[5000:], X_train[:5000] 
(correct_train, correct_valid) = correct_data[5000:], correct_data[:5000]

n_train = X_train.shape[0] 
n_valid = X_valid.shape[0]


# In[13]:


# 設定值
n_in = 784  # 輸入層
n_mid = 28 # 中間層
n_out = 10  # 輸出層

wb_width = 0.1  # 權重與偏值
eta = 0.01  # 學習率
epoch = 1000
batch_size = 8
interval = 100


# In[14]:


# 繼承來源
class BaseLayer:
  def __init__(self, n_upper, n):
    self.w = wb_width * np.random.randn(n_upper, n)
    self.b = wb_width * np.random.randn(n)

    self.h_w = np.zeros(( n_upper, n)) + 1e-8
    self.h_b = np.zeros(n) + 1e-8
        
  def update(self, eta):      
    self.h_w += self.grad_w * self.grad_w
    self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
    self.h_b += self.grad_b * self.grad_b
    self.b -= eta / np.sqrt(self.h_b) * self.grad_b
    
# 中間層
class MiddleLayer(BaseLayer):
  def forward(self, x):
    self.x = x
    self.u = np.dot(x, self.w) + self.b
    self.y = 0.5 * (1 + np.tanh(0.5 * self.u)) # sigmoid

  def backward(self, grad_y):
    delta = grad_y *(1-self.y)*self.y  # sigmoid的微分
    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T) 
    

# 輸出層
class OutputLayer(BaseLayer):     
  def forward(self, x):
    self.x = x
    u = np.dot(x, self.w) + self.b
    self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # Softmax

  def backward(self, t):
    delta = self.y - t       
    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T) 
    
# 丟棄層 
class Dropout:
  def __init__(self, dropout_ratio):
    self.dropout_ratio = dropout_ratio  # 丟棄率

  def forward(self, x, is_train):  
    if is_train:
      rand = np.random.rand(*x.shape)  
      self.dropout = np.where(rand > self.dropout_ratio, 1, 0)  # 1:有効 0:無効
      self.y = x * self.dropout  
    else:
      self.y = (1-self.dropout_ratio)*x
        
  def backward(self, grad_y):
    self.grad_x = grad_y * self.dropout
    
# 各層的實體化
middle_layer_1 = MiddleLayer(n_in, n_mid)
dropout_1 = Dropout(0.5)
middle_layer_2 = MiddleLayer(n_mid, n_mid)
dropout_2 = Dropout(0.5)
output_layer = OutputLayer(n_mid, n_out)


# In[15]:


# 前向傳播
def forward_propagation(x, is_train):
  middle_layer_1.forward(x)
  dropout_1.forward(middle_layer_1.y, is_train)
  middle_layer_2.forward(dropout_1.y)
  dropout_2.forward(middle_layer_2.y, is_train)
  output_layer.forward(dropout_2.y)

# 反向傳播
def backpropagation(t):
  output_layer.backward(t)
  dropout_2.backward(output_layer.grad_x)
  middle_layer_2.backward(dropout_2.grad_x)
  dropout_1.backward(middle_layer_2.grad_x)
  middle_layer_1.backward(dropout_1.grad_x)

# 更新權重與偏值
def uppdate_wb():
  middle_layer_1.update(eta)
  middle_layer_2.update(eta)
  output_layer.update(eta)

# 計算誤差
def get_error(t, batch_size):
  return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size  # 交叉熵誤差


# In[16]:


if __name__ == '__main__':
  # 記錄誤差
  train_error_x = []
  train_error_y = []
  test_error_x = []
  test_error_y = []
    
  # 記錄訓練進度 
  n_batch = n_train // batch_size 
    
  for i in range(epoch):
    # 計算誤差  
    forward_propagation(X_train, False)
    error_train = get_error(correct_train, n_train)
    forward_propagation(X_valid, False)
    error_test = get_error(correct_valid, n_valid)
        
    # 記錄誤差
    test_error_x.append(i)
    test_error_y.append(error_test) 
    train_error_x.append(i)
    train_error_y.append(error_train) 
        
    # 顯示進度
    if i%interval == 0:
      print("Epoch:" + str(i) + "/" + str(epoch),
         "Error_train:" + str(error_train),
         "Error_test:" + str(error_test))
    
    # 訓練 
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)  # 索引洗牌
    for j in range(n_batch):
      # 取出小批次
      mb_index = index_random[j*batch_size : (j+1)*batch_size]
      x = X_train[mb_index, :]
      t = correct_train[mb_index, :]
            
      # 前向傳播與反向傳播
      forward_propagation(x, True)
      backpropagation(t)
            
      # 更新權重與偏值
      uppdate_wb()


# In[ ]:


# 以圖表顯示誤差記錄
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# 計算準確率  
forward_propagation(X_train, False)
count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))

forward_propagation(X_valid, False)
count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_valid, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%",
      "Accuracy Test:", str(count_test/n_valid*100) + "%")


# In[ ]:




