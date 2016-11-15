--1021
load_images = require 'load_images'
require 'image'
require 'nn'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')

n_images = 750
images = load_images.load('face_images', n_images)

train_images = torch.Tensor(6750,3,128,128)
test_images = torch.Tensor(75,3,128,128)
rand = torch.randperm(n_images)
-- divide dataset into 2 parts; 90% for train; 10% for test
-- shuffle the data
for i = 1,675 do
  train_images[i] = images[rand[i]]
end
for i = 676,750 do
  test_images[i-675] = images[rand[i]]
end

--argumente train data by factor of 10
for i=2,10 do
  rand = torch.randperm(675)
  for j=1,675 do
    x = math.random(3)
    if x == 1 then
      -- do horizontal filps
      train_images[(i-1) * 675 + j] = image.hflip(train_images[rand[j]])
    elseif x == 2 then
      -- do random cops
      randCrop = math.random(64)
      train_images[(i-1) * 675 + j] = image.scale(image.crop(train_images[rand[j]],'c',randCrop,randCrop),128,128)
      --print(:clone()):size())
    elseif x == 3 then
      -- scaling of RGB value, ramdon chonse from [0.6,1.0]
      scale = math.random()*0.4 + 0.6
      train_images[(i-1) * 675 + j] = train_images[rand[j]] * scale
    end
  end
end

-- create trainData and testData
trainData = {
  data = torch.Tensor(6750, 1, 128, 128),
  labels = torch.Tensor(6750, 2, 128, 128),
  size = function() return 6750 end
}
for i = 1,trainData:size() do
  local img = train_images[i]
  img = image.rgb2lab(img)
  --preprocessing L channel
  trainData.data[i]:copy(img[{{1}, {}, {}}]):div(100)
  trainData.labels[{{i},{1},{},{}}]:copy(img[{{2},{},{}}])
  trainData.labels[{{i},{2},{},{}}]:copy(img[{{3},{},{}}])
end

testData = {
  data = torch.Tensor(75, 1, 128, 128),
  labels = torch.Tensor(75, 2, 128, 128),
  size = function() return 75 end
}
for i = 1,testData:size() do
  local img = test_images[i]
  img = image.rgb2lab(img)
  --preprocessing L channel
  testData.data[i]:copy(img[{{1}, {}, {}}]):div(100)
  testData.labels[{{i},{1},{},{}}]:copy(img[{{2},{},{}}])
  testData.labels[{{i},{2},{},{}}]:copy(img[{{3},{},{}}])
end
print('ok')
-- create neural network
model = nn.Sequential()  -- make a multi-layer perceptron
model:add(nn.SpatialConvolution(1, 2, 5, 5, 2, 2, 2, 2))
model:add(nn.SpatialBatchNormalization(2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 8, 5, 5, 2, 2, 2, 2))
model:add(nn.SpatialBatchNormalization(8))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(8, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 64, 5, 5, 2, 2, 2, 2))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, 2, 2))
model:add(nn.SpatialBatchNormalization(256))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(512, 512, 5, 5, 2, 2, 2, 2))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.ReLU(true))
model:add(nn.SpatialFullConvolution(512,512,2,2,2,2))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.ReLU(true))

model:add(nn.SpatialFullConvolution(512,256,2,2,2,2))
model:add(nn.SpatialBatchNormalization(256))
model:add(nn.ReLU(true))
model:add(nn.SpatialFullConvolution(256,128,3,3,1,1,1,1))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))

model:add(nn.SpatialFullConvolution(128,64,2,2,2,2))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(nn.SpatialFullConvolution(64,32,3,3,1,1,1,1))
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))

model:add(nn.SpatialFullConvolution(32,16,2,2,2,2))
model:add(nn.SpatialBatchNormalization(16))
model:add(nn.ReLU(true))
model:add(nn.SpatialFullConvolution(16,4,3,3,1,1,1,1))
model:add(nn.SpatialBatchNormalization(4))
model:add(nn.ReLU(true))

model:add(nn.SpatialFullConvolution(4,2,2,2,2,2))
model:add(nn.SpatialBatchNormalization(2))
model:add(nn.ReLU(true))



parameters,gradParameters = model:getParameters()
criterion = nn.MSECriterion()
optimState = {
  learningRate = 0.01
}
optimMethod = optim.sgd
batchSize = 5

for epoch = 1,5 do

  shuffle = torch.randperm(6750)
  local f = 0
  model:training()
  for t = 1,6750,batchSize do
    --local inputs = {}
    --local targets = {}
    local inputs = torch.Tensor(batchSize,1,128,128)
    local targets = torch.Tensor(batchSize,2,128,128)
    for i = t,t+batchSize-1 do
      local input = trainData.data[shuffle[i]]
      local target = trainData.labels[shuffle[i]]
      inputs[i - t + 1] = input
      targets[i - t + 1] = target
      --table.insert(inputs, input)
      --table.insert(targets, target)
    end
    local feval = function(x)
      if x~= parameters then
        parameters:copy(x)
      end

      gradParameters:zero()

      local output = model:forward(inputs)
      local err = criterion:forward(output, targets)
      f = f + err
      local df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)

      return f,gradParameters
    end
    optimMethod(feval, parameters, optimState)
  end
  print(("epoch = %d; train mse = %.6f"):format(epoch,f/6750))


  shuffle = torch.randperm(75)
  f=0
  model:evaluate()
  for t = 1,75,batchSize do
    local inputs = torch.Tensor(batchSize,1,128,128)
    local targets = torch.Tensor(batchSize,2,128,128)
    for i = t,t+batchSize-1 do
      local input = testData.data[shuffle[i]]
      local target = testData.labels[shuffle[i]]
      inputs[i - t + 1] = input
      targets[i - t + 1] = target
    end
    local output = model:forward(inputs)
    local err = criterion:forward(output, targets)
    f = f + err
  end
  print(("epoch = %d; test mse = %.6f"):format(epoch,f/75))
end

------------------------------------------------------------
final_result = torch.Tensor(75,3,128,128)
mse_a = 0
mse_b = 0
mse = 0
for t = 1,75,batchSize do
  local inputs = torch.Tensor(batchSize,1,128,128)
  local targets = torch.Tensor(batchSize,2,128,128)
  for i = t,t+batchSize-1 do
    local input = testData.data[i]
    local target = testData.labels[i]
    inputs[i - t + 1] = input
    targets[i - t + 1] = target
  end
  local output = model:forward(inputs)
  local err = criterion:forward(output, targets)
  mse = mse + err
  mse_a = mse_a + criterion:forward(output[{{},{1},{},{}}], targets[{{},{1},{},{}}])
  mse_b = mse_b + criterion:forward(output[{{},{2},{},{}}], targets[{{},{2},{},{}}])

  local output_lab = torch.Tensor(5,3,128,128)
  for i = 1,5 do
    output_lab[{{i},{1},{},{}}]:copy(inputs[{{i},{1},{},{}}]):mul(100)
    output_lab[{{i},{2},{},{}}]:copy(output[{{i},{1},{},{}}])
    output_lab[{{i},{3},{},{}}]:copy(output[{{i},{2},{},{}}])
  end
  for i = 1,5 do
    final_result[t+i-1]:copy(image.lab2rgb(output_lab[i]))
  end
end
print(("MSE for a*: %.6f; MSE for b*: %6f"):format(mse_a/75,mse_b/75))
print(("MSE = %6f"):format(mse/75))
image.display(final_result)
image.display(testData.data)
image.display(test_images)
