--1021
load_images = require 'load_images'
require 'image'
require 'nn'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')

n_images = 750
timer = torch.Timer()


images = load_images.load('face_images', n_images)

-- shuffle the data
argumented_images = torch.Tensor(n_images*10,3,128,128)


rand = torch.randperm(n_images)
for i = 1,n_images do
  argumented_images[i] = images[rand[i]]
end

--argumented data by factor of 10
for i=2,10 do
  rand = torch.randperm(n_images)
  for j=1,n_images do
    x = math.random(3)
    if x == 1 then
      --print(1)
      -- do horizontal filps
      argumented_images[(i-1) * n_images + j] = image.hflip(images[rand[j]])
    elseif x == 2 then
      --print(2)
      -- do random cops
      randCrop = math.random(64)
      argumented_images[(i-1) * n_images + j] = image.scale(image.crop(images[rand[j]],'c',randCrop,randCrop),128,128)
      --print(:clone()):size())
    elseif x == 3 then
      --print(3)
      -- scaling of RGB value, ramdon chonse from [0.6,1.0]
      scale = math.random()*0.4 + 0.6
      argumented_images[(i-1) * n_images + j] = images[rand[j]] * scale
    end
  end
end

rand = torch.randperm(n_images*10)
images_lab = torch.Tensor(n_images*10,3,128,128)

for i = 1,n_images*10 do
  images_lab[i] = image.rgb2lab(argumented_images[rand[i]]:clone())
end


-- create neural network
model = nn.Sequential()  -- make a multi-layer perceptron
model:add(nn.SpatialConvolution(1, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(2, 2, 2, 2, 2, 2))
model:add(nn.ReLU(true))

-- preprocess data
batchInputs = torch.Tensor(n_images*10, 1, 128, 128)
batchLabels = torch.Tensor(n_images*10, 2)
for i = 1, n_images*10 do
   batchInputs[i]:copy(images_lab[{{i},{1},{},{}}]):div(100)
   batchLabels[{i,1}] = images_lab[{{i},{2},{},{}}]:mean()
   batchLabels[{i,2}] = images_lab[{{i},{3},{},{}}]:mean()
end

criterion = nn.MSECriterion()
params, gradParams = model:getParameters()
local optimState = {learningRate = 0.01}

for epoch = 1, 10 do
  mse = 0
   function feval(params)
      gradParams:zero()

      local outputs = model:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      model:backward(batchInputs, dloss_doutputs)

      mse = mse + loss
      return loss, gradParams
   end
   optim.sgd(feval, params, optimState)
   print(("epoch = %d; mse = %.6f"):format(epoch,mse))
end

print('Time elapsed ' .. timer:time().real .. ' seconds')
