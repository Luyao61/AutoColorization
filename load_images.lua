
require 'torch'
require 'image'

local module = {}

function module.load(path, n)
    h = 128
    w = h

    images = torch.Tensor(n, 3, h, w)

    for i=1,n do
--        print('loading', i)
        I = image.load(string.format('%s/image%05d.jpg', path, i-1))
--        print('done with image load')
        images[i] = I
--        print('done loading')
    end

--    print('after for loop')

    return images
end

return module
