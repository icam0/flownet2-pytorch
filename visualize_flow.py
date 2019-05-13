import mmcv
from skimage import io

gray_flow = mmcv.flow2rgb(mmcv.flowread('./gray.flo'))
io.imsave('./gray.png',gray_flow)

og_flow = mmcv.flow2rgb(mmcv.flowread('./og.flo'))
io.imsave('./og.png',og_flow)