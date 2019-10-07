import numpy as np
import matplotlib.pyplot as plt
import mmcv
oldflow = np.load('./oldnorm.pickle')
newflow = np.load('./newnorm.pickle')

gt = mmcv.flowread('./gt.flo')

print(np.linalg.norm(oldflow-gt,axis=2).mean())
print(np.linalg.norm(newflow-gt,axis=2).mean())

plt.plot(oldflow[192,:,0],label='old')
plt.plot(newflow[192,:,0],label='new')
plt.legend()

fig,axes = plt.subplots(1,2)
axes[0].imshow(mmcv.flow2rgb(oldflow))
axes[1].imshow(mmcv.flow2rgb(newflow))
plt.show()
