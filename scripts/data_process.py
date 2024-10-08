import numpy as np
import h5py, os, pickle, imageio


def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std


# out_kspace_dir = '../mri_recon/DDPM/noscale/train/kspace/'
# visual_dir = '../mri_recon/DDPM/noscale/train/visual/'

# out_img_dir = './data/val/img/'
# inp_dir = '../mri_recon/val/'

# out_img_dir = 'F:\\fastmri\\trainImg\\'
# inp_dir = 'F:\\fastmri\\train\\'

out_img_dir = 'D:\\Research_Topic\\医学超分\\DiffuseRecon-main\\scripts\\data\\val\\img10\\'
inp_dir = 'D:\\Research_Topic\\医学超分\\DiffuseRecon-main\\mri_recon\\val\\'

os.makedirs(out_img_dir, exist_ok=True)

files = os.listdir(inp_dir)
for file in files:
    data = h5py.File(inp_dir + file, 'r')['kspace']
    for i in range(data.shape[0]):
        # norm_kspace,std_kspace = normalize_complex(data[i])
        # img = np.fft.ifft2(norm_kspace)
        # img = np.fft.fftshift(img)
        # print(file,i)
        # pickle.dump({'img':norm_kspace},open(out_img_dir+file.replace('.h5','_'+str(i)+'.pt'),
        #                                                          'wb'))

        img = np.fft.ifft2(data[i])
        img = np.fft.fftshift(img)
        print(file, i)
        pickle.dump({'img': img}, open(out_img_dir + file.replace('.h5', '_' + str(i) + '.pt'),
                                       'wb'))
