import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import scipy.io
import yaml

from evaluation_metrics import compute_CMC_mAP

from model.make_model_finetune import make_model


try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda '
          'support (https://github.com/NVIDIA/apex)')


# Load model
def load_network(network, args):
    save_path = os.path.join(args.f_name, args.m_name, 'net_%s.pth' % args.which_epoch)  # Make sure which model to use!
    network.load_state_dict(torch.load(save_path))
    return network


# Flip image
def fliplr(img):
    """
    Flip horizontal
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# Extract feature from a trained model.# Extract feature from  a trained model.
def extract_feature(model, data_loaders, args):
    features = torch.FloatTensor()
    count = 0
    for data in data_loaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)

        if args.backbone_name == 'RN50':
            features_dim = 3072  # For CLIP, ResNet50
        elif args.backbone_name == 'ViT-B/16':
            features_dim = 1280  # For CLIP, vit-16

        ff = torch.FloatTensor(n, features_dim).zero_().cuda()
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = img.cuda()

            score, features1, features2 = model(input_img)
            if args.use_features_before_neck:
                output_ffs = features2[0]  # Features before neck_feat (look into make_model_finetune.py for more
                # details). This gives better result for ViT-B/16.
            else:
                output_ffs = features2[1]  # Features after neck_feat (look into make_model_finetune.py for more
                # details). This gives better result for RN50.

            ff += output_ffs

        # Normalize feature
        ff_norm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(ff_norm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


# Get ids
def get_id(img_path):
    labels = []
    for path, v in img_path:
        # filename = os.path.basename(path)
        # label = filename.split('_')[0]
        if os.name == 'nt':
            label = path.split('\\')[-2]  # For 11k hand data set, on Windows
        else:
            label = path.split('/')[-2]  # For 11k hand data set, on Linux
        labels.append(int(label))
    return labels


def main():
    parser = argparse.ArgumentParser(description='Testing a trained ResNet50 with MBA model for hand identification as '
                                                 'part of a H-Unique project.')
    parser.add_argument('--test_dir',
                        default='./11k/train_val_test_split_dorsal_r',
                        type=str,
                        help=' Path to test_data: '
                             './11k/train_val_test_split_dorsal_r'  './11k/train_val_test_split_dorsal_l'
                             './11k/train_val_test_split_palmar_r'  './11k/train_val_test_split_palmar_l'  # For 11k
                             './HD/Original Images/train_val_test_split')  # For HD
    parser.add_argument('--f_name', default='./model_11k_d_r', type=str,
                        help='Output folder name - '
                             './model_11k_d_r  ./model_11k_d_l  ./model_11k_p_r  ./model_11k_p_l'  # For 11k
                             'or ./model_HD'   # For HD
                             'Note: Adjust the data-type in opts.yaml when evaluating cross-domain performance.')
    parser.add_argument('--m_name', default='clip_hand_vit', type=str,
                        help='Saved model name - - clip_hand_vit OR clip_hand_rn50.')
    parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or best')
    parser.add_argument('--batch_size', default=50, type=int, help='batch_size')  # 256, 50, 14
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use: 0, 8, etc. Setting to 8 workers may run faster.')
    parser.add_argument('--fp16', action='store_true', help='use fp16.')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')

    # For CLIP
    parser.add_argument('--input_size', type=tuple, default=(224, 224), help='Image input size for training and test')
    parser.add_argument('--stride_size', type=tuple, default=(16, 16), help='Stride size for creating image patches')
    parser.add_argument('--use_features_before_neck', action='store_true', default=True,
                        help='Which features to use for evaluation: before neck or after neck. Please look into look '
                             'into make_model_finetune.py for more details.')
    # Args
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # Load the training config
    config_path = os.path.join(args.f_name, args.m_name, 'opts.yaml')

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    args.fp16 = config['fp16']
    args.data_type = config['data_type']
    args.backbone_name = config['backbone_name']

    if 'num_classes' in config:
        args.num_classes = config['num_classes']  # The number of classes the model is trained on!
    else:
        args.num_classes = 251

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # Set GPU ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # Load Data: We will use torchvision and torch.utils.data packages for loading the data, with appropriate
    # transforms.
    data_transforms = transforms.Compose([    # input_size = (224, 224)
        transforms.Resize((args.input_size[0], args.input_size[1]), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = args.test_dir

    # Load Collected data Trained model
    print('-------Test has started ------------------')

    model_structure = make_model(args, num_class=args.num_classes)

    model = load_network(model_structure, args)

    # Send model to GPU; it is recommended to use DistributedDataParallel, instead of DataParallel, to do multi-GPU
    # training, even if there is only a single node.
    model = model.eval()
    if use_gpu:
        # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model = model.cuda()

    # For N = 10 Monte Carlo runs
    # You can change the data_type in opts.yaml if you want to perform cross-domain performance evaluation!
    if args.data_type == '11k':
        galleries = ['gallery0_all', 'gallery1_all', 'gallery2_all', 'gallery3_all', 'gallery4_all', 'gallery5_all',
                     'gallery6_all', 'gallery7_all', 'gallery8_all', 'gallery9_all']  # For 11k
    elif args.data_type == 'HD':
        galleries = ['gallery0', 'gallery1', 'gallery2', 'gallery3', 'gallery4', 'gallery5', 'gallery6', 'gallery7',
                     'gallery8', 'gallery9']   # For HD
    else:
        print('Please choose the correct data type!')
        exit()

    queries = ['query0', 'query1', 'query2', 'query3', 'query4', 'query5', 'query6', 'query7', 'query8', 'query9']

    CMC_total = 0
    mAP_total = 0
    for i in range(len(galleries)):
        g = galleries[i]
        q = queries[i]

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in [g, q]}
        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                       shuffle=False, num_workers=args.num_workers) for x in [g, q]}

        gallery_path = image_datasets[g].imgs
        query_path = image_datasets[q].imgs

        gallery_label = get_id(gallery_path)
        query_label = get_id(query_path)

        # Extract feature
        with torch.no_grad():
            gallery_feature = extract_feature(model, data_loaders[g], args)
            query_feature = extract_feature(model, data_loaders[q], args)

        # Save to Matlab for check
        result_fl = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
                     'query_f': query_feature.numpy(), 'query_label': query_label}
        scipy.io.savemat('result.mat', result_fl)

        print(args.m_name)
        result = '%s/%s/result.txt' % (args.f_name, args.m_name)
        # os.system('python3 compute_accuracy.py | tee -a %s' % result_file)
        # os.system('python3 compute_CMC_mAP.py | tee -a %s' % result)

        CMC_i, mAP_i = compute_CMC_mAP(result_fl)
        CMC_total += CMC_i
        mAP_total += mAP_i

        print('Result on %s and %s is:' % (g, q))
        print('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC_i[0], CMC_i[4], CMC_i[9], mAP_i))

    # Mean over N = 10 Monte Carlo runs
    CMC = CMC_total/len(galleries)
    mAP = mAP_total/len(galleries)

    print('\nMean result over N = %s Monte Carlo runs is:' % (len(galleries)))
    print('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

    res = open(result, 'w')
    res.write('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

    # # Save for the later plot
    # res_cmc = '%s/%s/CMC_gpa.npy' % (args.f_name, args.m_name)  # CMC_gpa.npy, CMC_res50.npy, CMC_vgg.npy
    # np.save(res_cmc, CMC)

    print('-----Test is done!------------------')


# Execute from the interpreter
if __name__ == "__main__":
    main()
