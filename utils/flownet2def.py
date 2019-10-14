
def get_layer_details(cnn_layer_idx,inp_h,inp_w):
    filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]
    recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
    stride_power = [1,2,3,3,4,4,5,5,6,6,6]
    layer_names = ['conv1','conv2','conv3','conv3_1','conv4','conv4_1','conv5','conv5_1','conv6','conv6_1','predict_flow6']

    if cnn_layer_idx == 0:
        nr_of_samples = 350
    elif cnn_layer_idx == 1:
        nr_of_samples = 700
    else:
        nr_of_samples= 1041

    layer_name = layer_names[cnn_layer_idx]
    recep_field_layer = recep_field[cnn_layer_idx]
    reduction_power = stride_power[cnn_layer_idx]
    fmap_h, fmap_w = int(inp_h / (2 ** (reduction_power))), int(inp_w / (2 ** (reduction_power)))
    filters_layer = filters_nr[cnn_layer_idx]

    return nr_of_samples, layer_name, fmap_h, fmap_w, filters_layer, recep_field_layer, reduction_power