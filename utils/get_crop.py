
def get_crop(hw_pos,stride_power,cnn_layer_idx,size,inp_h,inp_w):


    crop_h = [int(hw_pos[0] * 2 ** (stride_power[cnn_layer_idx]) - 0.5 * size),
              int(hw_pos[0] * 2 ** (stride_power[cnn_layer_idx]) + 0.5 * size)]
    crop_w = [int(hw_pos[1] * 2 ** (stride_power[cnn_layer_idx]) - 0.5 * size),
              int(hw_pos[1] * 2 ** (stride_power[cnn_layer_idx]) + 0.5 * size)]

    if crop_h[0] < 0:
        crop_h[1] = crop_h[1]-crop_h[0]
        crop_h[0] = 0

    if crop_h[1] > inp_h:
        crop_h[0] = crop_h[0] - (crop_h[1]-inp_h)
        crop_h[1] = inp_h
    if crop_w[0] < 0:
        crop_w[1] = crop_w[1] - crop_w[0]
        crop_w[0] = 0
    if crop_w[1] > 1024:
        crop_w[0] = crop_w[0] - (crop_w[1] - inp_w)
        crop_w[1] = inp_w

    crop_h = tuple(crop_h)
    crop_w = tuple(crop_w)

    return crop_h,crop_w