# https://www.sojson.com/hexconvert/10to16.html
def generate_write_param(config_values):
    config_stereo_res = (config_values['config_stereo_image_width'] & 0x7FF) | ((config_values['config_stereo_image_height'] & 0x7FF) << 11)
    config_stereo_ds_res = (config_values['config_stereo_image_width_new'] & 0x7FF) | ((config_values['config_stereo_image_height_new'] & 0x7FF) << 11)
    config_stereo_range_p1_p2 = (config_values['config_stereo_range'] & 0x1FF) | ((config_values['config_stereo_p1'] & 0x7f) << 12) | ((config_values['config_stereo_p2'] & 0x7f) << 22)
    config_stereo_postprocess = (config_values['config_stereo_lrc_param'] & 0x3FFFF) << 10| ((config_values['config_sel_col'] & 0x1) ) | ((config_values['config_stereo_median_sel'] & 0x3) << 1) | ((config_values['config_stereo_post_sel'] & 0x1) << 3) | ((config_values['config_stereo_downsampling_sel'] & 0x1) << 4) | ((config_values['config_stereo_depth_format'] & 0x1) << 5)
    config_stereo_depth_param = (config_values['config_stereo_depth_param'] & 0xFFFFFFFF)
    config_stereo_crop_param = (config_values['config_stereo_crop_size_top'] & 0x3) | ((config_values['config_stereo_crop_size_left'] & 0x3) << 2)
    config_stereo_hole_filling = (config_values['config_stereo_clip_threshold'] & 0xFFFF) | ((config_values['config_stereo_disp_threshold'] & 0xFFFF) << 16)
    
    print(f"write_param(config_stereo_res, 0x{config_stereo_res:08X});")
    print(f"write_param(config_stereo_ds_res, 0x{config_stereo_ds_res:08X});")
    print(f"write_param(config_stereo_range_p1_p2, 0x{config_stereo_range_p1_p2:08X});")
    print(f"write_param(config_stereo_postprocess, 0x{config_stereo_postprocess:08X});")
    print(f"write_param(config_stereo_depth_param, 0x{config_stereo_depth_param:08X});")
    print(f"write_param(config_stereo_crop_param, 0x{config_stereo_crop_param:08X});")
    print(f"write_param(config_stereo_hole_filling, 0x{config_stereo_hole_filling:08X});")


config_values = {
    'config_stereo_image_width': 1920,
    'config_stereo_image_height': 1080,
    'config_stereo_image_width_new': 1920,
    'config_stereo_image_height_new': 1080,
    'config_stereo_range': 128,
    'config_stereo_p1': 2,
    'config_stereo_p2': 8,
    'config_stereo_lrc_param': 10240,
    'config_sel_col': 0,
    'config_stereo_median_sel': 0,
    'config_stereo_post_sel': 0,
    'config_stereo_downsampling_sel': 0,
    'config_stereo_depth_format': 0,
    'config_stereo_depth_param': 1,
    'config_stereo_crop_size_top': 0,
    'config_stereo_crop_size_left': 0,
    'config_stereo_clip_threshold': 2560,
    'config_stereo_disp_threshold': 65535
}

generate_write_param(config_values)
