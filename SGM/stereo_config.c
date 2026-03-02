void test_stereo_base(__IOM uint32_t *address) {
    // config_stereo_image_width: [10:0], config_stereo_image_height: [21:11]
    uint32_t *config_stereo_res = (__IOM uint32_t *)(0x260UL/sizeof(uint32_t) + address);
    // config_stereo_image_width_new: [10:0], config_stereo_image_height_new: [21:11]
    uint32_t *config_stereo_ds_res = (__IOM uint32_t *)(0x264UL/sizeof(uint32_t) + address);
    // config_stereo_range: [8:0], config_stereo_p1: [18:12], config_stereo_p2: [28:22]
    uint32_t *config_stereo_range_p1_p2 = (__IOM uint32_t *)(0x268UL/sizeof(uint32_t) + address);
    // config_stereo_lrc_param: [27:10], config_sel_col: [0], config_stereo_median_sel: [2:1]
    // config_stereo_post_sel: [3], config_stereo_downsampling_sel: [4], config_stereo_depth_format: [5]
    uint32_t *config_stereo_postprocess = (__IOM uint32_t *)(0x26CUL/sizeof(uint32_t) + address);
    uint32_t *config_stereo_depth_param = (__IOM uint32_t *)(0x270UL/sizeof(uint32_t) + address);
    // config_stereo_crop_size_top: [1:0], config_stereo_crop_size_left: [3:2]
    uint32_t *config_stereo_crop_param = (__IOM uint32_t *)(0x274UL/sizeof(uint32_t) + address);
    // config_stereo_clip_threshold: [15:0], config_stereo_disp_threshold: [31:16]
    uint32_t *config_stereo_hole_filling = (__IOM uint32_t *)(0x278UL/sizeof(uint32_t) + address);
    
    write_param(config_stereo_res, 0x0021C780);
    write_param(config_stereo_ds_res, 0x0021C780);
    write_param(config_stereo_range_p1_p2, 0x02002080);
    write_param(config_stereo_postprocess, 0x00A00000);
    write_param(config_stereo_depth_param, 0x00000001);
    write_param(config_stereo_crop_param, 0x00000000);
    write_param(config_stereo_hole_filling, 0xFFFF0A00);
}
