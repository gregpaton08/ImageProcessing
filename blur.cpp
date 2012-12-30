// Greg Paton
// 26 Dec 2012
// test.cpp
// testing loading and saving
// of bitmap images

#include <iostream>
#include <string>
#include <cstdlib>

#include "bitmap_image.hpp"

#define round(x) (int)(x+0.5)

typedef struct cuda_info {
    int tb_x;
    int tb_y;
    int gr_x;
    int gr_y;
} ci;



// blur() - function for blurring an image
//
// arguments:
//      o_data   - original pixel data
//      n_data   - resized image
//      width    - width of image
//      height   - height of image
//      w        - width of window
//      h        - height of window
//      bpp      - bytes per pixel
//      info     - struct containing block and thread dimensions
__global__
void blur(unsigned char *old_data, unsigned char *new_data, int width, int height, int w, int h, int bpp, ci info) {
    // error checking
    if (old_data == NULL || new_data == NULL || width <= 0 || height <= 0 || w <= 0 || h <= 0 || bpp <= 0)
        return;
    
    // determine blur factor
    const double bf = 1.0 / (double)(w * h);
    
    int _i, _j, curr;
    double r = 0;
    double g = 0;
    double b = 0;
    
    // distribute work among threads by width
    // distribute work among blocks by height
    const int num_threads = info.tb_x * info.tb_y;
    const int num_blocks  = info.gr_x * info.gr_y;
    const int w_work_width = w / num_threads;
    const int h_work_width = h / num_blocks;
    const int tid = threadIdx.x + (threadIdx.y * info.tb_x);
    const int bid = blockIdx.x + (blockIdx.y * info.gr_x);
    int w_start = tid * w_work_width;
    int w_stop = w_start + w_work_width;
    int h_start = bid * h_work_width;
    int h_stop = h_start + h_work_width;
    
    // make sure data dimensions are never exceeded
    if (w_stop > w)
        w_stop = w;
    if (h_stop > h)
        h_stop = h;
    
    // if w is not evenly divisible by num_threads
    // give remainder of work to last thread
    if (tid == num_threads - 1)
        w_stop = w;
    if (bid == num_blocks - 1)
        h_stop = h;
    
    for (int i = 0; i < height; ++i) {
        _i = i * width * bpp;
        for (int j = 0; j < width; ++j) {
            _j = j * bpp;
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    curr = _i - (h/2) + k + _j - (w/2) + l;
                    
                    // skip edge cases for now
                    if (i - h < 0 || i + h > height)
                        continue;
                    else if (j - w < 0 || j + w > width)
                        continue;
                    
                    // write to new image
//                    new_data[_i + _j + 2] += old_data[curr + 2] * bf;
//                    new_data[_i + _j + 1] += old_data[curr + 1] * bf;
//                    new_data[_i + _j + 0] += old_data[curr + 0] * bf;
//                    r += old_data[curr + 2] * bf;
//                    g += old_data[curr + 1] * bf;
//                    b += old_data[curr + 0] * bf;
//                    new_data[_i + _j + 2] = old_data[(i * width * bpp) + (j * bpp) + 2];
//                    new_data[_i + _j + 1] = old_data[(i * width * bpp) + (j * bpp) + 1];
//                    new_data[_i + _j + 0] = old_data[(i * width * bpp) + (j * bpp) + 0];
//                    r += old_data[(i * width * bpp) + (j * bpp) + 2] * bf;
//                    g += old_data[(i * width * bpp) + (j * bpp) + 1] * bf;
//                    b += old_data[(i * width * bpp) + (j * bpp) + 0] * bf;
//                    r += old_data[_i + _j + 2] * bf;
//                    g += old_data[_i + _j + 1] * bf;
//                    b += old_data[_i + _j + 0] * bf;
                    r += old_data[((i - (h/2) + k) * width * bpp) + ((j - (w/2) - l) * bpp) + 2] * bf;
                    g += old_data[((i - (h/2) + k) * width * bpp) + ((j - (w/2) - l) * bpp) + 1] * bf;
                    b += old_data[((i - (h/2) + k) * width * bpp) + ((j - (w/2) - l) * bpp) + 0] * bf;

                }
            }
            new_data[_i + _j + 2] += round(r);
            new_data[_i + _j + 1] += round(g);
            new_data[_i + _j + 0] += round(b);
            r = 0;
            g = 0;
            b = 0;
        }
    }
    new_data[0] = round(bf*100.0);
}



int main (int argc, char **argv)
{  

    if (argc != 8) {
        printf("usage: %s FILE_NAME WINDOW_WIDTH WINDOW_HEIGHT THREAD_BLOCK_WIDTH THREAD_BLOCK_HEIGHT GRID_WIDTH GRID_HEIGHT\n", argv[0]);
        return -1;
    }
    
    ci info;
    info.tb_x = atoi(argv[4]);
    info.tb_y = atoi(argv[5]);
    info.gr_x = atoi(argv[6]);
    info.gr_y = atoi(argv[7]);
    if (info.tb_x <= 0 || info.tb_y <= 0 || info.gr_x <= 0 || info.gr_y <= 0) {
        printf("Invalid grid or block dimensions!\n");
        return -1;
    }
    
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    if (width <= 0 || height <= 0) {
        printf("Invalid window dimensions!\n");
        return -1;
    }
    
    float time;
    cudaEvent_t start, stop;
    
    unsigned char *o_data;
    unsigned char *n_data;
    
    std::string file_name = argv[1];
    bitmap_image o_img(file_name);
    bitmap_image n_img(o_img.width(), o_img.height());

    // start timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // allocate device memory for original image and copy data
    cudaMalloc((void**)&o_data, o_img.width() * o_img.height() * o_img.bytes_per_pixel() * sizeof(unsigned char));
    cudaMemcpy(o_data, o_img.data(), o_img.width() * o_img.height() * o_img.bytes_per_pixel(), cudaMemcpyHostToDevice);
    
    // allocate device memory for new image and zero it
    cudaMalloc((void**)&n_data, n_img.width() * n_img.height() * n_img.bytes_per_pixel() * sizeof(unsigned char));
    cudaMemset(n_data, 0, n_img.width() * n_img.height() * n_img.bytes_per_pixel() * sizeof(unsigned char));

    // set up grid and blocks
    dim3 dimBlock(info.tb_x, info.tb_y);
    dim3 dimGrid(info.gr_x, info.gr_y);

    blur<<<dimGrid, dimBlock>>>(o_data, n_data, o_img.width(), o_img.height(), width, height, n_img.bytes_per_pixel(), info);
    
    cudaMemcpy(n_img.data(), n_data, n_img.width() * n_img.height() * n_img.bytes_per_pixel(), cudaMemcpyDeviceToHost);
    
    // stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("time: %fms\n", time);
    printf("%d\n", n_img.data()[0]);
    
    n_img.save_image("blur.bmp");

    cudaFree(o_data);
    cudaFree(n_data);
    
    return 0;
}
