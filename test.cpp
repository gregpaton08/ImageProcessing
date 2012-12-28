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

bitmap_image *resize(const unsigned char *data, int width, int height, int w, int h, int bpp);

int main() 
{
    unsigned int width, height, bpp;
    unsigned char r, g, b;
    std::string file_name = "lena.bmp";
    bitmap_image image(file_name);
    
    // get image pixel data
    const unsigned char *data = image.data();
    
    // get image dimensions
    width = image.width();
    height = image.height();
    
    // get bytes per pixel
    bpp = image.bytes_per_pixel();
    
    bitmap_image *img = resize(data, width, height, 256, 256, bpp);
    if (img != NULL)
        img->save_image("new.bmp");
    else
        return -1;
    
    return 0;
}

// resize() - function for up/down sampling an image to an arbitrary size
//            using nearest neighbor interpolation
// return   - pointer to resize bitmap_image
// data     - original pixel data
// image    - resized image
// width    - width of original image
// height   - height of original image
// w        - width to resize to
// h        - height to resize to
// bpp      - bytes per pixel
bitmap_image *resize(const unsigned char *data, int width, int height, int w, int h, int bpp) {
    // allocate new image
    bitmap_image *img = new bitmap_image(w, h);
    
    // error checking
    if (data == NULL || img == NULL || width <= 0 || height <= 0 || w <= 0 || h <= 0 || bpp <= 0) 
        return NULL;
        
    // determine scale factors
    double sf_w = (double)width / (double)w;
    double sf_h = (double)height / (double)h;
    unsigned char r, g, b;
    int _i, _j;
    
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            // find nearest neighbor
            _i = round(i*sf_w);
            _j = round(j*sf_h);
            
            r = data[(_j * width * bpp) + (_i * bpp) + 2];
            g = data[(_j * width * bpp) + (_i * bpp) + 1];
            b = data[(_j * width * bpp) + (_i * bpp) + 0];
            
            // write to new image
            img->set_pixel(i, j, r, g, b);
        }
    }
    return img;
}