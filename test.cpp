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

void resize(const unsigned char *data, bitmap_image &image, int width, int height, int w, int h, int bpp);

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
    
    bitmap_image img(1024, 1024);
    resize(data, img, width, height, img.width(), img.height(), bpp);
    img.save_image("new.bmp");
    
    return 0;
}


void resize(const unsigned char *data, bitmap_image &image, int width, int height, int w, int h, int bpp) {
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
            image.set_pixel(i, j, r, g, b);
        }
    }
}