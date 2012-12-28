// Greg Paton
// 26 Dec 2012
// test.cpp
// testing loading and saving
// of bitmap images

#include <iostream>
#include <string>
#include <cstdlib>

#include "bitmap_image.hpp"

void upsize(const unsigned char *data, bitmap_image &image, int width, int height, int bpp);
void downsize(const unsigned char *data, bitmap_image &image, int width, int height, int bpp);

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
    
//    for (int i = 0; i < width; ++i) {
//        for (int j = 0; j < height; ++j) {
//            // access pixels using
//            // image.get_pixel(i, j, r, g, b);
//            // or 
//            r = data[(j * width * bpp) + (i * bpp) + 2];
//            g = data[(j * width * bpp) + (i * bpp) + 1];
//            b = data[(j * width * bpp) + (i * bpp) + 0];
//            //image.set_pixel(i, j, r, g, b);
//        }
//    }
//    image.save_image("saved.bmp");
    
    bitmap_image upimg(1024, 1024);
    upsize(data, upimg, width, height, bpp);
    upimg.save_image("up.bmp");
    bitmap_image downimg(256, 256);
    downsize(data, downimg, width, height, bpp);
    downimg.save_image("down.bmp");
    
    return 0;
}

// type : specifies upsize/downsize
//        1 - upsize
//        2 - downsize
bool resize(const unsigned char *data, unsigned char *res, int width, int height, int type) {
    if (type == 1) {
        
    }
    else if (type == 2) {
        
    }
    else
        return false;
    
    return true;
}

void down_size(const unsigned char *data, bitmap_image &image, int width, int height, int w, int h, int bpp) {
    double sf_w = (double)width / (double)w;
    double sf_h = (double)height / (double)h;
}

void downsize(const unsigned char *data, bitmap_image &image, int width, int height, int bpp) {
    
    unsigned char r, g, b;
    
    for (int i = 0; i < width / 2; ++ i) {
        for (int j = 0; j < height / 2; ++j) {
            r = data[(j * 2 * width * bpp) + (i * 2 * bpp) + 2];
            g = data[(j * 2 * width * bpp) + (i * 2 * bpp) + 1];
            b = data[(j * 2 * width * bpp) + (i * 2 * bpp) + 0];
            
            image.set_pixel(i, j, r, g, b);
        }
    }
}

void upsize(const unsigned char *data, bitmap_image &image, int width, int height, int bpp) {
    
    unsigned char r, g, b;
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            r = data[(j * width * bpp) + (i * bpp) + 2];
            g = data[(j * width * bpp) + (i * bpp) + 1];
            b = data[(j * width * bpp) + (i * bpp) + 0];
            
            image.set_pixel(i*2,   j*2,   r, g, b);
            image.set_pixel(i*2,   j*2+1, r, g, b);
            image.set_pixel(i*2+1, j*2,   r, g, b);
            image.set_pixel(i*2+1, j*2+1, r, g, b);
        }
    }
}
