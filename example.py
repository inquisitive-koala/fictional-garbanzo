from scipy import misc
import imgTools as it
import hough

print("Reading image...")
img_dir = "/home/endersgate/Projects/garbanzo/fictional-garbanzo/images/"
sample_text = misc.imread(img_dir + "sample_text.bmp")

print("Horizonal edge detection...")
htext = it.filterImg(sample_text, it.horizKernel)
print("Output htext.bmp")
misc.imsave(img_dir + "htext.bmp", htext)

print("Hough Transform...")
hough_htext = hough.houghTransImg(htext)
print("Output hough_htext.bmp")
misc.imsave(img_dir + "hough_htext.bmp", hough_htext)

print("Reverse Hough Transform...")
unhough_htext = hough.unHoughTransImg(hough_htext, htext.shape)
print("Output unhough_htext.bmp")
misc.imsave(img_dir + "unhough_htext.bmp", unhough_htext)

print("Invert Black/White...")
invUH_htext = it.invGreyScaleImg(unhough_htext)
print("Output invUH_htext.bmp")
misc.imsave(img_dir + "invUH_htext.bmp", invUH_htext)

print("Color red...")
redUH_htext = it.toRedBmp(invUH_htext)
print("Output redUH_htext.bmp")
misc.imsave(img_dir + "redUH_htext.bmp", redUH_htext)

print("Overlay on original image...")
composite = it.overlayImg(sample_text, redUH_htext)
print("Output composite.bmp")
misc.imsave(img_dir + "composite.bmp", composite)
