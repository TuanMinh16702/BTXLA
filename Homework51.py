import cv2
import numpy as np
import streamlit as st

# a.

input_image = np.fromfile('img/salesman.bin', dtype=np.uint8).reshape(256, 256)


kernel = np.ones((7,7), np.float32)/ 49
image_padding = np.pad(input_image, ((3,3), (3,3)), mode='constant',constant_values=0)
ROI = image_padding[4:260, 4:260]
result = cv2.filter2D(ROI, -1, kernel)
image_padding[4:260, 4:260] = result

#1.A
tab1,tab2,tab3 = st.tabs(["1.A", "1.B", "1.C"])

with tab1:
    st.subheader("")
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, use_column_width=True, channels="L", caption="Original Image")
    with col2:
        st.image(result, use_column_width=True, channels="L", caption="Filter Image")

# 1.B
#padded image
Padsize = 256 + 128 - 1
ZPX = np.zeros((Padsize, Padsize))
ZPX[:256, :256] = input_image
cv2.imwrite("img/tmp1.jpg", ZPX)


#Zero Padded Impulse Resp
H = np.zeros((128, 128))
H[62:69, 62:69] = 1 / 49

ZPH = np.zeros((Padsize, Padsize))
ZPH[:128, :128] = H

# Compute DFT's of zero-padded images
ZPXtilde = np.fft.fft2(ZPX)
ZPHtilde = np.fft.fft2(ZPH)

# Show centered log-magnitude spectra
ZPXtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPXtilde)))

ZPHtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPHtilde)))

# Compute the convolution by pointwise multiplication of DFT's
ZPYtilde = ZPXtilde * ZPHtilde
ZPY = np.fft.ifft2(ZPYtilde)
# Show the resulting zero-padded image and its centered log-magnitude spectrum
ZPYtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPYtilde)))

Y = np.real(ZPY[64:320, 64:320])
cv2.imwrite("img/tmp2.jpg", Y)
#Display result
with tab2:
    st.subheader("")
    b1,b2,b3 = st.columns(3)
    with b1:
        st.image(input_image,  use_column_width=True, channels="L", caption="Original Image")
        st.image(ZPXtildeDisplay, clamp=True, use_column_width=True, channels="L", caption="Centered DFT log-magnitude spectrum")
        st.image(ZPY,  use_column_width=True, channels="L", clamp=True, caption="Zero Padded Result")

    with b2:
        st.image("img/tmp1.jpg", use_column_width=True, clamp=True, channels="L", caption="Zero Padded Original Image")
        st.image(ZPHtildeDisplay, use_column_width=True, clamp=True,  channels="L", caption="Log-magnitude spectrum H")
        st.image('img/tmp2.jpg', use_column_width=True, channels="L", clamp= True, caption="Final Filtered Image")
    with b3:
        st.image(ZPH, use_column_width=True, channels="L", clamp=True, caption="Zero Padded Impulse Resp")
        st.image(ZPYtildeDisplay, use_column_width=True, channels="L", clamp=True, caption="Log-magnitude spectrum of result")

#1.C
def stretch(img):
    xMax = np.max(img)
    xMin = np.min(img)
    scale_factor = 255.0 / (xMax - xMin)
    y = np.round((img - xMin) * scale_factor)
    return y.astype(np.uint8)

# Make the 256x256 impulse response image
H1 = np.zeros((256, 256))
H1[126:133, 126:133] = 1/49

# Get the true zero-phase impulse response image using fftshift
H2 = np.fft.fftshift(H1)

# Zero pad the input image
ZPX = np.zeros((512, 512))
ZPX[:256, :256] = input_image

# Make the zero-padded zero-phase impulse response image
ZPH2 = np.zeros((512, 512))
ZPH2[:128, :128] = H2[:128, :128]
ZPH2[:128, 385:512] = H2[:128, 129:256]
ZPH2[385:512, :128] = H2[129:256, :128]
ZPH2[385:512, 385:512] = H2[129:256, 129:256]

# Compute the filtered result by pointwise multiplication of DFTs
Y = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
Y = stretch(Y[:256, :256])

with tab3:
    c1,c2 = st.columns(2)
    with c1:
        st.image(stretch(input_image), use_column_width=True, channels="L", clamp=True, caption="Zero Phase Impulse Resp")
        st.image(stretch(ZPH2), use_column_width=True, channels="L", clamp=True, caption="Zero Padded zero-phase H")

    with c2:
        st.image(stretch(H2), use_column_width=True, channels="L", clamp=True, caption="Zero Phase Impulse Resp")
        st.image(Y, use_column_width=True, channels="L", clamp=True, caption="Final Filtered Image")
