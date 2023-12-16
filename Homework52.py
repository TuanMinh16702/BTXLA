import cv2
import numpy as np
import streamlit as st


#Input image
ori_image = np.fromfile('img/girl2.bin', dtype=np.uint8).reshape(256, 256)
noise32hi_image = np.fromfile('img/girl2Noise32Hi.bin', dtype=np.uint8).reshape(256, 256)
noise32_image = np.fromfile('img/girl2Noise32.bin', dtype=np.uint8).reshape(256, 256)

# 2.A
def calculate_mse(image1, image2):
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    error = float("{: .4f}".format(error))
    return error

# Tính MSE
mseOri_ori = calculate_mse(ori_image, ori_image)
mseOri_Noise32 = calculate_mse(ori_image, noise32_image)
mseOri_NoiseHiPass = calculate_mse(ori_image, noise32hi_image)

# show result by streamlit


tab1, tab2, tab3, tab4 = st.tabs(["2.A", "2.B", "2.C", "2.D"])

with tab1:
    st.subheader("Display and calculate the MSE between the noisy image and the original image")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.image(ori_image, use_column_width=True, channels="L", caption="Original Image")
        st.write("MSE: ", mseOri_ori)
    with a2:
        st.image(noise32hi_image, use_column_width=True, channels="L", caption="High Pass White Gaussian Noise")
        st.write("MSE: ", mseOri_NoiseHiPass)

    with a3:
        st.image(noise32_image, use_column_width=True, channels="L", caption=" Broadband White Gaussian Noise")
        st.write("MSE: ", mseOri_Noise32)

# 2.B.

#Tạo ma trận DFT 256x256
[U, V] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
U_cutoff = 64
HLtildeCenter = np.double(np.sqrt(U**2 + V**2) <= U_cutoff)
HLtilde = np.fft.fftshift(HLtildeCenter)


#Tính toán DFT của từng ảnh và áp dụng bộ lọc
def DFT_Filter(img_name):
    dft = np.fft.fft2(img_name)
    filtered_img = np.fft.ifft2(dft * HLtilde).real
    cv2.imwrite('img/tmp.jpg', filtered_img)
    return filtered_img


# Tính toán MSE
def mse_filter(filter_img):
    mse_filter = np.mean((DFT_Filter(filter_img) - filter_img)**2)
    mse_filter = float("{: .4f}".format(mse_filter))
    return mse_filter

with tab2:
    st.subheader("Apply an isotropic ideal low-pass filter and calculate the MSE")
    b1, b2, b3 = st.columns(3)
    with b1:
        DFT_Filter(ori_image)
        st.image("img/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2")
        st.write('MSE Of Filtered Image', mse_filter(ori_image))
    with b2:
        DFT_Filter(noise32_image)
        st.image("img/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2Noise32")
        st.write('MSE Of Filtered Image', mse_filter(noise32hi_image))

    with b3:
        DFT_Filter(noise32hi_image)
        st.image("img/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2Noise32Hi")
        st.write('MSE Of Filtered Image', mse_filter(noise32_image))


# 2.C
# U_cutoff_H = 64
def lowPassFilter(U_cutoff_H):
    SigmaH = 0.19 * 256 / U_cutoff_H
    # Tạo lưới cho các giá trị tần số không gian
    [U, V] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))

    # Tính toán HtildeCenter
    HtildeCenter = np.exp((-2 * np.pi**2 * SigmaH**2) / (256**2) * (U**2 + V**2))

    # Sử dụng fftshift để "giữa" mảng DFT
    Htilde = np.fft.fftshift(HtildeCenter)

    # Áp dụng fftshift lại để "trung tâm" hồi đáp dimpulse
    H = np.fft.ifft2(Htilde)
    H2 = np.fft.fftshift(H)

    # Tạo một ma trận zero với kích thước 512x512
    ZPH2 = np.zeros((512, 512))
    ZPH2[:256, :256] = H2
    #DFT of ZPH2
    DFT_ZPH2 = np.fft.fft2(ZPH2)
    return DFT_ZPH2


def fft_FilteredDFT(input_img, cutoff):
    img_padded = np.pad(input_img, ((0, 256), (0, 256)), mode='constant', constant_values=0)
    # dft of image padded
    DFT_Img = np.fft.fft2(img_padded)
    filter_dft = DFT_Img * lowPassFilter(cutoff)
    filtered_Img = np.fft.ifft2(filter_dft)
    # 512 X 512 => 256 X 256
    result = filtered_Img[128:384, 128:384]
    return result
def mse_filter1(img, filter_img):
    mse_filter = np.sum((img.astype("float") - filter_img.astype("float")) ** 2)
    mse_filter /= float(img.shape[0] * filter_img.shape[1])
    mse_filter = float("{: .4f}".format(mse_filter))
    return mse_filter
def calculate_isnr(image_noisy, cutoff):
    mse_noisy = calculate_mse(image_noisy, ori_image)  # Thay `ori_image` bằng ảnh gốc
    mse_filtered = calculate_mse(fft_FilteredDFT(image_noisy,cutoff), ori_image)  # Thay `ori_image` bằng ảnh gốc

    isnr = 10 * np.log10(mse_noisy / mse_filtered)
    isnr = float("{: .4f}".format(isnr))

    return isnr

#Display result with cut off 64
with tab3:
    st.subheader("Apply the Gaussian low-pass filter and calculate the MSE, ISNR")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(fft_FilteredDFT(ori_image,64), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Original Image")
        st.write("MSE: ", mse_filter1(ori_image, fft_FilteredDFT(ori_image,64)))

    with c3:
        st.image(fft_FilteredDFT(noise32_image,64), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Noise32")
        st.write("MSE: ", mse_filter1(noise32_image, fft_FilteredDFT(noise32_image,64)))
        st.write("ISNR: ", calculate_isnr(noise32_image,64))

    with c2:
        st.image(fft_FilteredDFT(noise32hi_image,64), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Noise32 High Pass")
        st.write("MSE: ", mse_filter1(noise32hi_image, fft_FilteredDFT(noise32hi_image,64)))
        st.write("ISNR: ", calculate_isnr(noise32hi_image,64))

#2.D

#Display result with cutoff = 77.5
with tab4:
    st.subheader("Apply the Gaussian low-pass filter and calculate the MSE, ISNR")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.image(fft_FilteredDFT(ori_image,77.5), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Original Image")
        st.write("MSE: ", mse_filter1(ori_image, fft_FilteredDFT(ori_image,77.5)))

    with d3:
        st.image(fft_FilteredDFT(noise32_image,77.5), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Noise32")
        st.write("MSE: ", mse_filter1(noise32_image, fft_FilteredDFT(noise32_image,77.5)))
        st.write("ISNR: ", calculate_isnr(noise32_image,77.5))

    with d2:
        st.image(fft_FilteredDFT(noise32hi_image,77.5), use_column_width=True, channels="L", clamp=True,
                 caption="Filtered Noise32 High Pass")
        st.write("MSE: ", mse_filter1(noise32hi_image, fft_FilteredDFT(noise32hi_image,77.5)))
        st.write("ISNR: ", calculate_isnr(noise32hi_image,77.5))
