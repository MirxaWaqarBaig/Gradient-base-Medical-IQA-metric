%%
% taken the dicom image as input
input_img = double(dicomread('Subject_1.dcm'));
%%
img=linspace(min(input_img(:)),max(input_img(:)),256);
img=uint8(arrayfun(@(x) find(abs(img(:)-x)==min(abs(img(:)-x))),input_img));
%% 
hist_eq_image = histeq(img);
%%
% Defined the Sobel filters for horizontal and vertical directions
sobel_h = [-1 0 1; -2 0 2; -1 0 1];
sobel_v = [-1 -2 -1; 0 0 0; 1 2 1];

% Convolved the input image with Sobel filters
grad_h_img = conv2(double(img), sobel_h, 'same');
grad_v_img = conv2(double(img), sobel_v, 'same');
% Computed gradient direction of input image
grad_direction_img = atan2(grad_v_img, grad_h_img) * 180 / pi;
% Computed gradient magnitude
grad_img = sqrt(grad_h_img.^2 + grad_v_img.^2);
% Convolved histogram equalized image with Sobel filters
grad_h_hist = conv2(double(hist_eq_image), sobel_h, 'same');
grad_v_hist = conv2(double(hist_eq_image), sobel_v, 'same');
% Computed gradient direction of histogram equalized image
grad_direction_hist = atan2(grad_v_hist, grad_h_hist) * 180 / pi;
% Computed gradient magnitude
grad_hist_eq_image = sqrt(grad_h_hist.^2 + grad_v_hist.^2);
% Displaying gradient images
% figure;
% subplot(2,2,1); imshow(uint8(grad_h_img)); title('Horizontal Gradient of Input Image');
% subplot(2,2,2); imshow(uint8(grad_v_img)); title('Vertical Gradient of Input Image');
% subplot(2,2,3); imshow(uint8(grad_h_hist)); title('Horizontal Gradient of Histogram Equalized Image');
% subplot(2,2,4); imshow(uint8(grad_v_hist)); title('Vertical Gradient of Histogram Equalized Image');
%%
% Computed cosine similarity between grad_direction_img and grad_direction_hist
diff_angle = grad_direction_img - grad_direction_hist;
diff_angle_rad = diff_angle * pi / 180;
cos_sim = mean(cos(diff_angle_rad));
% Normalized the values of cosine similarity in every pixel between 0 and 1
cos_sim_norm = (cos_sim - min(cos_sim(:))) ./ (max(cos_sim(:)) - min(cos_sim(:)));
%% Combined Gradient Map
avg_grad = (grad_img + grad_hist_eq_image) / 2;
%% New Combined Gradient Map
for i = 1:size(avg_grad, 1)
    for j = 1:size(avg_grad, 2)
        if avg_grad(i,j) <= 20
            avg_grad(i,j) = 0;
        else
            avg_grad(i,j) = avg_grad(i,j) - 20;
        end
    end
end
%% Convolving with 2-D Gaussian kernel
sigma = 2;
% Created a 2-D Gaussian smoothing kernel
kernel_size = ceil(6*sigma);
[X,Y] = meshgrid(-kernel_size:kernel_size);
kernel = exp(-(X.^2 + Y.^2) / (2*sigma^2));
kernel = kernel / sum(kernel(:));
% Convolved avg_grad with the Gaussian kernel using the imfilter function
smoothed_grad = imfilter(avg_grad, kernel);
%%
% Computed the product of smoothed_grad and cos_sim_norm
grad_cos_product = smoothed_grad .* cos_sim_norm;

% Divided by smoothed_grad, making sure to avoid division by zero
grad_cos_ratio = smoothed_grad;
grad_cos_ratio(grad_cos_ratio == 0) = 1; % set zero values to 1
grad_cos_ratio = grad_cos_product ./ grad_cos_ratio;

% Computed the mean of grad_cos_ratio as the final score
score = mean(grad_cos_ratio(:));