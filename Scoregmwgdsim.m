function score = Scoregmwgdsim(img1, img2)
% Implementation of "Image Quality Assessment for Medical Images 
% Based on Gradient Information" - Waqar Mirza
%  2018 5th International Conference on Business and Industrial Research (ICBIR)
% Date of Conference: 17-18 May 2018
% Date Added to IEEE Xplore: 21 June 2018
% Input: img1 is reference and img2 is the processed image, 
% Output: score is the similarity score between the two images
% Convert the input images to double precision
img1 = double(img1);
img2 = double(img2);
% Define the Sobel filters for horizontal and vertical directions
sobel_h = [-1 0 1; -2 0 2; -1 0 1];
sobel_v = [-1 -2 -1; 0 0 0; 1 2 1];
% Convolve img1 and img2 with Sobel filters
grad_h_img1 = conv2(img1, sobel_h, 'same');
grad_v_img1 = conv2(img1, sobel_v, 'same');
grad_h_img2 = conv2(img2, sobel_h, 'same');
grad_v_img2 = conv2(img2, sobel_v, 'same');
% Compute the gradient direction and magnitude of img1 and img2
grad_direction_img1 = atan2(grad_v_img1, grad_h_img1) * 180 / pi;
grad_direction_img2 = atan2(grad_v_img2, grad_h_img2) * 180 / pi;
grad_mag_img1 = sqrt(grad_h_img1.^2 + grad_v_img1.^2);
grad_mag_img2 = sqrt(grad_h_img2.^2 + grad_v_img2.^2);
% Compute the cosine similarity between grad_direction_img1 and grad_direction_img2
diff_angle = grad_direction_img1 - grad_direction_img2;
diff_angle_rad = diff_angle * pi / 180;
cos_sim = mean(cos(diff_angle_rad));
% Normalize the values of cosine similarity in every pixel between 0 and 1
cos_sim_norm = (cos_sim - min(cos_sim(:))) ./ (max(cos_sim(:)) - min(cos_sim(:)));
% Combine the gradient maps of img1 and img2
avg_grad = (grad_mag_img1 + grad_mag_img2) / 2;
% Apply threshold to the combined gradient map
% to make a new combined gradient map
for i = 1:size(avg_grad, 1)
    for j = 1:size(avg_grad, 2)
        if avg_grad(i,j) <= 20
            avg_grad(i,j) = 0;
        else
            avg_grad(i,j) = avg_grad(i,j) - 20;
        end
    end
end
% Create a 2-D Gaussian smoothing kernel
sigma = 2;
kernel_size = ceil(6*sigma);
[X,Y] = meshgrid(-kernel_size:kernel_size);
kernel = exp(-(X.^2 + Y.^2) / (2*sigma^2));
kernel = kernel / sum(kernel(:));
% Smooth the thresholded gradient map using the Gaussian kernel
smoothed_grad = imfilter(avg_grad, kernel);
% Compute the product and ratio of smoothed_grad and cos_sim_norm
grad_cos_product = smoothed_grad .* cos_sim_norm;
grad_cos_ratio = smoothed_grad;
grad_cos_ratio(grad_cos_ratio == 0) = 1;
grad_cos_ratio = grad_cos_product ./ grad_cos_ratio;
% Compute the mean of grad_cos_ratio as the final score
score = mean(grad_cos_ratio(:));
end