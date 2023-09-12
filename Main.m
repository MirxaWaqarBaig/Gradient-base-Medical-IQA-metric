clear all
clc
%%
% taken the dicom image as input
input_img = double(dicomread('Subject_1.dcm'));
dis_img = ReinhardTMO(input_img);
%%
% A higher value of the metric score is better. 
score_met = Scoregmwgdsim(input_img,dis_img);


