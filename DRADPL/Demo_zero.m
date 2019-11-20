%% =========================================================================
% DL : Discriminative Local Sparse Representations by Robust Adaptive Dictionary Pair Le arning
% Daitu
% 进行一次测试，查看函数是否正确运行
%% =========================================================================
clear;
close all; 
clc;
%% Load training and testing data
DataPath   = 'YaleB_Jiang';
load(DataPath);
% Column normalization
TrData = normcol_equal(TrData);
TtData = normcol_equal(TtData);

%Parameter setting
DictSize = 15;
tau    = 0.00001;
alpha = 0.0001;
beta  = 0.005;
lambda = 0.0001;
Iter = 20;
%% DRADPL trainig
tic
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,lambda,Iter);
TrTime = toc
%% DPL testing
tic
[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Error;
TtTime = toc
% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc)])
%% confusion matrix
confmat = confusionmat(TtLabel,PredictLabel);
% heatmap
h = heatmap(confmat);
h.Title = 'Dictionart Learning';
h.XLabel = 'Test Label';
h.YLabel = 'Predict Label';
h.ColorScaling = 'scaledcolumns';
%% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])