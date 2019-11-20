%% =========================================================================
% DL
% 2018.4.4
% Daitu
% 绘制W
%% =========================================================================
clear;
close all; 
clc;
%% Load training and testing data
DataPath   = 'UMIST_20classes_1012points_32_32';
load(DataPath);
% add noise
A = A + sqrt(200)*randn(size(A));
% splist data
A = normcol_equal(A);
[TrData,TtData,TrLabel,TtLabel]=ExtractData(A,10,labels,1709);
%% Parameter setting
DictSize = 10;
tau    = 0.00001;
alpha = 0.005;
beta  = 0.05;
lambda = 0.00005;
Iter = 20;
%% DPL trainig
% [ DictMat,P_Mat,W_Mat, EncoderMat ] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,Iter);
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,lambda,Iter);
%
[~,PredictLabel] = ClassificationRADPL( TrData, EncoderMat);
Acc = sum(TrLabel==PredictLabel)/size(TrLabel,2);
disp(['训练集最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])
%% DPL testing
tic
[Error,PredictLabel] = ClassificationRADPL( TtData(:,5), EncoderMat);
%[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Error;
TtTime = toc;
%% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])

%% plot W_Mat
Wsize = length(W_Mat) * length(W_Mat{1});
len = length(W_Mat{1});
W = zeros(Wsize,Wsize);
for ii = 1:length(W_Mat)
    W((ii-1)*len+1:ii*len,(ii-1)*len+1:ii*len) = W_Mat{ii};  
end
%
clims = [min(min(W)), max(max(W))];
figure
imshow(W,clims)
x1=20;
x2=100;
y1=20;
y2=100;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);
% colorbar
% xlabel('Training Sample Index')
% ylabel('Training Sample Index')
figure
imshow(W(x1:x2,y1:y2),clims)

imagesc(W)
colormap('gray')

%% plot S_Mat
Ssize = length(S_Mat) * length(S_Mat{1});
len = length(S_Mat{1});
S = zeros(Ssize,Ssize);
for ii = 1:length(S_Mat)
    clims = [min(min(S_Mat{ii})), max(max(S_Mat{ii}))];
    aa = (S_Mat{ii}-clims(1)) / (clims(2)-clims(1));
    S((ii-1)*len+1:ii*len,(ii-1)*len+1:ii*len) = aa;  
end
%
figure
imshow(S)
x1=30;
x2=80;
y1=30;
y2=80;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);

figure
imshow(S(x1:x2,y1:y2))


%% plot DPL S_mat
Ssize = length(CoefMat) * length(CoefMat{1});
len = length(CoefMat{1});
CoefMat2 = zeros(Ssize,Ssize);
for ii = 1:length(CoefMat)
    clims = [min(min(CoefMat{ii})), max(max(CoefMat{ii}))];
    aa = (CoefMat{ii}-clims(1)) / (clims(2)-clims(1));
    CoefMat2((ii-1)*len+1:ii*len,(ii-1)*len+1:ii*len) = aa;  
end
%
figure
imshow(CoefMat2)
x1=30;
x2=80;
y1=30;
y2=80;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);

figure
imshow(CoefMat2(x1:x2,y1:y2))


% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================


%% Load training and testing data
DataPath   = 'Random_face_features_AR';
load(DataPath);
% splist data
A = normcol_equal(A);
[TrData,TtData,TrLabel,TtLabel]=ExtractData(A,10,labels,1234);
%% Parameter setting 0.0001       0.001     0.005
DictSize = 10;
tau    = 0.00001;
alpha = 0.0001;
beta  = 0.001;
lambda = 0.005;
Iter = 20;
%% DPL trainig
% [ DictMat,P_Mat,W_Mat, EncoderMat ] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,Iter);
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,lambda,Iter);
%
[~,PredictLabel] = ClassificationRADPL( TrData, EncoderMat);
Acc = sum(TrLabel==PredictLabel)/size(TrLabel,2);
disp(['训练集最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])
%% DPL testing
tic
[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Error;
TtTime = toc;
%% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])

%% plot W_Mat
Wsize = length(W_Mat) * length(W_Mat{1});
len = length(W_Mat{1});
W = zeros(Wsize,Wsize);
for ii = 1:length(W_Mat)
    W((ii-1)*len+1:ii*len,(ii-1)*len+1:ii*len) = W_Mat{ii};  
end
%
clims = [min(min(W)), max(max(W))];
figure
imshow(W,clims)
x1=100;
x2=300;
y1=100;
y2=300;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);
figure
imshow(W(x1:x2,y1:y2))

imagesc(W)
colormap('gray')

%% 绘制稀疏编码
%% plot S_Mat
Ssize = length(S_Mat) * length(S_Mat{1});
len = length(S_Mat{1});
S = zeros(Ssize,Ssize);
for ii = 1:length(S_Mat)
    clims = [min(min(S_Mat{ii})), max(max(S_Mat{ii}))];
    S_Mat{ii} = (S_Mat{ii}-clims(1)) / (clims(2)-clims(1));
    S((ii-1)*len+1:ii*len,(ii-1)*len+1:ii*len) = S_Mat{ii};  
end

%% 
figure
imshow(S)
x1=30;
x2=80;
y1=30;
y2=80;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);


figure
imshow(S(x1:x2,y1:y2))








% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================

%% Load training and testing data
DataPath   = 'UMIST_20classes_1012points_32_32';
load(DataPath);
% splist data
% A = A + sqrt(100)*randn(size(A));
A = normcol_equal(A);
[TrData,TtData,TrLabel,TtLabel]=ExtractData(A,10,labels,17);
%% Parameter setting
DictSize = 10;
tau    = 0.00001;
alpha = 0.005;
beta  = 0.05;
lambda = 0.00005;
Iter = 20;
%% DPL trainig
% [ DictMat,P_Mat,W_Mat, EncoderMat ] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,Iter);
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,lambda,Iter);
%
[~,PredictLabel] = ClassificationRADPL( TrData, EncoderMat);
Acc = sum(TrLabel==PredictLabel)/size(TrLabel,2);
disp(['训练集最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])
[PY_Mat,DPY_Mat] = DlPlYl(TrData, TrLabel, DictMat,P_Mat);
%% DPL testing 
% 每类选择10张图像进行测试
[TtData,~,TtLabel,~]=ExtractData(TtData,10,TtLabel,17);
[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Error;
TtTime = toc;
% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])

[PY_Mat,DPY_Mat] = DlPlYl(TtData, TtLabel, DictMat,P_Mat);
% tabulate(TtLabel)
%% Plot PX
% figure
% imshow(PY_Mat,[min(min(PY_Mat)),max(max(PY_Mat))])
figure
imshow(PY_Mat)

x1=80;
x2=130;
y1=80;
y2=130;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);


figure
imshow(PY_Mat(x1:x2,y1:y2))
%% 
figure
imshow(DPY_Mat)


%% mean PSNR
indexim = [18,24,37,73,83,129,172,156,198];
orimdata = TtData(:,indexim);
dpyimdata = DPY_Mat(:,indexim);
impsnr = 1:9;
meandist = 1:9;
for ii = 1:9
    imor = reshape(orimdata(:,ii),32,32);
    low = min(min(imor));
    high = max(max(imor));
    imor = (imor - low) /(high-low) ;
    imdpy = reshape(dpyimdata(:,ii),32,32);
    low = min(min(imdpy));
    high = max(max(imdpy));
    imdpy = (imdpy - low) /(high-low) ;
    impsnr(ii) = psnr(imdpy,imor);
    meandist(ii) = norm((imor(:) - imdpy(:)));
end
mean(impsnr)
mean(meandist)
%% 
dist(immatTr',immatDPY');

norm(imor - imdpy)

%% Plot DPL PX
% figure
[PY_Mat,DPY_Mat] = DlPlYl(TtData, TtLabel, DictMat,P_Mat);
figure
imshow(PY_Mat)

x1=80;
x2=130;
y1=80;
y2=130;
hold on 
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, 'r-', 'LineWidth', 1);


figure
imshow(PY_Mat(x1:x2,y1:y2))
%% 
figure
imshow(DPY_Mat)

%% plot PXW

P_lX_l = TrData(:,TrLabel == 2) * W_Mat{2};


