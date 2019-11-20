function [ DataMat, DictMat, P_Mat, DataInvMat,W_Mat,S_Mat] = ...
    Initilization( Data , Label, DictSize, tau)
%% In this intilization function, we do the following things:
%-------------------------------------------------------------------------
% input:
% Data:The original data set,Each column as a sample
% Label: The original data set label,Each column as a label
% DictSize: each class Dictionary(D) autom number 
% tau: Prevent matrix singular 

%======================================================================
% output:
% DataMat: The original data array, each classify is an array matrix
% DictMat: Dictionary array D, each classify is an array matrix
% P_Mat:  Dictionary array P, each classify is an array matrix
% DataInvMat :  (~X_i)*(~X_i)^T
% W_Mat : Adaptive Representations W
%% program
ClassNum = length(unique(Label)); % class number
Dim      = size(Data,1);  % feature number
%% matrix to cell
DataMat = cell(1,ClassNum);
DictMat = cell(1,ClassNum);
P_Mat = cell(1,ClassNum);
S_Mat = cell(1,ClassNum);
DataInvMat = cell(1,ClassNum);
W_Mat = cell(1,ClassNum);
% initolization D and P,updata W
for i=1:ClassNum
    TempData      = Data(:,Label==i);  % classify i Data
    DataMat{i}    = TempData;   % cell i element
    rng(i,'twister');      % randm seed                  
    % initolization Dictionary D_i
    DictMat{i}    = normcol_equal(randn(Dim, DictSize)); 
    rng(2*i,'twister');
    % initolization  Dictionary P_i
    P_Mat{i}      = normcol_equal(randn(Dim, DictSize)');
    % initolization  Dictionary S_i
    S_Mat{i}      = P_Mat{i}*TempData;
    % Calculate the not class i  data
    TempDataC     = Data(:,Label~=i); % not i classifi Data
    % (~X_l)*(~X_l)^T
    DataInvMat{i} = TempDataC * (TempDataC');
    % initolization W_i
    Dim_W = size(TempData,2);
    rng(2*i,'twister');
    temp_W  = normcol_equal(randn(Dim_W));
    W_Mat{i} = temp_W;
end
% umdate W
W_Mat = UpdateW( DataMat, P_Mat,  W_Mat,tau);

