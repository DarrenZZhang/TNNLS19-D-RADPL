function [ P_Mat ] = UpdateP(  S_Mat, W_Mat, P_Mat, Data, DataInvMat, tau, alpha, beta,lambda)
%% Update P
% input:
% S_Mat : sparse code S
% W_Mat : adaptive weights W
% P_Mat : Dictionary P
% Data: The original data array, each classify is an array matrix
% DataInvMat :  (~X_i)*(~X_i)^T
% tau : Prevent matrix singular additions
% alpha : Robust projective parameter
% beta : discriminative adaptive representation parameter
% lambda :  parameter
%------------------------------------------------
% output:
% P_Mat : Dictionary array P, each classify is an array matrix
%============================================================
%% cumpater

ClassNum = size(S_Mat,2);
% cumputer M:训练样本的均值
K = size(P_Mat{1},1);  %dictionary size
% 整理P*X
M =W_Mat;
for i=1:ClassNum
    Temp_P = P_Mat{i};
    Temp_Data = Data{i};
    M{i}  = Temp_P * Temp_Data;
end
NL = size(Data{1},2);
index = 1:ClassNum;
for i=1:ClassNum
    Temp_W = W_Mat{i};
    Temp_P = P_Mat{i};
    Temp_S = S_Mat{i};
    Temp_Data = Data{i};
    % (~X_l)*(~X_l)^T
    Temp_Da_i = DataInvMat{i};
    %======================================================
    % Q = (I-W)(I-W)^T    
    I = eye(size(Temp_W));
    Q = (I - Temp_W)*(I -Temp_W)';
    % cumputer M ; H
    % cumputer U
    Temp_M = Temp_S' - Temp_Data'*Temp_P';
    U = L21Parameter(Temp_M,tau);
    % cumputer H
    H = L21Parameter(Temp_P',tau);
    H = ones(size(H));
    % cumputer ML
    ML = repmat(mean(Temp_P*Temp_Data),K,1);
    % cumputer ML
    MM = M(:,index~=i);
    MM = cell2mat(MM);
    MM = reshape(MM,K,NL,(ClassNum-1));
    MM = mean(MM,3);
    % ===============================================
    Temp_A = 4*Temp_Data*U*Temp_Data'+2*alpha*Temp_Da_i+4*alpha*H+...
        beta*Temp_Data*Q*Temp_Data'+beta*Temp_Data*Q'*Temp_Data'+...
        2*lambda*(NL-1)*Temp_Data*U*Temp_Data';
    Temp_A = Temp_A + tau*eye(size(Temp_A));
    P_Mat{i} = (4*Temp_S*U*Temp_Data'+2*lambda*ML*Temp_Data'-2*lambda*NL*MM*Temp_Data')/Temp_A;


end

