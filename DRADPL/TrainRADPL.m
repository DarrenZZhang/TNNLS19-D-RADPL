function [ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat,Ofv ] = TrainRADPL( Data, Label,...
    DictSize, tau, alpha, beta,lambda, Iter )
%% This is the RADPL training function
% Discriminative Local Sparse Representations by Robust Adaptive Dictionary Pair Le arning
%
% Input arguments:
%  Data : Train Data,every class is one of cell;Data{i} class i train data
%  Label : Train Label;example:[1,1,1,1,........,i,i,i,......,k,k]
%  DictSize : Dictionary size,The number of atoms per sub Dictionaries
%  alpha : Robust projective parameter
%  beta : discriminative adaptive representation parameter
% lambda :  parameter
%  Iter : max Iter times
%  tau : a small number,avoid the singularity issue
% output:
% DictMat : Dictionary D
% P_Mat : Dictionary P
% W_Mat : adaptive weights W
% S_Mat : sparse code S 
% EncoderMat : D*P
% Ofv : object function value
%%
% Initilize D and P , precompute the update W for one time 
[ DataMat, DictMat, P_Mat, DataInvMat, W_Mat,S_Mat ] = Initilization( Data , Label, DictSize, tau);
% Alternatively update P, D and A
Ofv = 1:Iter;
for i=1:Iter
    [ P_Mat ]   = UpdateP(  S_Mat,  W_Mat, P_Mat, DataMat,DataInvMat, tau ,alpha,beta,lambda);
    [ S_Mat ]   = UpdateS(  S_Mat,  DictMat, P_Mat, DataMat, tau );
    [ DictMat] = UpdateD(  DictMat, S_Mat,DataMat,tau);
    W_Mat = UpdateW( DataMat, P_Mat,  W_Mat,tau);
    Ofv(i) = objectfunvalue(Data, Label,DictMat,P_Mat,W_Mat,alpha,beta,lambda);
end

% Reorganize the D * P  matrix to make the classification fast
EncoderMat = cell(size(P_Mat));
for ii = 1:size(EncoderMat,2)
    EncoderMat{ii} = DictMat{ii}*P_Mat{ii};
end


    
