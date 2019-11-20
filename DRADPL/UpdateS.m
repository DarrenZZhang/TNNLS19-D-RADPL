function [ S_Mat ] = UpdateS(  S_Mat, D_Mat, P_Mat, Data, tau)
%% Update S
% input:
% S_Mat : sparse code S 
% D_Mat: Dictionary D
% P_Mat: Dictionary array P, each classify is an array matrix
% Data: The original data array, each classify is an array matrix
% tau : Prevent matrix singular additions


%------------------------------------------------
% output:
% S_Mat : sparse code S 
%============================================================
%% cumpater

ClassNum = size(S_Mat,2);

for i=1:ClassNum
    Temp_P = P_Mat{i};
    Temp_D = D_Mat{i};
    Temp_S = S_Mat{i};
    Temp_Data = Data{i};
    %======================================================
    % cumputer M ; V
    % cumputer M
    Temp_M = Temp_S' - Temp_Data'*Temp_P';
    M = L21Parameter(Temp_M,tau);
    % cumputer V
    Temp_V = Temp_Data' - Temp_S'*Temp_D';
    V = L21Parameter(Temp_V,tau);
    %=======================================================
    Temp_Mol = Temp_D'*Temp_Data*V+Temp_P*Temp_Data*M;
    Temp_Den = Temp_D'*Temp_D*Temp_S*V+Temp_S*M;
    S_Mat{i} = Temp_S.*(Temp_Mol./Temp_Den);

    % ===============================================
end

