function [ D_Mat] = UpdateD( Dict, S_Mat, Data,tau )
%% Update Dictionary D
% input:
% Dict: Dictionary D_(t-1)
% Data: The original data array, each classify is an array matrix
% S_Mat: S 
% tau : Prevent matrix singular additions
%------------------------------------------------
% output:
% D_Mat : Dictionary D

%============================================================

%%
[ ClassNum] = size(Data,2);
D_Mat = Dict;
for i=1:ClassNum 
    Temp_S       = S_Mat{i};
    TempData       = Data{i};
    Temp_D = Dict{i};
    V = TempData' - Temp_S'*Temp_D';
    % cumputer V
    V = L21Parameter(V,tau);
    % update D
    Tempinv = Temp_S*V*Temp_S';
    Tempinv = Tempinv + tau*eye(size(Tempinv));
    D_Mat{i} = (TempData*V*Temp_S')/ Tempinv;   
%     D_Mat{i} = normcol_equal((TempData*V*Temp_S')/ Tempinv);
end




