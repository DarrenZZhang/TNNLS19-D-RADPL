function [ofv] = objectfunvalue(TrData, TrLabel,DictMat,P_Mat,W_Mat,alpha,beta,lambda)
%% computer objective function value
ClassNum = length(unique(TrLabel));
ofv = 1:ClassNum;
K = size(P_Mat{1},1);
% 整理P*X
M =W_Mat;
for i=1:ClassNum
    Temp_P = P_Mat{i};
    Temp_Data = TrData(:,TrLabel==i);
    M{i}  = Temp_P * Temp_Data;
end
NL = size(Temp_Data,2);
index = 1:ClassNum;
for i=1:ClassNum
    % 第一项的值
    TempData      = TrData(:,TrLabel==i);  % classify i Data
    temp_one = TempData' - TempData'*P_Mat{i}'*DictMat{i}';
    temp_one_O = L21Parameter(temp_one,0);
    one = 2*trace(temp_one'*temp_one_O*temp_one);
    % 第2项的值
    Temp_P = P_Mat{i};
    % Calculate the not class i  data
    TempDataC     = TrData(:,TrLabel~=i); % not i classifi Data
    PX_norF = norm(Temp_P*TempDataC,'fro');
    P21 = 2*trace(Temp_P* L21Parameter(Temp_P',0)*Temp_P');
    two = alpha*(PX_norF+P21);
    % 第3项的值
    Temp_W = W_Mat{i};
    three = beta*(norm((TempData-TempData*Temp_W),'fro')+...
        norm((Temp_P*TempData-Temp_P*TempData*Temp_W),'fro')+...
        norm(Temp_W,'fro'));
    % 第4项的值
    ML = repmat(mean(Temp_P*TempData),K,1);
    MM = M(:,index~=i);
    MM = cell2mat(MM);
    MM = reshape(MM,K,NL,(ClassNum-1));
    MM = mean(MM,3);
    four = lambda*(norm((M{i}-ML),'fro') - NL*norm((M{i} - MM),'fro'));
    
    
    ofv(i) = one+two+three+four;
end
ofv = mean(ofv);
