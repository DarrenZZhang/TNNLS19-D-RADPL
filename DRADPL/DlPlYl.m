function [PY_Mat,DPY_Mat] = DlPlYl(Data,Label, DictMat,P_Mat)
%UNTITLED D_l * P_l * Y_L
%   此处显示详细说明
ClassNum = length(P_Mat); % class number
%% matrix to cell
DPY = cell(1,ClassNum);
DataMat = cell(1,ClassNum);
PY = cell(ClassNum,1);
for i=1:ClassNum
    TempData      = Data(:,Label==i);  % classify i Data
    DataMat{i}    = TempData;   % cell i element
    PY{i} =  P_Mat{i} * Data ;
end
for i=1:ClassNum
    DPY{i} = DictMat{i} * P_Mat{i} * DataMat{i};
end
DPY_Mat = cell2mat(DPY);
PY_Mat = cell2mat(PY);
end

