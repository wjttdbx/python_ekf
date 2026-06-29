function [n]=QuanMatrix(U)
%用以实现圈乘矩阵，其中，u可以是角速度（仅有三元素），则给定第四元素定为0。
global i
n=zeros(4,4);
Temp_U=zeros(4,1);
if length(U)==3
    Temp_U(1:3)=U;
end
if length(U)==4
    Temp_U=U;
end
n=Temp_U(4).*eye(4)+[-tilde(Temp_U(1:3)) Temp_U(1:3);-Temp_U(1:3)' 0];
if mod(i,50)==0
%     n
%     Temp_U
end
