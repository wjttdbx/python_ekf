function     [Xout,Pout]=UKF2(X,P,Z,T,time)
global d_time
global mt inertia_t
global i
global VarOfMear1 VarOfProc1 VarOfMear2 VarOfProc2
global n1 alpha1 beta1 lamda1 gama1 %提前固定好，最大程度减小计算量
global n2 alpha2 beta2 lamda2 gama2
global RMea1 Q1 RMea2 Q2
Wm=zeros(2*n2+1,1);
Wc=zeros(2*n2+1,1);

Wm(:,1)=1/(2*(n2+lamda2));
Wc(:,1)=1/(2*(n2+lamda2));

Wm(1)=lamda2/(n2+lamda2);
Wc(1)=lamda2/(n2+lamda2)+1-alpha2^2+beta2;

%%
%第一次采样
XTemp1=zeros(length(X),n2);
XTemp2=zeros(length(X),n2);
P=P+(1e-18)*eye(n2);
choP=gama2*(chol(P))';
for j=1:n2
    XTemp1(:,j)=X-choP(:,j);
    XTemp2(:,j)=X+choP(:,j);
end
XSam=[X XTemp1 XTemp2];
%%
%状态变换
XPred=zeros(size(XSam));
for j=1:(2*n2+1)
    XPred(:,j)=xStateFunction2(XSam(:,j),T);
end    
%%
%求取状态更新后的平均值及状态量
XPredAver=XPred*Wm;
PXX=Q2;%%%%Q还未给出
for j=1:(2*n2+1)
    PXX=PXX+Wc(j)*[(XPred(:,j)-XPredAver)*(XPred(:,j)-XPredAver)'];
end
%%
%进一步采样
choQ=gama2*(chol(Q2+1e-18*eye(n2)))';
XTemp3=zeros(n2,n2);
XTemp4=zeros(n2,n2);
for j=1:n2
    XTemp3(:,j)=XPred(:,1)-choQ(:,j);
    XTemp4(:,j)=XPred(:,1)+choQ(:,j);
end
XSamSEC=[XPred,XTemp3,XTemp4];
%%
%输入观测方程
ZPred=zeros(n2,4*n2+1);
for j=1:4*n2+1
    ZPred(:,j)=obs(XSamSEC(:,j));
end
%%
%求平均值及相关矩阵
Wm2=zeros(4*n2+1,1);
Wc2=zeros(4*n2+1,1);

Wm2(:,1)=1/(2*(2*n2+lamda2));
Wc2(:,1)=1/(2*(2*n2+lamda2));

Wm2(1,1)=lamda2/(2*n2+lamda2);
Wc2(1,1)=lamda2/(2*n2+lamda2)+1-alpha2^2+beta2;

ZPredAve=ZPred*Wm2;

PXZ=zeros(n2,n2);
PZZ=RMea2;
for j=1:(4*n2+1)
    PXZ=PXZ+Wc2(j)*[(XSamSEC(:,j)-XPredAver)*(ZPred(:,j)-ZPredAve)'];%
    PZZ=PZZ+Wc2(j)*[(ZPred(:,j)-ZPredAve)*(ZPred(:,j)-ZPredAve)'];
end
K=PXZ/PZZ;
Xout=XPredAver+K*(Z-ZPredAve);
%Xout(1:4)=(quatnormalize((Xout(1:4))'))';
Pout=PXX-K*PZZ*K';