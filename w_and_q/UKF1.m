function     [Xout,Pout]=UKF1(X,P,Z,w)
global d_time
global mt inertia_t
global i
global VarOfMear1 VarOfProc1 VarOfMear2 VarOfProc2
global n1 alpha1 beta1 lamda1 gama1 %提前固定好，最大程度减小计算量
global n2 alpha2 beta2 lamda2 gama2
global RMea1 Q1 RMea2 Q2
Wm=zeros(2*n1+1,1);
Wc=zeros(2*n1+1,1);

Wm(:,1)=1/(2*(n1+lamda1));
Wc(:,1)=1/(2*(n1+lamda1));

Wm(1)=lamda1/(n1+lamda1);
Wc(1)=lamda1/(n1+lamda1)+1-alpha1^2+beta1;

%%
%第一次采样
XTemp1=zeros(length(X),n1);
XTemp2=zeros(length(X),n1);
P=P+(1e-10)*eye(n1);
choP=gama1*(chol(P))';
for j=1:n1
    XTemp1(:,j)=X-choP(:,j);
    XTemp2(:,j)=X+choP(:,j);
end
XSam=[X XTemp1 XTemp2];
%%
%状态变换
XPred=zeros(size(XSam));
for j=1:(2*n1+1)
    XPred(:,j)=xStateFunction1(XSam(:,j),w);
end    
%%
%求取状态更新后的平均值及状态量
XPredAver=XPred*Wm;
PXX=Q1; 
for j=1:(2*n1+1)
    PXX=PXX+Wc(j)*[(XPred(:,j)-XPredAver)*(XPred(:,j)-XPredAver)'];
end
%%
%进一步采样
choQ=gama1*(chol(Q1+1e-18*eye(n1)))';
XTemp3=zeros(n1,n1);
XTemp4=zeros(n1,n1);
for j=1:n1
    XTemp3(:,j)=XPred(:,1)-choQ(:,j);
    XTemp4(:,j)=XPred(:,1)+choQ(:,j);
end
XSamSEC=[XPred,XTemp3,XTemp4];
%%
%输入观测方程
ZPred=zeros(n1,4*n1+1);
for j=1:4*n1+1
    ZPred(:,j)=obs(XSamSEC(:,j));
end
%%
%求平均值及相关矩阵
Wm2=zeros(4*n1+1,1);
Wc2=zeros(4*n1+1,1);

Wm2(:,1)=1/(2*(2*n1+lamda1));
Wc2(:,1)=1/(2*(2*n1+lamda1));

Wm2(1,1)=lamda1/(2*n1+lamda1);
Wc2(1,1)=lamda1/(2*n1+lamda1)+1-alpha1^2+beta1;

ZPredAve=ZPred*Wm2;

PXZ=zeros(n1,n1);
PZZ=RMea1;
for j=1:(4*n1+1)
    PXZ=PXZ+Wc2(j)*[(XSamSEC(:,j)-XPredAver)*(ZPred(:,j)-ZPredAve)'];
    PZZ=PZZ+Wc2(j)*[(ZPred(:,j)-ZPredAve)*(ZPred(:,j)-ZPredAve)'];
end
K=PXZ/PZZ;
Xout=XPredAver+K*(Z-ZPredAve);
%Xout(1:4)=(quatnormalize((Xout(1:4))'))';
Pout=PXX-K*PZZ*K';