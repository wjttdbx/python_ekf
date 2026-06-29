function     [Xout,Pout]=UKF(X,P,Z,T,time)
global d_time
global mt inertia_t
global i
global VarOfMear VarOfProc
global n alpha beta lamda gama
global RMea Q
Wm=zeros(2*n+1,1);
Wc=zeros(2*n+1,1);

Wm(:,1)=1/(2*(n+lamda));
Wc(:,1)=1/(2*(n+lamda));

Wm(1)=lamda/(n+lamda);
Wc(1)=lamda/(n+lamda)+1-alpha^2+beta;

%%
%第一次采样
XTemp1=zeros(length(X),n);
XTemp2=zeros(length(X),n);
P=P+(1e-18)*eye(7);
choP=gama*(chol(P))';
for j=1:n
    XTemp1(:,j)=X-choP(:,j);
    XTemp2(:,j)=X+choP(:,j);
end
XSam=[X XTemp1 XTemp2];
%%
%状态变换
XPred=zeros(size(XSam));
for j=1:(2*n+1)
    XPred(:,j)=xStateFunction(XSam(:,j),T);
end    
%%
%求取状态更新后的平均值及状态量
XPredAver=XPred*Wm;
PXX=Q;
for j=1:(2*n+1)
    PXX=PXX+Wc(j)*[(XPred(:,j)-XPredAver)*(XPred(:,j)-XPredAver)'];
end
%%
%进一步采样
choQ=gama*(chol(Q+1e-18*eye(7)))';
for j=1:n
    XTemp3(:,j)=XPred(:,1)-choQ(:,j);
    XTemp4(:,j)=XPred(:,1)+choQ(:,j);
end
XSamSEC=[XPred,XTemp3,XTemp4];
%%
%输入观测方程
ZPred=zeros(7,4*n+1);
for j=1:4*n+1
    ZPred(:,j)=obs(XSamSEC(:,j));
end
%%
%求平均值及相关矩阵
Wm2=zeros(4*n+1,1);
Wc2=zeros(4*n+1,1);

Wm2(:,1)=1/(2*(2*n+lamda));
Wc2(:,1)=1/(2*(2*n+lamda));

Wm2(1,1)=lamda/(2*n+lamda);
Wc2(1,1)=lamda/(2*n+lamda)+1-alpha^2+beta;

ZPredAve=ZPred*Wm2;

PXZ=zeros(7,7);
PZZ=RMea;%%%R没有给出
for j=1:(4*n+1)
    PXZ=PXZ+Wc2(j)*[(XSamSEC(:,j)-XPredAver)*(ZPred(:,j)-ZPredAve)'];
    PZZ=PZZ+Wc2(j)*[(ZPred(:,j)-ZPredAve)*(ZPred(:,j)-ZPredAve)'];
end
K=PXZ*inv(PZZ);
Xout=XPredAver+K*(Z-ZPredAve);
Xout(1:4)=(quatnormalize((Xout(1:4))'))';
Pout=PXX-K*PZZ*K';