%%Main
%请注意参数的大小
clc
clear all
global d_time
global mt inertia_t
global i
global VarOfMear VarOfProc
%%
%parameter
d_time=0.01;
mt=10;
TotalTime=50;
inertia_t=diag([1;1;1]);
VarOfMear=0.0001*[1*[1;1.5;2];1*[1.3;3;4]];
VarOfProc=0.0001*[0;0;0;1;2;1];%%%只能修改系数
%%
%选择画图对象
%0为w，1为q
ff=1;
%%
%Initial condition
AU=eye(3);
AAA=AU;
qU=[0;0;0;1];
wU=1*[1;-0.2;-0.05];
RU=[0;0;0];
vU=[0;0;0];
wN=wU;
qN=qU;
i=0;
Zin=zeros(7,1);
%%
%滤波初始化

x0=[qU(1:3);0;0;0];%1*[0.3,-0.4,-0.2,0.8,0.2,0.3]';
xPred=x0;
Z=zeros(6,1);
P0=eye(6);%???????
PPred=P0;

H=eye(6);
R=diag(VarOfMear);
hPred=H*x0;
qPred=[xPred(1:3);sqrt(1-xPred(1:3)'*xPred(1:3))];
cons=zeros(length(0:d_time:TotalTime),4);
tempCons1=0;
tempCons2=0;
tempCons3=0;
%第4列为名义值

ObserNoise=[sqrt(VarOfMear(1))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(2))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(3))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(4))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(5))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(6))*randn(1,length(0:d_time:TotalTime))];
ProcNoise=[sqrt(VarOfProc(4))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc(5))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc(6))*randn(1,length(0:d_time:TotalTime))];
  %% 
  %用以记录观测值平均数，每十个记录一个平均数
  j=0;
  k=0;
  consZ=zeros(fix(length(0:d_time:TotalTime)/10),1);
  tempConsZ=zeros(10,1);
 
%%
%临时修改区
    xGuess=zeros(6,1);
    qGuess=zeros(6,1);
%%
for time=0:d_time:TotalTime

    i=i+1;
    F=[0;0;0];
    Toe=0.1*(0.5*sin(pi/5*time)+0.2*cos(pi/2*time)+0.1*sin(pi*time)).*[1;1;1];
   T=ProcNoise(:,i)+Toe;

%%
%循环
    R_touch_e=RU;
    [RU,AU,vU,wU,qU]=f_dyn_U2 ( RU,R_touch_e,AU,vU,wU,qU, F, T);
%    [RU,AU,vU,wN,qN]=f_dyn_U2 ( RU,R_touch_e,AU,vU,wN,qN, F, Toe);
%%
%EKF滤波区
    Z(1:3)=qU(1:3)+ObserNoise(1:3,i); 
    Z(4:6)=wU+ObserNoise(4:6,i);

%%
%%
%每一步矫正一次
if mod(i,1)==0
    K=PPred*H'/(H*PPred*H'+R);
	xDeltaGuess=K*(Z-hPred);
    xGuess(4:6)=xPred(4:6)+xDeltaGuess(4:6);%???
%    qDeltaGuess=[xDeltaGuess(1:3);sqrt(1-xDeltaGuess(1:3)'*xDeltaGuess(1:3))];
qDeltaGuessTemp=[xDeltaGuess(1:3);1];
qDeltaGuess=(quatnormalize(qDeltaGuessTemp'))';
qGuess=QuanMatrix(qDeltaGuess)*qPred;   %%%%务必注意delta与时间微分的计算方法的区别
%qGuess=qDeltaGuess+qPred;      %%%%%delta需要圈乘，不需要归一化，而微分直接相加，需归一化
xGuess(1:3)=qGuess(1:3);  
PGuess=(eye(6)-K*H)*PPred;
else
    xGuess=xPred;
    qGuess=qPred;
    PGuess=PPred;
end
%

%预测
	[phiX,FX]=RecursionFunction(xGuess);
	Q=calc_Q(xGuess);  
[xDot,q0Dot]=Deriv(xGuess,qGuess,Toe);
    xPred=xGuess+xDot*d_time;%仅限于w，即后三个元素
    qPredTemp=[xPred(1:3);qGuess(4)+q0Dot*d_time];
    qPred=(quatnormalize(qPredTemp'))';
%     qPredTemp-qPred
%     qPredTemp
%     qPred
	hPred=H*xPred;         %h的预测值直接用x的预测值
	PPred=phiX*PGuess*phiX'+Q;
    xReal=[qU(1:3);wU];
    delta=xReal-xGuess;    

    %%
    %估计结束，延时
% if i>5
%     break
% end
   qU(1:3);
   qPred;
% if mod(i,10)==0
% abs(mean(delta));
% Q*100;
% qU;
% end

if ff==0
cons(i,1)=wU(2);
cons(i,2)=xGuess(5);
cons(i,3)=Z(5);
cons(i,4)=wN(2);
else
    cons(i,1)=qU(2);
cons(i,2)=xGuess(2);
cons(i,3)=Z(2);
cons(i,4)=qN(2);
end 
%K(5)
%%记录观测平均值
j=j+1;
if ff==0
    tempConsZ(j)=Z(5);
    else
    tempConsZ(j)=Z(2);
end
if j==10
    j=0;
    k=k+1;
    consZ(k)=mean(tempConsZ);
    tempConsZ=zeros(10,1);
else
end
% %%
% %利用平均法平滑估计值
% tempCons3=cons(i,2);
% cons(i,2)=1/10*tempCons1+2/10*tempCons2+7/10*tempCons3;
% tempCons1=tempCons2;
% tempCons2=tempCons3;
% %%

end
ptime=0:10*d_time:(TotalTime);

figure
plot(0:d_time:(TotalTime),cons(:,1),0:d_time:(TotalTime),cons(:,2),'b')%,0:d_time:(TotalTime),cons(:,3),'k*'
hold on
plot(ptime(1:(length(ptime))-1)+5*d_time,consZ)
plot(0:d_time:(TotalTime),cons(:,3),'r*')
hold on
plot(0:d_time:(TotalTime),cons(:,4),'y')
%默认色为实际值，蓝色为估计值（三个的平均值或者权平均），红色为观测值，黄色为名义值,橙色为10个观测的平均值
%
%,0:d_time:(TotalTime),cons(:,3),'r*'
% plot(0:d_time:(TotalTime),cons(:,1),)