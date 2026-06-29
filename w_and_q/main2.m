%%
%实验证明，对w的滤波很失败，原因未知；
%q很完美，基本上没有误差，甚至不需要观测。很奇怪。有可能是交叉污染？？
%另外，归一化也可以看做一个观测条件。
%按理说，w自身的P与q有很强的相关性。这个可以稍微分析。
%%
clc
clear all
global d_time
global mt inertia_t
global i
global VarOfMear VarOfProc
global n alpha beta lamda gama %提前固定好，最大程度减小计算量
global RMea Q
%%
%parameter
figureNumber=6;
d_time=0.05;
mt=10;
TotalTime=10;
inertia_t=diag([1;1;1]);

VarOfMear1=1e-1*[1;1.5;2;1];
VarOfMear2=1e-2*[1.3;3;4];
VarOfProc1=0*[1;1;1;1];%%%只能修改系数
VarOfProc2=1e-4*[1;2;1];
VarOfMear=[VarOfMear1;VarOfMear2];
VarOfProc=[VarOfProc1;VarOfProc2];
%%
%Initial condition
A=eye(3);
AAA=A;
q=[0;0;0;1];
w=1*[1;-0.2;-0.05];
R=[0;0;0];
v=[0;0;0];
wN=w;
qN=q;
%%
%滤波初始化
i=0;
Z=zeros(7,1);
Xin=[0.2,0.3,0.1,sqrt(0.86),0.1,0.02,0.3]';
Pin=eye(7);
consZ=zeros(7,length(0:d_time:TotalTime));
consXReal=zeros(7,length(0:d_time:TotalTime));
consXout=zeros(7,length(0:d_time:TotalTime));%分别为，观测值、真实值、估计值
%UKF滤波固定参数
n=7;
alpha=1;
lamda=3*alpha^2-n;
beta=2;
gama=sqrt(n+lamda);
%构造
ITemp=inv(inertia_t);
QTemp=ITemp*diag(VarOfProc(5:7))*ITemp';
Q=zeros(7,7);
Q(5:7,5:7)=QTemp;
RMea=diag(VarOfMear);
%%
%噪音生成器
ObserNoise=[sqrt(VarOfMear(1))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(2))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(3))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(4))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(5))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(6))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear(6))*randn(1,length(0:d_time:TotalTime))
      ];
ProcNoise=[sqrt(VarOfProc(5))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc(6))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc(7))*randn(1,length(0:d_time:TotalTime))];
%%
for time=0:d_time:TotalTime
    %%
    i=i+1;
    F=[0;0;0];
    TName=0.1*(0.5*sin(pi/5*time)+0.2*cos(pi/2*time)+0.1*sin(pi*time)).*[1;1;1];
    T=ProcNoise(:,i)+TName;
    R_touch_e=R;
    %状态方程
    [R,A,v,w,q]=f_dyn_U2( R,R_touch_e,A,v,w,q, F, T);
    %观测方程
    Z(1:4)=q(1:4)+ObserNoise(1:4,i); 
    Z(5:7)=w+ObserNoise(5:7,i);
    %%
    %UKF
    [Xout,Pout]=UKF(Xin,Pin,Z,TName,time);
    Pin=Pout;
    Xin=Xout;
    %%
    %存储
    consZ(:,i)=Z;
    consXReal(1:4,i)=q;
    consXReal(5:7,i)=w;
    consXout(:,i)=Xout;
end

figure
plot(0:d_time:(TotalTime),consZ(figureNumber,:),'r*',0:d_time:(TotalTime),consXReal(figureNumber,:),'b-',0:d_time:(TotalTime),consXout(figureNumber,:))%,0:d_time:(TotalTime),cons(:,3),'k*'
