%%
%实验证明，对w的滤波很失败，原因未知；
%q很完美，基本上没有误差，甚至不需要观测。很奇怪。有可能是交叉污染？？
%另外，归一化也可以看做一个观测条件。
%按理说，w自身的P与q有很强的相关性。这个可以稍微分析。
%可以加速度级的进行一次滤波，速度级二次，位置级三次，又有点像龙格库塔，积分器
%主要根据是，速度级没有加速度级的项。
%标号1为q相关项，标号2为w相关项。
%当误差太小，则影响q的PXX的正定性。而误差太大，则导致w无法识别 
%主要的问题是KPK太大，因此需要Q很大才能弥补。保持q的Pxx正定。
%%
clc
clear all
global d_time
global mt inertia_t
global i
global VarOfMear1 VarOfProc1 VarOfMear2 VarOfProc2
global n1 alpha1 beta1 lamda1 gama1 %提前固定好，最大程度减小计算量
global n2 alpha2 beta2 lamda2 gama2
global RMea1 Q1 RMea2 Q2
%%
%确定画图变量
figureNumber=7;
%%
%parameter
d_time=0.1;
mt=10;
TotalTime=50;
inertia_t=diag([1;1;1]);
VarOfMear1=1e-2*[1;1.5;2;1];
VarOfMear2=1e-4*[1.3;3;4];
VarOfProc1=0*[1;1;1;1];%%%只能修改系数
VarOfProc2=1e-6*[1;2;1];

%%
%Initial condition
%均为真实值
A=eye(3);
AAA=A;
q=[0;0;0;1];
w=1*[1;-0.2;-0.05];
R=[0;0;0];
v=[0;0;0];
wN=w;
qN=q;
%%
%滤波器设置
%状态变量初始化
i=0;
Z=zeros(7,1);
win=[0.1,0.02,0.3]';
qin=[0.2,0.3,0.1,sqrt(0.86)]';
Pin1=eye(4);
Pin2=eye(3);
%UKF滤波固定参数
n1=4;
n2=3;
alpha1=1;
lamda1=3*alpha1^2-n1;
beta1=2;
gama1=sqrt(n1+lamda1);
alpha2=1;
lamda2=3*alpha1^2-n2;
beta2=2;
gama2=sqrt(n2+lamda2);
%构造Q与R
Q1=diag(VarOfProc1);
ITemp=inv(inertia_t);
Q2=ITemp*diag(VarOfProc2)*ITemp';
RMea1=diag(VarOfMear1);
RMea2=diag(VarOfMear2);
%%
%噪音生成器
ObserNoise1=[sqrt(VarOfMear1(1))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear1(2))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear1(3))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear1(4))*randn(1,length(0:d_time:TotalTime))];
ObserNoise2=[sqrt(VarOfMear2(1))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear2(2))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfMear2(3))*randn(1,length(0:d_time:TotalTime))
      ];
ProcNoise2=[sqrt(VarOfProc2(1))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc2(2))*randn(1,length(0:d_time:TotalTime));
      sqrt(VarOfProc2(3))*randn(1,length(0:d_time:TotalTime))];
%%
%存储器
%分别为，观测值、真实值、估计值
consZ=zeros(7,length(0:d_time:TotalTime));
consXReal=zeros(7,length(0:d_time:TotalTime));
consXout=zeros(7,length(0:d_time:TotalTime));
%%
for time=0:d_time:TotalTime
    %%
    i=i+1;
    F=[0;0;0];
    TName=0.1*(0.5*sin(pi/5*time)+0.2*cos(pi/2*time)+0.1*sin(pi*time)).*[1;1;1];
    T=ProcNoise2(:,i)+TName;
    R_touch_e=R;
    %状态方程
    [R,A,v,w,q]=f_dyn_U2( R,R_touch_e,A,v,w,q, F, T);
    %观测方程
    Z(1:4)=q+ObserNoise1(:,i); %%%注意这个地方，是加性还是乘性误差
    Z(5:7)=w+ObserNoise2(:,i);
    %%
    %UKF1滤出w
    [wout,Pout2]=UKF2(win,Pin2,Z(5:7),TName,time);
    Pin2=Pout2;
    win=wout;
    %UKF1滤出q
    [qout,Pout1]=UKF1(qin,Pin1,Z(1:4),wout);
    qin=qout;
    Pin1=Pout1;
    %%
    %存储
    consZ(:,i)=Z;
    consXReal(1:4,i)=q;
    consXReal(5:7,i)=w;
    consXout(1:4,i)=qout;
    consXout(5:7,i)=wout;
end

figure
plot(0:d_time:(TotalTime),consZ(figureNumber,:),'r*',0:d_time:(TotalTime),consXReal(figureNumber,:),'b-',0:d_time:(TotalTime),consXout(figureNumber,:))%,0:d_time:(TotalTime),cons(:,3),'k*'