function [RU,AU,vU,wU,qU]=f_dyn_U2 ( RU,R_touch_e,AU,vU,wU,qU, FU, TU)
%已知RU，AU，vU，wU的微分，求取他们的函数
%使用龙格库塔法进行计算
%注意利用四元数描述姿态
%注意校对目标的四元数与余弦矩阵
%注意，A的导数算法为AU*tilde（wU),与教科书完全不一样，实际证明，这个不够精确，原程序的精确。能够保证单位正交矩阵
%原程序的动力学方程描述在惯性空间中。将其转换为本体中更好一些。因此选用了上述的积分方法
%需要再次考虑四元数的计算方法,无法使用圈乘计算。
%%
global d_time
%1st step
[tmp_vdU,tmp_wdU]=f_dyn_U(RU,R_touch_e,AU,vU,wU,FU,TU);
k1_RU=d_time*vU;
kk1_AU = aw( wU ) * AU-AU;
k1_vU = d_time * tmp_vdU;
k1_wU = d_time * tmp_wdU;
k1_qU=(QuanMatrix(wU))*qU*d_time/2;
k1_AU=AU*tilde(wU)*d_time;
%2nd step
[tmp_vdU,tmp_wdU]=f_dyn_U(RU+k1_RU/2,R_touch_e,AU+k1_AU/2,vU+k1_vU/2,wU+k1_wU/2,FU,TU);
k2_RU=d_time*(vU+k1_vU/2);
kk2_AU=aw( (wU+k1_wU/2) )*AU-AU;
k2_vU=d_time*tmp_vdU;
k2_wU=d_time*tmp_wdU;
k2_qU=QuanMatrix(wU+k1_wU/2)*qU*d_time/2;
k2_AU=AU*tilde(wU+k1_wU/2)*d_time;
%3rd step
[tmp_vdU,tmp_wdU]=f_dyn_U(RU+k2_RU/2,R_touch_e,AU+k2_AU/2,vU+k2_vU/2,wU+k2_wU/2,FU,TU);
k3_RU=d_time*(vU+k2_vU/2);
kk3_AU=aw( (wU+k2_wU/2) )*AU-AU;
k3_vU=d_time*tmp_vdU;
k3_wU=d_time*tmp_wdU;
k3_qU=(QuanMatrix(wU+k2_wU/2))*qU*d_time/2;
k3_AU=AU*tilde(wU+k2_wU/2)*d_time;
%4th step
[tmp_vdU,tmp_wdU]=f_dyn_U(RU+k3_RU/2,R_touch_e,AU+k3_AU/2,vU+k3_vU/2,wU+k3_wU/2,FU,TU);
k4_RU=d_time*(vU+k3_vU/2);
kk4_AU=aw( (wU+k3_wU/2) )*AU-AU;
k4_vU=d_time*tmp_vdU;
k4_wU=d_time*tmp_wdU;
k4_qU=QuanMatrix(wU+k3_wU/2)*qU*d_time/2;
k4_AU=AU*tilde(wU+k3_wU/2)*d_time;
%计算
RU_next=RU+(k1_RU+2*k2_RU+2*k3_RU+k4_RU)/6;
kAU_next=AU+(kk1_AU+2*kk2_AU+2*kk3_AU+kk4_AU)/6;
vU_next=vU+(k1_vU+2*k2_vU+2*k3_vU+k4_vU)/6;
wU_next=wU+(k1_wU+2*k2_wU+2*k3_wU+k4_wU)/6;
qU_next=qU+(k1_qU+2*k2_qU+2*k3_qU+k4_qU)/6;
%qU_next=QuanMatrix(qU)*(k1_qU+2*k2_qU+2*k3_qU+k4_qU)/6;
%%
%qU_next=QuanMatrix(wU)*qU*d_time/2+qU;%%
qU_next=(quatnormalize(qU_next'))';
AAA=AU+(k1_AU+2*k2_AU+2*k3_AU+k4_AU)/6;
%Solution
RU = RU_next;
AU = kAU_next;
vU = vU_next;
wU = wU_next;
qU = qU_next;